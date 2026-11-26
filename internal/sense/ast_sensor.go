// Package sense provides a System 1 + "Bullet Time" System 2 local sense-making
// engine for pi-go. It plugs into extension.ComposeAfterToolChain to:
//  1. Detect implicit state-machine desyncs in Go packages using go/ast
//  2. Query a local System 1 decision model (/v1/systemone) in ~15ms (0 output tokens)
//  3. Trigger "Bullet Time" AST state-coupling analysis when an edit mutates
//     coupled struct state fields asymmetrically or without a mutex guard
//  4. Gate internal/memory observation capture so ephemeral tool calls don't
//     flood the memory-compressor LLM subagent.
package sense

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// MethodTouch records how a single method interacts with its receiver struct's fields.
type MethodTouch struct {
	StructName string   `json:"struct_name"`
	MethodName string   `json:"method_name"`
	File       string   `json:"file"`
	Line       int      `json:"line"`
	Reads      []string `json:"reads"`
	Writes     []string `json:"writes"`
	Calls      []string `json:"calls,omitempty"`
	UsesLock   bool     `json:"uses_lock"`
}

// CouplingAnomaly describes an implicit state-machine hazard discovered in Bullet Time.
type CouplingAnomaly struct {
	Kind        string   `json:"kind"` // "MISSING_COUPLED_WRITE" | "UNLOCKED_STATE_WRITE"
	StructName  string   `json:"struct_name"`
	MethodName  string   `json:"method_name"`
	Field       string   `json:"field"`
	PeerField   string   `json:"peer_field,omitempty"`
	PeerMethods []string `json:"peer_methods,omitempty"`
	Summary     string   `json:"summary"`
}

// PackageStateGraph holds the extracted implicit state machine of a Go package.
type PackageStateGraph struct {
	PackageName string                 `json:"package_name"`
	StructFields map[string][]string   `json:"struct_fields"`
	LockFields   map[string][]string   `json:"lock_fields"`
	Methods      []MethodTouch         `json:"methods"`
	Anomalies    []CouplingAnomaly     `json:"anomalies"`
}

// AnalyzePackageDir parses all non-test Go files in dir and extracts the
// implicit state-machine coupling graph across struct fields and methods.
func AnalyzePackageDir(dir string, focusFile string) (*PackageStateGraph, error) {
	fset := token.NewFileSet()
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	var files []*ast.File
	var fileNames []string
	pkgName := ""

	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		fullPath := filepath.Join(dir, name)
		f, parseErr := parser.ParseFile(fset, fullPath, nil, 0)
		if parseErr != nil {
			continue
		}
		if pkgName == "" {
			pkgName = f.Name.Name
		}
		files = append(files, f)
		fileNames = append(fileNames, name)
	}

	graph := &PackageStateGraph{
		PackageName:  pkgName,
		StructFields: make(map[string][]string),
		LockFields:   make(map[string][]string),
	}

	// Pass 1: Collect struct definitions and their state/mutex fields.
	for _, f := range files {
		for _, decl := range f.Decls {
			gen, ok := decl.(*ast.GenDecl)
			if !ok || gen.Tok != token.TYPE {
				continue
			}
			for _, spec := range gen.Specs {
				ts, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}
				st, ok := ts.Type.(*ast.StructType)
				if !ok || st.Fields == nil {
					continue
				}
				sName := ts.Name.Name
				for _, field := range st.Fields.List {
					typeStr := exprTypeString(field.Type)
					isLock := strings.Contains(typeStr, "Mutex") || strings.Contains(typeStr, "RWMutex")
					for _, id := range field.Names {
						graph.StructFields[sName] = append(graph.StructFields[sName], id.Name)
						if isLock {
							graph.LockFields[sName] = append(graph.LockFields[sName], id.Name)
						}
					}
				}
			}
		}
	}

	// Pass 2: Inspect every method on known structs for field reads, writes, and lock calls.
	for idx, f := range files {
		fName := fileNames[idx]
		for _, decl := range f.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv == nil || len(fn.Recv.List) == 0 || fn.Body == nil {
				continue
			}
			recv := fn.Recv.List[0]
			structName := receiverTypeName(recv.Type)
			if _, known := graph.StructFields[structName]; !known {
				continue
			}
			if len(recv.Names) == 0 {
				continue
			}
			recvVar := recv.Names[0].Name
			knownFields := make(map[string]bool)
			for _, fld := range graph.StructFields[structName] {
				knownFields[fld] = true
			}
			lockFields := make(map[string]bool)
			for _, lfld := range graph.LockFields[structName] {
				lockFields[lfld] = true
			}

			reads := make(map[string]bool)
			writes := make(map[string]bool)
			usesLock := false

			// Walk AST statements in method body.
			ast.Inspect(fn.Body, func(n ast.Node) bool {
				switch stmt := n.(type) {
				case *ast.AssignStmt:
					for _, lhs := range stmt.Lhs {
						if fld := rootReceiverField(lhs, recvVar, knownFields); fld != "" && !lockFields[fld] {
							writes[fld] = true
						}
					}
					for _, rhs := range stmt.Rhs {
						collectReceiverReads(rhs, recvVar, knownFields, lockFields, reads)
					}
					return false
				case *ast.IncDecStmt:
					if fld := rootReceiverField(stmt.X, recvVar, knownFields); fld != "" && !lockFields[fld] {
						writes[fld] = true
					}
					return false
				case *ast.CallExpr:
					// Check for t.mu.Lock() / RLock() or delete(t.mapField, ...)
					if sel, ok := stmt.Fun.(*ast.SelectorExpr); ok {
						mName := sel.Sel.Name
						if (mName == "Lock" || mName == "RLock") && rootReceiverField(sel.X, recvVar, lockFields) != "" {
							usesLock = true
						}
					}
					if ident, ok := stmt.Fun.(*ast.Ident); ok && ident.Name == "delete" && len(stmt.Args) > 0 {
						if fld := rootReceiverField(stmt.Args[0], recvVar, knownFields); fld != "" {
							writes[fld] = true
						}
					}
				case *ast.SelectorExpr:
					if fld := rootReceiverField(stmt, recvVar, knownFields); fld != "" && !lockFields[fld] {
						reads[fld] = true
					}
				}
				return true
			})

			pos := fset.Position(fn.Pos())
			graph.Methods = append(graph.Methods, MethodTouch{
				StructName: structName,
				MethodName: fn.Name.Name,
				File:       fName,
				Line:       pos.Line,
				Reads:      sortedKeys(reads),
				Writes:     sortedKeys(writes),
				UsesLock:   usesLock,
			})
		}
	}

	graph.Anomalies = detectCouplingAnomalies(graph, filepath.Base(focusFile))
	return graph, nil
}

// expandIntraClassCalls propagates 1-hop intra-class helper calls (e.g. Bot.task()
// calling self.watch() or self.close() in taskbot.py) so coupled state mutations
// factored into private/helper methods are visible at the caller level.
func expandIntraClassCalls(g *PackageStateGraph) {
	lookup := make(map[[2]string]MethodTouch)
	for _, m := range g.Methods {
		lookup[[2]string{m.StructName, m.MethodName}] = m
	}
	for i := range g.Methods {
		m := &g.Methods[i]
		if len(m.Calls) == 0 {
			continue
		}
		rSet := make(map[string]bool)
		wSet := make(map[string]bool)
		for _, r := range m.Reads {
			rSet[r] = true
		}
		for _, w := range m.Writes {
			wSet[w] = true
		}
		for _, calleeName := range m.Calls {
			if callee, ok := lookup[[2]string{m.StructName, calleeName}]; ok {
				for _, r := range callee.Reads {
					rSet[r] = true
				}
				for _, w := range callee.Writes {
					wSet[w] = true
				}
			}
		}
		m.Reads = sortedKeys(rSet)
		m.Writes = sortedKeys(wSet)
	}
}

// detectCouplingAnomalies discovers implicit state-machine hazards:
//  1. UNLOCKED_STATE_WRITE: A method writes a struct field that ≥2 other methods
//     only write under a mutex lock, but this method does not acquire the lock.
//  2. MISSING_COUPLED_WRITE: Field A is co-mutated with Field B in ≥2 methods
//     (and co-read in ≥1 method, forming a coupled state invariant like
//     lastPromptTokens <-> cachePrefixTokens), but a method mutates A without B.
func detectCouplingAnomalies(g *PackageStateGraph, focusBase string) []CouplingAnomaly {
	expandIntraClassCalls(g)
	var anomalies []CouplingAnomaly

	// Group methods by struct
	byStruct := make(map[string][]MethodTouch)
	for _, m := range g.Methods {
		byStruct[m.StructName] = append(byStruct[m.StructName], m)
	}

	for sName, methods := range byStruct {
		hasLocks := len(g.LockFields[sName]) > 0

		// Count how often each field is written with vs without a lock,
		// and how often pairs of fields (A, B) are co-written or co-read.
		lockedWrites := make(map[string]int)
		coWrite := make(map[[2]string][]string)
		coRead := make(map[[2]string]int)

		for _, m := range methods {
			for _, w := range m.Writes {
				if m.UsesLock {
					lockedWrites[w]++
				}
				for _, w2 := range m.Writes {
					if w != w2 {
						coWrite[[2]string{w, w2}] = append(coWrite[[2]string{w, w2}], m.MethodName)
					}
				}
			}
			allRead := make(map[string]bool)
			for _, r := range m.Reads {
				allRead[r] = true
			}
			for _, w := range m.Writes {
				allRead[w] = true
			}
			rList := sortedKeys(allRead)
			for _, r1 := range rList {
				for _, r2 := range rList {
					if r1 != r2 {
						coRead[[2]string{r1, r2}]++
					}
				}
			}
		}

		for _, m := range methods {
			if focusBase != "" && m.File != focusBase {
				continue
			}
			// Ignore constructors / initializers like New* or load()
			if strings.HasPrefix(m.MethodName, "New") || m.MethodName == "load" || m.MethodName == "ensureToday" {
				continue
			}

			writeSet := make(map[string]bool)
			for _, w := range m.Writes {
				writeSet[w] = true
			}

			// Check 1: Unlocked write to a mutex-protected state field
			if hasLocks && !m.UsesLock && len(m.Writes) > 0 {
				for _, w := range m.Writes {
					if lockedWrites[w] >= 2 {
						anomalies = append(anomalies, CouplingAnomaly{
							Kind:       "UNLOCKED_STATE_WRITE",
							StructName: sName,
							MethodName: m.MethodName,
							Field:      w,
							Summary: fmt.Sprintf(
								"%s.%s (%s:%d) mutates mutex-guarded state field %q without acquiring %s.%s",
								sName, m.MethodName, m.File, m.Line, w, sName, g.LockFields[sName][0],
							),
						})
					}
				}
			}

			// Check 2: Asymmetric mutation of coupled state fields (Implicit State Desync)
			for _, w := range m.Writes {
				for _, peer := range g.StructFields[sName] {
					if w == peer || writeSet[peer] {
						continue
					}
					peers := coWrite[[2]string{w, peer}]
					if len(peers) >= 2 && coRead[[2]string{w, peer}] >= 2 {
						anomalies = append(anomalies, CouplingAnomaly{
							Kind:        "MISSING_COUPLED_WRITE",
							StructName:  sName,
							MethodName:  m.MethodName,
							Field:       w,
							PeerField:   peer,
							PeerMethods: peers,
							Summary: fmt.Sprintf(
								"Implicit State Desync: %s.%s (%s:%d) mutates %q without updating coupled field %q (co-mutated in %s)",
								sName, m.MethodName, m.File, m.Line, w, peer, strings.Join(peers, ", "),
							),
						})
					}
				}
			}
		}
	}
	return anomalies
}

func receiverTypeName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.StarExpr:
		return receiverTypeName(t.X)
	case *ast.Ident:
		return t.Name
	default:
		return ""
	}
}

func exprTypeString(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.SelectorExpr:
		return exprTypeString(t.X) + "." + t.Sel.Name
	case *ast.StarExpr:
		return "*" + exprTypeString(t.X)
	default:
		return ""
	}
}

// rootReceiverField unwraps chained selectors like `t.usage.InputTokens` or `t.lastPromptTokens`
// and returns the immediate struct field on `recvVar` (`usage` or `lastPromptTokens`).
func rootReceiverField(expr ast.Expr, recvVar string, validFields map[string]bool) string {
	switch e := expr.(type) {
	case *ast.SelectorExpr:
		if id, ok := e.X.(*ast.Ident); ok && id.Name == recvVar {
			if validFields[e.Sel.Name] {
				return e.Sel.Name
			}
			return ""
		}
		return rootReceiverField(e.X, recvVar, validFields)
	case *ast.IndexExpr:
		return rootReceiverField(e.X, recvVar, validFields)
	default:
		return ""
	}
}

func collectReceiverReads(expr ast.Expr, recvVar string, knownFields, lockFields map[string]bool, out map[string]bool) {
	ast.Inspect(expr, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if fld := rootReceiverField(sel, recvVar, knownFields); fld != "" && !lockFields[fld] {
				out[fld] = true
			}
		}
		return true
	})
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
