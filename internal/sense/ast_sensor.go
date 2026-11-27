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
	StructName     string   `json:"struct_name"`
	MethodName     string   `json:"method_name"`
	File           string   `json:"file"`
	Line           int      `json:"line"`
	Reads          []string `json:"reads"`
	Writes         []string `json:"writes"`
	UnlockedWrites []string `json:"unlocked_writes,omitempty"`
	Calls          []string `json:"calls,omitempty"`
	UsesLock       bool     `json:"uses_lock"`
	IsPrivate      bool     `json:"is_private,omitempty"`
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
					if len(field.Names) == 0 && isLock {
						emb := embeddedLockName(field.Type)
						if emb != "" {
							graph.LockFields[sName] = append(graph.LockFields[sName], emb)
						}
						continue
					}
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
			calls := make(map[string]bool)
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
						collectReceiverReadsAndCalls(rhs, recvVar, knownFields, lockFields, reads, calls)
					}
					return false
				case *ast.IncDecStmt:
					if fld := rootReceiverField(stmt.X, recvVar, knownFields); fld != "" && !lockFields[fld] {
						writes[fld] = true
					}
					return false
				case *ast.CallExpr:
					// Check for t.mu.Lock() / RLock(), embedded t.Lock(), intra-struct helper calls, or delete(t.mapField, ...)
					if sel, ok := stmt.Fun.(*ast.SelectorExpr); ok {
						mName := sel.Sel.Name
						if mName == "Lock" || mName == "RLock" {
							if rootReceiverField(sel.X, recvVar, lockFields) != "" {
								usesLock = true
							} else if id, ok := sel.X.(*ast.Ident); ok && id.Name == recvVar && len(lockFields) > 0 {
								usesLock = true
							}
						} else if mName != "Unlock" && mName != "RUnlock" && !ast.IsExported(mName) {
							if id, ok := sel.X.(*ast.Ident); ok && id.Name == recvVar && !knownFields[mName] && !lockFields[mName] {
								calls[mName] = true
							}
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
				Calls:      sortedKeys(calls),
				UsesLock:   usesLock,
				IsPrivate:  !ast.IsExported(fn.Name.Name),
			})
		}
	}

	graph.Anomalies = detectCouplingAnomalies(graph, cleanFocusBase(focusFile))
	return graph, nil
}

func cleanFocusBase(focusFile string) string {
	if strings.TrimSpace(focusFile) == "" {
		return ""
	}
	return filepath.Base(focusFile)
}

// expandIntraClassCalls propagates 1-hop intra-class helper calls while preserving
// per-field lock provenance: if a caller delegates to a helper that acquires the lock
// internally (e.g. Kubernetes waitingPod.Reject -> stopWithStatus, Kafka FetchBuffer.add -> addAll,
// or Hermes _kill_stale_stream -> _cancel_current_stream_attempt), those callee-locked
// writes are NOT marked as unlocked on the caller.
func expandIntraClassCalls(g *PackageStateGraph) {
	// Initialize direct lock provenance before 1-hop expansion
	for i := range g.Methods {
		m := &g.Methods[i]
		if m.UnlockedWrites == nil && !m.UsesLock && len(m.Writes) > 0 {
			m.UnlockedWrites = append([]string(nil), m.Writes...)
		}
	}

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
		uSet := make(map[string]bool)
		for _, r := range m.Reads {
			rSet[r] = true
		}
		for _, w := range m.Writes {
			wSet[w] = true
		}
		if !m.UsesLock {
			for _, u := range m.UnlockedWrites {
				uSet[u] = true
			}
		}
		for _, calleeName := range m.Calls {
			if calleeName == "ensureToday" || calleeName == "load" || calleeName == "init" {
				continue
			}
			if callee, ok := lookup[[2]string{m.StructName, calleeName}]; ok {
				for _, r := range callee.Reads {
					rSet[r] = true
				}
				for _, w := range callee.Writes {
					wSet[w] = true
				}
				if !m.UsesLock && !callee.UsesLock {
					for _, u := range callee.UnlockedWrites {
						uSet[u] = true
					}
				}
			}
		}
		m.Reads = sortedKeys(rSet)
		m.Writes = sortedKeys(wSet)
		if m.UsesLock {
			m.UnlockedWrites = nil
		} else {
			m.UnlockedWrites = sortedKeys(uSet)
		}
	}
}

// detectCouplingAnomalies discovers implicit state-machine hazards:
//  1. UNLOCKED_STATE_WRITE: A method writes a struct field that ≥2 other methods
//     only write under a mutex lock, but this method does not acquire the lock.
//  2. MISSING_COUPLED_WRITE: Field A is co-mutated with Field B in ≥2 methods
//     (and co-read in ≥1 method, forming a coupled state invariant like
//     lastPromptTokens <-> cachePrefixTokens), but a method mutates A without B.
func detectCouplingAnomalies(g *PackageStateGraph, focusBase string) []CouplingAnomaly {
	// Track intra-class callers before expanding helper calls: private sub-step helpers
	// called by intra-class methods have their reads/writes audited at the caller level
	// (e.g. Kubernetes threadSafeMap.updateLocked or Kafka BufferPool.freeUp).
	intraCallers := make(map[[2]string]int)
	for _, m := range g.Methods {
		for _, c := range m.Calls {
			intraCallers[[2]string{m.StructName, c}]++
		}
	}

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
			if isAssumedLockedOrInitMethod(m.MethodName) {
				// Avoid counting *Locked/*Unlocked sub-helpers twice alongside their callers
				continue
			}
			unlockedSet := make(map[string]bool)
			for _, u := range m.UnlockedWrites {
				unlockedSet[u] = true
			}
			for _, w := range m.Writes {
				if !unlockedSet[w] {
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
			// Ignore constructors, initializers, explicit *Locked/*Unlocked caller-contract helpers,
			// and private sub-step helpers already audited via their intra-class callers.
			if isAssumedLockedOrInitMethod(m.MethodName) {
				continue
			}
			if m.IsPrivate && intraCallers[[2]string{sName, m.MethodName}] > 0 {
				continue
			}

			writeSet := make(map[string]bool)
			for _, w := range m.Writes {
				writeSet[w] = true
			}

			// Check 1: Unlocked write to a mutex-protected state field (using per-field lock provenance)
			if hasLocks && len(m.UnlockedWrites) > 0 {
				for _, w := range m.UnlockedWrites {
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

func collectReceiverReadsAndCalls(expr ast.Expr, recvVar string, knownFields, lockFields map[string]bool, outReads, outCalls map[string]bool) {
	ast.Inspect(expr, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.CallExpr:
			if sel, ok := node.Fun.(*ast.SelectorExpr); ok {
				mName := sel.Sel.Name
				if mName != "Lock" && mName != "Unlock" && mName != "RLock" && mName != "RUnlock" && !ast.IsExported(mName) {
					if id, ok := sel.X.(*ast.Ident); ok && id.Name == recvVar && !knownFields[mName] && !lockFields[mName] {
						outCalls[mName] = true
					}
				}
			}
		case *ast.SelectorExpr:
			if fld := rootReceiverField(node, recvVar, knownFields); fld != "" && !lockFields[fld] {
				outReads[fld] = true
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

func embeddedLockName(expr ast.Expr) string {
	switch t := expr.(type) {
	case *ast.Ident:
		return t.Name
	case *ast.SelectorExpr:
		return t.Sel.Name
	case *ast.StarExpr:
		return embeddedLockName(t.X)
	default:
		return ""
	}
}

func isAssumedLockedOrInitMethod(name string) bool {
	if strings.HasPrefix(name, "New") || strings.HasPrefix(name, "new") ||
		name == "__init__" || name == "constructor" || name == "apply" ||
		name == "load" || name == "init" || name == "ensureToday" ||
		name == "String" || name == "toString" || name == "hashCode" || name == "equals" {
		return true
	}
	lower := strings.ToLower(name)
	return strings.HasSuffix(lower, "locked") ||
		strings.HasSuffix(lower, "unlocked") ||
		strings.HasSuffix(lower, "nolock") ||
		strings.HasSuffix(lower, "unsynchronized")
}
