package sense

import (
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// BlastImpactLevel classifies the cross-file scope of an edit.
const (
	BlastLevelLocal        = "LOCAL"         // 0 external files affected
	BlastLevelPackage      = "PACKAGE"       // 1+ sibling files in the same directory/package
	BlastLevelCrossPackage = "CROSS_PACKAGE" // 1+ files across sibling Go packages or Python modules
	BlastLevelCrossService = "CROSS_SERVICE" // 1+ wire-contract consumers across separate services/containers
)

// ImpactConsumer records one downstream caller, duck-typed interface, or
// wire-contract consumer outside the edited file.
type ImpactConsumer struct {
	Kind         string `json:"kind"`          // "SYMBOL_CALLER" | "WIRE_CONTRACT_CONSUMER"
	Token        string `json:"token"`         // function/method name or wire-contract key
	RelPath      string `json:"rel_path"`      // path relative to search root
	Line         int    `json:"line"`          // 1-based line number
	Snippet      string `json:"snippet"`       // trimmed source line
	CrossPackage bool   `json:"cross_package"` // true if in a different directory than focusFile
}

// BlastRadiusReport summarizes the downstream impact radius of an edit.
type BlastRadiusReport struct {
	Level               string           `json:"level"`
	SearchRoot          string           `json:"search_root"`
	TouchedSymbols      []string         `json:"touched_symbols,omitempty"`
	TouchedContractKeys []string         `json:"touched_contract_keys,omitempty"`
	ExternalFilesCount  int              `json:"external_files_count"`
	ExternalDirsCount   int              `json:"external_dirs_count"`
	UnsoundnessDetected bool             `json:"unsoundness_detected,omitempty"`
	UnsoundnessReasons  []string         `json:"unsoundness_reasons,omitempty"`
	Consumers           []ImpactConsumer `json:"consumers,omitempty"`
	Summary             string           `json:"summary,omitempty"`
}

var (
	goFuncHeaderRe  = regexp.MustCompile(`^func\s+(?:\([^)]+\)\s*)?([A-Z][A-Za-z0-9_]*)\s*\(`)
	goTypeHeaderRe  = regexp.MustCompile(`^type\s+([A-Z][A-Za-z0-9_]*)\s+(?:struct|interface)\b`)
	pyDefHeaderRe   = regexp.MustCompile(`^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	pyClassHeaderRe = regexp.MustCompile(`^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b`)
	stringLitKeyRe  = regexp.MustCompile(`["']([a-zA-Z][a-zA-Z0-9_./-]{3,48})["']`)
	goUnsoundRe     = regexp.MustCompile(`\b(?:reflect\.(?:ValueOf|TypeOf)|MethodByName|unsafe\.Pointer)\b|\.\((?:any|interface\{\})\)`)
	pyUnsoundRe     = regexp.MustCompile(`\b(?:getattr|setattr|__getattr__|__call__|eval|exec|importlib)\s*\(|\*\*kwargs\b`)
)

var ignoredGenericMethodNames = map[string]bool{
	"__init__": true, "main": true, "get": true, "set": true, "add": true,
	"read": true, "write": true, "close": true, "run": true, "status": true,
	"send": true, "say": true, "log": true, "parse": true,
}

var ignoredContractLiterals = map[string]bool{
	"true": true, "false": true, "null": true, "none": true,
	"application/json": true, "text/plain": true, "text/event-stream": true,
	"content-type": true, "content-length": true, "authorization": true,
	"utf-8": true, "http://": true, "https://": true,
}

// AddSearchRoot registers an additional root directory (e.g. a repo's internal/
// or apps/ tree) for the Cross-File Blast-Radius Scanner to sweep when a file
// is edited.
func (e *Engine) AddSearchRoot(root string) {
	if root == "" {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, r := range e.extraSearchRoots {
		if r == root {
			return
		}
	}
	e.extraSearchRoots = append(e.extraSearchRoots, root)
}

// ScanBlastRadius analyzes an edited file (and the edit snippet in args) to:
//  1. Identify the enclosing or modified Go/Python symbols (functions, methods, types)
//  2. Extract modified wire-contract dictionary keys / Kubernetes label keys
//  3. Sweep sibling packages and services to locate every downstream caller,
//     duck-typed interface definition, or cross-service JSON/K8s contract consumer.
func (e *Engine) ScanBlastRadius(filePath string, args map[string]any) *BlastRadiusReport {
	if filePath == "" {
		return nil
	}
	rawBytes, err := os.ReadFile(filePath)
	if err != nil {
		return nil
	}
	content := string(rawBytes)

	oldStr, _ := args["old_string"].(string)
	newStr, _ := args["new_string"].(string)

	unsoundReasons := detectStaticUnsoundness(filePath, content+"\n"+oldStr+"\n"+newStr)
	symbols, contractKeys := extractTouchedSymbolsAndContracts(filePath, content, oldStr, newStr)
	if len(symbols) == 0 && len(contractKeys) == 0 && len(unsoundReasons) == 0 {
		return nil
	}

	roots := e.resolveSearchRoots(filePath, args)
	if len(roots) == 0 {
		return nil
	}

	absFocus, _ := filepath.Abs(filePath)
	focusBase := filepath.Base(filePath)
	focusDir := filepath.Dir(absFocus)

	var consumers []ImpactConsumer
	extFiles := make(map[string]bool)
	extDirs := make(map[string]bool)
	hasWireCrossService := false
	hasCrossPkg := false

	// Build word-boundary regexes for touched symbols
	symMatchers := make(map[string]*regexp.Regexp, len(symbols))
	for _, sym := range symbols {
		symMatchers[sym] = regexp.MustCompile(`\b` + regexp.QuoteMeta(sym) + `\b`)
	}

	primaryRoot := roots[0]
	for _, root := range roots {
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			name := d.Name()
			if d.IsDir() {
				if name == ".git" || name == "vendor" || name == "node_modules" || name == "__pycache__" || name == ".venv" {
					return filepath.SkipDir
				}
				return nil
			}
			// Skip test files and non-code files so the blast radius focuses on real production consumers
			if strings.HasSuffix(name, "_test.go") || strings.HasPrefix(name, "test_") || strings.HasSuffix(name, "_test.py") {
				return nil
			}
			ext := filepath.Ext(name)
			if ext != ".go" && ext != ".py" && ext != ".sh" {
				return nil
			}
			absP, _ := filepath.Abs(p)
			if absP == absFocus || name == focusBase {
				return nil
			}

			blob, rErr := os.ReadFile(p)
			if rErr != nil {
				return nil
			}
			text := string(blob)

			// Quick substring pre-filter before splitting lines
			matchedAny := false
			for _, sym := range symbols {
				if strings.Contains(text, sym) {
					matchedAny = true
					break
				}
			}
			if !matchedAny {
				for _, ck := range contractKeys {
					if strings.Contains(text, ck) {
						matchedAny = true
						break
					}
				}
			}
			if !matchedAny {
				return nil
			}

			rel, relErr := filepath.Rel(root, p)
			if relErr != nil {
				rel = p
			}
			isCrossDir := filepath.Dir(absP) != focusDir

			lines := strings.Split(text, "\n")
			for idx, rawLine := range lines {
				trimmed := strings.TrimSpace(rawLine)
				if trimmed == "" || strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") {
					continue
				}

				for _, sym := range symbols {
					re := symMatchers[sym]
					if re != nil && re.MatchString(trimmed) {
						consumers = append(consumers, ImpactConsumer{
							Kind:         "SYMBOL_CALLER",
							Token:        sym,
							RelPath:      rel,
							Line:         idx + 1,
							Snippet:      truncateSnippet(trimmed, 96),
							CrossPackage: isCrossDir,
						})
						extFiles[rel] = true
						if isCrossDir {
							extDirs[filepath.Dir(rel)] = true
							hasCrossPkg = true
						}
					}
				}

				for _, ck := range contractKeys {
					if strings.Contains(trimmed, `"`+ck+`"`) || strings.Contains(trimmed, `'`+ck+`'`) {
						consumers = append(consumers, ImpactConsumer{
							Kind:         "WIRE_CONTRACT_CONSUMER",
							Token:        ck,
							RelPath:      rel,
							Line:         idx + 1,
							Snippet:      truncateSnippet(trimmed, 96),
							CrossPackage: isCrossDir,
						})
						extFiles[rel] = true
						if isCrossDir {
							extDirs[filepath.Dir(rel)] = true
							hasWireCrossService = true
						}
					}
				}
			}
			return nil
		})
	}

	level := BlastLevelLocal
	switch {
	case hasWireCrossService:
		level = BlastLevelCrossService
	case hasCrossPkg:
		level = BlastLevelCrossPackage
	case len(extFiles) > 0:
		level = BlastLevelPackage
	}

	report := &BlastRadiusReport{
		Level:               level,
		SearchRoot:          primaryRoot,
		TouchedSymbols:      symbols,
		TouchedContractKeys: contractKeys,
		ExternalFilesCount:  len(extFiles),
		ExternalDirsCount:   len(extDirs),
		UnsoundnessDetected: len(unsoundReasons) > 0,
		UnsoundnessReasons:  unsoundReasons,
		Consumers:           dedupAndCapConsumers(consumers, 10),
	}
	report.Summary = formatBlastRadiusSummary(filePath, report)
	return report
}

func detectStaticUnsoundness(filePath, text string) []string {
	reasons := make(map[string]bool)
	if strings.HasSuffix(filePath, ".go") {
		for _, m := range goUnsoundRe.FindAllString(text, -1) {
			reasons[m] = true
		}
	} else if strings.HasSuffix(filePath, ".py") {
		for _, m := range pyUnsoundRe.FindAllString(text, -1) {
			reasons[strings.TrimSpace(strings.TrimSuffix(m, "("))] = true
		}
	}
	return sortedKeys(reasons)
}

func (e *Engine) resolveSearchRoots(filePath string, args map[string]any) []string {
	var roots []string
	seen := make(map[string]bool)
	add := func(r string) {
		if r == "" {
			return
		}
		if info, err := os.Stat(r); err == nil && info.IsDir() {
			abs, _ := filepath.Abs(r)
			if !seen[abs] {
				seen[abs] = true
				roots = append(roots, abs)
			}
		}
	}

	if sr, ok := args["search_root"].(string); ok && sr != "" {
		add(sr)
		return roots
	}
	e.mu.Lock()
	extra := append([]string(nil), e.extraSearchRoots...)
	e.mu.Unlock()
	for _, r := range extra {
		add(r)
	}

	// Walk upward from filePath up to 4 levels looking for internal/, apps/, or go.mod
	cur := filepath.Dir(filePath)
	for depth := 0; depth < 4; depth++ {
		base := filepath.Base(cur)
		if base == "internal" || base == "apps" {
			add(cur)
			break
		}
		if _, err := os.Stat(filepath.Join(cur, "go.mod")); err == nil {
			add(cur)
			break
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	if len(roots) == 0 {
		add(filepath.Dir(filePath))
	}
	return roots
}

// extractTouchedSymbolsAndContracts determines which Go/Python functions, methods,
// or types and which wire-contract keys are touched by an edit (or defined in the file).
func extractTouchedSymbolsAndContracts(filePath, content, oldStr, newStr string) (symbols []string, contracts []string) {
	symSet := make(map[string]bool)
	conSet := make(map[string]bool)

	snippet := strings.TrimSpace(newStr)
	if snippet == "" {
		snippet = strings.TrimSpace(oldStr)
	}

	isGo := strings.HasSuffix(filePath, ".go")
	isPy := strings.HasSuffix(filePath, ".py")

	// 1. Find enclosing function/method/type around the edited region in content
	var activeBlock string
	if snippet != "" {
		firstLine := strings.TrimSpace(strings.Split(snippet, "\n")[0])
		idx := strings.Index(content, snippet)
		if idx < 0 && firstLine != "" {
			idx = strings.Index(content, firstLine)
		}
		if idx < 0 && oldStr != "" {
			oldTrim := strings.TrimSpace(oldStr)
			idx = strings.Index(content, oldTrim)
			if idx < 0 {
				oldFirst := strings.TrimSpace(strings.Split(oldTrim, "\n")[0])
				if oldFirst != "" {
					idx = strings.Index(content, oldFirst)
				}
			}
		}
		if idx >= 0 {
			lineNum := strings.Count(content[:idx], "\n")
			allLines := strings.Split(content, "\n")
			// Walk upward from lineNum to find enclosing func / def header
			for i := lineNum; i >= 0; i-- {
				l := allLines[i]
				if isGo {
					if m := goFuncHeaderRe.FindStringSubmatch(strings.TrimSpace(l)); m != nil {
						symSet[m[1]] = true
						end := min(len(allLines), i+15)
						activeBlock = strings.Join(allLines[i:end], "\n")
						break
					}
					if m := goTypeHeaderRe.FindStringSubmatch(strings.TrimSpace(l)); m != nil {
						symSet[m[1]] = true
						break
					}
				} else if isPy {
					if m := pyDefHeaderRe.FindStringSubmatch(l); m != nil {
						if !ignoredGenericMethodNames[m[1]] && !strings.HasPrefix(m[1], "_") {
							symSet[m[1]] = true
						}
						end := min(len(allLines), i+15)
						activeBlock = strings.Join(allLines[i:end], "\n")
						break
					}
				}
			}
		}
	}

	// 2. Also scan oldStr + newStr directly for func/def headers and contract keys
	combined := oldStr + "\n" + newStr + "\n" + activeBlock
	for _, line := range strings.Split(combined, "\n") {
		if isGo {
			if m := goFuncHeaderRe.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
				symSet[m[1]] = true
			}
			if m := goTypeHeaderRe.FindStringSubmatch(strings.TrimSpace(line)); m != nil {
				symSet[m[1]] = true
			}
		} else if isPy {
			if m := pyDefHeaderRe.FindStringSubmatch(line); m != nil && !ignoredGenericMethodNames[m[1]] && !strings.HasPrefix(m[1], "_") {
				symSet[m[1]] = true
			}
			if m := pyClassHeaderRe.FindStringSubmatch(line); m != nil {
				symSet[m[1]] = true
			}
		}

		for _, sm := range stringLitKeyRe.FindAllStringSubmatch(line, -1) {
			lit := sm[1]
			if isWireContractKey(lit) {
				conSet[lit] = true
			}
		}
	}

	return sortedKeys(symSet), sortedKeys(conSet)
}

func isWireContractKey(s string) bool {
	lower := strings.ToLower(s)
	if ignoredContractLiterals[lower] {
		return false
	}
	// Wire-contract keys in portal and pi-go are snake_case ("task_budget_usd",
	// "over_budget", "task_result", "lsp_diagnostics") or domain-prefixed K8s
	// labels/annotations ("portal.dev/task", "portal.dev/requested-by").
	if strings.Contains(s, "portal.dev/") {
		return true
	}
	if strings.Contains(s, "_") && !strings.HasPrefix(s, "_") && !strings.HasSuffix(s, ".go") && !strings.HasSuffix(s, ".py") {
		return true
	}
	return false
}

func dedupAndCapConsumers(in []ImpactConsumer, limit int) []ImpactConsumer {
	// Prioritize cross-package / cross-service consumers first
	sort.SliceStable(in, func(i, j int) bool {
		if in[i].CrossPackage != in[j].CrossPackage {
			return in[i].CrossPackage
		}
		if in[i].Kind != in[j].Kind {
			return in[i].Kind == "WIRE_CONTRACT_CONSUMER"
		}
		if in[i].RelPath != in[j].RelPath {
			return in[i].RelPath < in[j].RelPath
		}
		return in[i].Line < in[j].Line
	})

	// Coalesce multiple tokens matched on the same (Kind, RelPath, Line)
	indexByLine := make(map[string]int)
	var out []ImpactConsumer
	for _, c := range in {
		lineKey := fmt.Sprintf("%s:%s:%d", c.Kind, c.RelPath, c.Line)
		if idx, exists := indexByLine[lineKey]; exists {
			if !strings.Contains(out[idx].Token, c.Token) {
				out[idx].Token += ", " + c.Token
			}
			continue
		}
		if len(out) >= limit {
			continue
		}
		indexByLine[lineKey] = len(out)
		out = append(out, c)
	}
	return out
}

func formatBlastRadiusSummary(filePath string, r *BlastRadiusReport) string {
	if r == nil || (len(r.Consumers) == 0 && !r.UnsoundnessDetected) {
		return ""
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf(
		"[S3 BLAST-RADIUS SCANNER · %s (%d external file(s) across %d sibling package(s)/service(s))]\n",
		r.Level, r.ExternalFilesCount, max(1, r.ExternalDirsCount),
	))
	if r.UnsoundnessDetected {
		b.WriteString(fmt.Sprintf(
			"  • UNSOUNDNESS WARNING (CodeAnchor/CaMeLoT): Static call-graph analysis is incomplete due to dynamic dispatch/reflection %v. Do not assume 0 unseen callers.\n",
			r.UnsoundnessReasons,
		))
	}
	if len(r.Consumers) > 0 {
		var touched []string
		touched = append(touched, r.TouchedSymbols...)
		for _, ck := range r.TouchedContractKeys {
			touched = append(touched, fmt.Sprintf("%q", ck))
		}
		b.WriteString(fmt.Sprintf("Touched symbols & wire contracts in %s: [%s]\n", filepath.Base(filePath), strings.Join(touched, ", ")))
		b.WriteString("Downstream Callers, Duck-Typed Interfaces & Wire-Contract Consumers:\n")
		for _, c := range r.Consumers {
			b.WriteString(fmt.Sprintf("  • %s %q -> %s:%d: %s\n", c.Kind, c.Token, c.RelPath, c.Line, c.Snippet))
		}
	}
	return strings.TrimSpace(b.String())
}

func truncateSnippet(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
