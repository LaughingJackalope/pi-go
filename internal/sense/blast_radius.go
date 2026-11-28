package sense

import (
	"bytes"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// BlastImpactLevel classifies the cross-file scope of an edit.
const (
	BlastLevelLocal        = "LOCAL"         // 0 external files affected
	BlastLevelPackage      = "PACKAGE"       // 1+ sibling files in the same directory/package
	BlastLevelCrossPackage = "CROSS_PACKAGE" // 1+ files across sibling packages/modules
	BlastLevelCrossService = "CROSS_SERVICE" // 1+ wire-contract or cross-language consumers across services/stacks
)

// ImpactConsumer records one downstream caller, duck-typed interface, or
// wire-contract consumer outside the edited file.
type ImpactConsumer struct {
	Kind          string `json:"kind"`                     // "SYMBOL_CALLER" | "WIRE_CONTRACT_CONSUMER"
	Token         string `json:"token"`                    // function/method/type name or wire-contract key
	RelPath       string `json:"rel_path"`                 // path relative to search root
	Line          int    `json:"line"`                     // 1-based line number
	Snippet       string `json:"snippet"`                  // trimmed source line
	CrossPackage  bool   `json:"cross_package"`            // true if in a different directory than focusFile
	CrossLanguage bool   `json:"cross_language,omitempty"` // true if in a different language family (e.g. .py <-> .ts)
}

// BlastRadiusReport summarizes the downstream impact radius of an edit.
type BlastRadiusReport struct {
	Level               string           `json:"level"`
	SearchRoot          string           `json:"search_root"`
	FilesScanned        int              `json:"files_scanned"`
	ScanElapsedMs       float64          `json:"scan_elapsed_ms"`
	TouchedSymbols      []string         `json:"touched_symbols,omitempty"`
	TouchedContractKeys []string         `json:"touched_contract_keys,omitempty"`
	ExternalFilesCount  int              `json:"external_files_count"`
	ExternalDirsCount   int              `json:"external_dirs_count"`
	CrossLanguageHits   int              `json:"cross_language_hits,omitempty"`
	UnsoundnessDetected bool             `json:"unsoundness_detected,omitempty"`
	UnsoundnessReasons  []string         `json:"unsoundness_reasons,omitempty"`
	Consumers           []ImpactConsumer `json:"consumers,omitempty"`
	Summary             string           `json:"summary,omitempty"`
}

var (
	goFuncHeaderRe    = regexp.MustCompile(`^func\s+(?:\([^)]+\)\s*)?([A-Z][A-Za-z0-9_]*)\s*\(`)
	goTypeHeaderRe    = regexp.MustCompile(`^type\s+([A-Z][A-Za-z0-9_]*)\s+(?:struct|interface)\b`)
	pyDefHeaderRe     = regexp.MustCompile(`^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	pyClassHeaderRe   = regexp.MustCompile(`^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b`)
	tsExportDeclRe    = regexp.MustCompile(`^\s*export\s+(?:default\s+)?(?:async\s+)?(?:function|const|class|interface|type|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)\b`)
	javaSymHeaderRe   = regexp.MustCompile(`^\s*(?:public|protected)\s+(?:static\s+|final\s+|synchronized\s+|abstract\s+)*(?:<[^>]+>\s+)?(?:(?:class|interface|enum|record)\s+([A-Za-z_][A-Za-z0-9_]*)|[A-Za-z_][A-Za-z0-9_<>.,?\[\]]*\s+([a-zA-Z_][A-Za-z0-9_]*)\s*\()`)
	scalaSymHeaderRe  = regexp.MustCompile(`^\s*(?:override\s+|protected\s+|final\s+)*(?:(?:class|trait|object)\s+([A-Za-z_][A-Za-z0-9_]*)|def\s+([A-Za-z_][A-Za-z0-9_]*)\b)`)
	stringLitKeyRe    = regexp.MustCompile("[\"'`]([a-zA-Z/][a-zA-Z0-9_./:-]{3,64})(?:\\$\\{|[\"'`?])")
	apiRouteLitRe     = regexp.MustCompile("[\"'`](/api/[a-zA-Z0-9_/-]+|/v[1-9][a-z]*/[a-zA-Z0-9_/-]+)")
	goUnsoundRe       = regexp.MustCompile(`\b(?:reflect\.(?:ValueOf|TypeOf)|MethodByName|unsafe\.Pointer)\b|\.\((?:any|interface\{\})\)`)
	pyUnsoundRe       = regexp.MustCompile(`\b(?:getattr|setattr|__getattr__|__call__|eval|exec|importlib)\s*\(|\*\*kwargs\b`)
	jvmUnsoundRe      = regexp.MustCompile(`\b(?:Class\.forName|Method\.invoke|getDeclaredMethod|sun\.misc\.Unsafe|VarHandle|asInstanceOf\[)\b`)
	tsUnsoundRe       = regexp.MustCompile(`\b(?:Reflect\.apply|eval\s*\(|new\s+Function\s*\()\b|\bas\s+any\b`)
)

var ignoredGenericMethodNames = map[string]bool{
	"__init__": true, "main": true, "get": true, "set": true, "add": true,
	"read": true, "write": true, "close": true, "run": true, "status": true,
	"send": true, "say": true, "log": true, "parse": true, "apply": true,
	"toString": true, "hashCode": true, "equals": true, "size": true,
	"isEmpty": true, "clear": true, "remove": true, "put": true, "update": true,
	"delete": true, "list": true, "start": true, "stop": true, "init": true,
}

var ignoredContractLiterals = map[string]bool{
	"true": true, "false": true, "null": true, "none": true, "undefined": true,
	"application/json": true, "text/plain": true, "text/event-stream": true,
	"content-type": true, "content-length": true, "authorization": true,
	"utf-8": true, "http://": true, "https://": true, "same-origin": true,
	"no-store": true, "include": true,
}

var blastSkippedDirs = map[string]bool{
	".git": true, "vendor": true, "node_modules": true, "__pycache__": true,
	".venv": true, "venv": true, "dist": true, "build": true, "target": true,
	"out": true, "website": true, "locales": true, "contributors": true,
	"optional-skills": true, "plugin-catalog": true, "assets": true,
	"tests": true, "tests-js": true, "test": true, "testdata": true, "testing": true,
}

// AddSearchRoot registers an additional root directory for the Cross-File Blast-Radius Scanner.
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

// ScanBlastRadius analyzes an edited or target file across Go, Python, TypeScript,
// Java, Scala, and Shell scripts using a parallel zero-alloc byte pre-filter:
//  1. Identifies enclosing or modified symbols (functions, methods, classes, interfaces)
//  2. Extracts modified wire-contract keys, REST/WS routes (/api/...), HTTP headers,
//     Kubernetes annotations, and Kafka dot-config keys
//  3. Sweeps sibling packages, services, and frontend/backend stacks in parallel.
func (e *Engine) ScanBlastRadius(filePath string, args map[string]any) *BlastRadiusReport {
	started := time.Now()
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

	// Merge explicit CLI / tool overrides if provided
	for _, s := range parseStringListArg(args["symbols"]) {
		symbols = appendUniqueSorted(symbols, s)
	}
	for _, c := range parseStringListArg(args["contracts"]) {
		contractKeys = appendUniqueSorted(contractKeys, c)
	}

	// If scanning a file directly (e.g. `pi-s3 s3 blast <file>`) with no edit snippet,
	// extract the file's primary exported/public symbols and wire contracts automatically.
	if len(symbols) == 0 && len(contractKeys) == 0 && oldStr == "" && newStr == "" {
		symbols, contractKeys = extractFileHeadlineContracts(filePath, content)
	}

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
	focusLang := langFamily(filePath)

	symMatchers := make(map[string]*regexp.Regexp, len(symbols))
	symNeedles := make([][]byte, 0, len(symbols))
	for _, sym := range symbols {
		symMatchers[sym] = regexp.MustCompile(`\b` + regexp.QuoteMeta(sym) + `\b`)
		symNeedles = append(symNeedles, []byte(sym))
	}
	conNeedles := make([][]byte, 0, len(contractKeys))
	for _, ck := range contractKeys {
		conNeedles = append(conNeedles, []byte(ck))
	}

	type candidateFile struct {
		root string
		path string
	}
	var candidates []candidateFile
	seenPaths := make(map[string]bool)
	const maxCandidateFiles = 8000

	primaryRoot := roots[0]
	for _, root := range roots {
		_ = filepath.WalkDir(root, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			name := d.Name()
			if d.IsDir() {
				if p != root && (strings.HasPrefix(name, ".") || blastSkippedDirs[name]) {
					return filepath.SkipDir
				}
				return nil
			}
			if len(candidates) >= maxCandidateFiles {
				return fs.SkipAll
			}
			if isSkippedBlastFile(name) {
				return nil
			}
			absP, _ := filepath.Abs(p)
			if absP == absFocus || name == focusBase || seenPaths[absP] {
				return nil
			}
			seenPaths[absP] = true
			candidates = append(candidates, candidateFile{root: root, path: absP})
			return nil
		})
	}

	workers := min(max(2, runtime.NumCPU()), 8)
	ch := make(chan candidateFile, len(candidates))
	for _, c := range candidates {
		ch <- c
	}
	close(ch)

	var (
		mu                  sync.Mutex
		consumers           []ImpactConsumer
		extFiles            = make(map[string]bool)
		extDirs             = make(map[string]bool)
		hasWireCrossService bool
		hasCrossPkg         bool
		crossLangHits       int
	)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			var localConsumers []ImpactConsumer
			for cand := range ch {
				blob, rErr := os.ReadFile(cand.path)
				if rErr != nil || len(blob) == 0 || len(blob) > 2*1024*1024 {
					continue
				}
				// Fast zero-alloc byte substring pre-filter before any string conversion!
				matchedAny := false
				for _, nb := range symNeedles {
					if bytes.Contains(blob, nb) {
						matchedAny = true
						break
					}
				}
				if !matchedAny {
					for _, cb := range conNeedles {
						if bytes.Contains(blob, cb) {
							matchedAny = true
							break
						}
					}
				}
				if !matchedAny {
					continue
				}

				rel, relErr := filepath.Rel(cand.root, cand.path)
				if relErr != nil {
					rel = cand.path
				}
				isCrossDir := filepath.Dir(cand.path) != focusDir
				isCrossLang := focusLang != "" && langFamily(cand.path) != "" && langFamily(cand.path) != focusLang

				lines := strings.Split(string(blob), "\n")
				for idx, rawLine := range lines {
					trimmed := strings.TrimSpace(rawLine)
					if trimmed == "" || strings.HasPrefix(trimmed, "//") || strings.HasPrefix(trimmed, "#") || strings.HasPrefix(trimmed, "*") {
						continue
					}

					for _, sym := range symbols {
						re := symMatchers[sym]
						if re != nil && re.MatchString(trimmed) {
							localConsumers = append(localConsumers, ImpactConsumer{
								Kind:          "SYMBOL_CALLER",
								Token:         sym,
								RelPath:       rel,
								Line:          idx + 1,
								Snippet:       truncateSnippet(trimmed, 96),
								CrossPackage:  isCrossDir,
								CrossLanguage: isCrossLang,
							})
						}
					}

					for _, ck := range contractKeys {
						if matchesContractLiteralOnLine(trimmed, ck) {
							localConsumers = append(localConsumers, ImpactConsumer{
								Kind:          "WIRE_CONTRACT_CONSUMER",
								Token:         ck,
								RelPath:       rel,
								Line:          idx + 1,
								Snippet:       truncateSnippet(trimmed, 96),
								CrossPackage:  isCrossDir,
								CrossLanguage: isCrossLang,
							})
						}
					}
				}
			}
			if len(localConsumers) > 0 {
				mu.Lock()
				for _, c := range localConsumers {
					consumers = append(consumers, c)
					extFiles[c.RelPath] = true
					if c.CrossPackage {
						extDirs[filepath.Dir(c.RelPath)] = true
						if c.Kind == "WIRE_CONTRACT_CONSUMER" {
							hasWireCrossService = true
						} else {
							hasCrossPkg = true
						}
					}
					if c.CrossLanguage {
						hasWireCrossService = true
						crossLangHits++
					}
				}
				mu.Unlock()
			}
		}()
	}
	wg.Wait()

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
		FilesScanned:        len(candidates),
		ScanElapsedMs:       float64(time.Since(started).Microseconds()) / 1000.0,
		TouchedSymbols:      symbols,
		TouchedContractKeys: contractKeys,
		ExternalFilesCount:  len(extFiles),
		ExternalDirsCount:   len(extDirs),
		CrossLanguageHits:   crossLangHits,
		UnsoundnessDetected: len(unsoundReasons) > 0,
		UnsoundnessReasons:  unsoundReasons,
		Consumers:           dedupAndCapConsumers(consumers, 12),
	}
	report.Summary = formatBlastRadiusSummary(filePath, report)
	return report
}

func isSkippedBlastFile(name string) bool {
	if strings.HasSuffix(name, "_test.go") ||
		strings.HasPrefix(name, "test_") ||
		strings.HasSuffix(name, "_test.py") ||
		strings.HasSuffix(name, ".test.ts") ||
		strings.HasSuffix(name, ".test.tsx") ||
		strings.HasSuffix(name, ".spec.ts") ||
		strings.HasSuffix(name, ".spec.tsx") ||
		strings.Contains(name, ".fixture.") ||
		strings.HasSuffix(name, "Test.java") ||
		strings.HasSuffix(name, "Tests.java") ||
		strings.HasSuffix(name, "Test.scala") ||
		strings.HasPrefix(name, "zz_generated") ||
		strings.HasSuffix(name, ".pb.go") ||
		strings.HasSuffix(name, ".d.ts") {
		return true
	}
	ext := filepath.Ext(name)
	switch ext {
	case ".go", ".py", ".ts", ".tsx", ".java", ".scala", ".sh":
		return false
	default:
		return true
	}
}

func langFamily(path string) string {
	switch filepath.Ext(path) {
	case ".go":
		return "go"
	case ".py":
		return "python"
	case ".ts", ".tsx":
		return "typescript"
	case ".java":
		return "java"
	case ".scala":
		return "scala"
	case ".sh":
		return "shell"
	default:
		return ""
	}
}

func matchesContractLiteralOnLine(trimmed, ck string) bool {
	if strings.Contains(trimmed, `"`+ck+`"`) || strings.Contains(trimmed, `'`+ck+`'`) || strings.Contains(trimmed, "`"+ck+"`") {
		return true
	}
	// Match TypeScript template literals or URL query suffixes on API routes, e.g. `/api/dashboard/plugins${...}` or `/api/pty?cwd=`
	if strings.HasPrefix(ck, "/") {
		return strings.Contains(trimmed, `"`+ck+`?`) ||
			strings.Contains(trimmed, `'`+ck+`?`) ||
			strings.Contains(trimmed, "`"+ck+"?") ||
			strings.Contains(trimmed, "`"+ck+"${") ||
			strings.Contains(trimmed, `"`+ck+`/{`)
	}
	return false
}

func detectStaticUnsoundness(filePath, text string) []string {
	reasons := make(map[string]bool)
	switch filepath.Ext(filePath) {
	case ".go":
		for _, m := range goUnsoundRe.FindAllString(text, -1) {
			reasons[m] = true
		}
	case ".py":
		for _, m := range pyUnsoundRe.FindAllString(text, -1) {
			reasons[strings.TrimSpace(strings.TrimSuffix(m, "("))] = true
		}
	case ".java", ".scala":
		for _, m := range jvmUnsoundRe.FindAllString(text, -1) {
			reasons[strings.TrimSuffix(m, "[")] = true
		}
	case ".ts", ".tsx":
		for _, m := range tsUnsoundRe.FindAllString(text, -1) {
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

	// Walk upward up to 7 levels looking for project/monorepo roots across Go, Python, TS, and Gradle/Maven
	cur := filepath.Dir(filePath)
	bestRepoRoot := ""
	for depth := 0; depth < 7; depth++ {
		base := filepath.Base(cur)
		if base == "internal" || base == "apps" {
			add(cur)
			break
		}
		for _, marker := range []string{"go.mod", "pyproject.toml", "settings.gradle", "settings.gradle.kts", "pom.xml", ".git"} {
			if _, err := os.Stat(filepath.Join(cur, marker)); err == nil {
				bestRepoRoot = cur
			}
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	if bestRepoRoot != "" {
		add(bestRepoRoot)
	}
	if len(roots) == 0 {
		add(filepath.Dir(filePath))
	}
	return roots
}

func extractTouchedSymbolsAndContracts(filePath, content, oldStr, newStr string) (symbols []string, contracts []string) {
	symSet := make(map[string]bool)
	conSet := make(map[string]bool)

	snippet := strings.TrimSpace(newStr)
	if snippet == "" {
		snippet = strings.TrimSpace(oldStr)
	}

	fam := langFamily(filePath)

	// 1. Find enclosing function/method/class around the edited region in content
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
			for i := lineNum; i >= 0; i-- {
				if sym := matchEnclosingDeclaration(fam, allLines[i]); sym != "" {
					symSet[sym] = true
					start := max(0, i-2) // include decorators like @router.get("/api/...") right above def!
					end := min(len(allLines), i+18)
					activeBlock = strings.Join(allLines[start:end], "\n")
					break
				}
			}
		}
	}

	combined := oldStr + "\n" + newStr + "\n" + activeBlock
	collectSymbolsAndContractsFromLines(fam, combined, symSet, conSet, 16)
	return sortedKeys(symSet), sortedKeys(conSet)
}

func extractFileHeadlineContracts(filePath, content string) (symbols []string, contracts []string) {
	symSet := make(map[string]bool)
	conSet := make(map[string]bool)
	collectSymbolsAndContractsFromLines(langFamily(filePath), content, symSet, conSet, 8)
	return sortedKeys(symSet), sortedKeys(conSet)
}

func collectSymbolsAndContractsFromLines(fam, block string, symSet, conSet map[string]bool, capCount int) {
	for _, line := range strings.Split(block, "\n") {
		if len(symSet) < capCount {
			if sym := matchEnclosingDeclaration(fam, line); sym != "" {
				symSet[sym] = true
			}
		}
		if len(conSet) < capCount {
			for _, rm := range apiRouteLitRe.FindAllStringSubmatch(line, -1) {
				route := strings.TrimRight(rm[1], "/")
				if len(route) >= 5 {
					conSet[route] = true
				}
			}
			for _, sm := range stringLitKeyRe.FindAllStringSubmatch(line, -1) {
				lit := sm[1]
				if isWireContractKey(lit) {
					conSet[lit] = true
				}
			}
		}
	}
}

func matchEnclosingDeclaration(fam, line string) string {
	trimmed := strings.TrimSpace(line)
	switch fam {
	case "go":
		if m := goFuncHeaderRe.FindStringSubmatch(trimmed); m != nil && !ignoredGenericMethodNames[m[1]] {
			return m[1]
		}
		if m := goTypeHeaderRe.FindStringSubmatch(trimmed); m != nil {
			return m[1]
		}
	case "python":
		if m := pyDefHeaderRe.FindStringSubmatch(line); m != nil && !ignoredGenericMethodNames[m[1]] && !strings.HasPrefix(m[1], "_") {
			return m[1]
		}
		if m := pyClassHeaderRe.FindStringSubmatch(line); m != nil && !strings.HasPrefix(m[1], "_") {
			return m[1]
		}
	case "typescript":
		if m := tsExportDeclRe.FindStringSubmatch(line); m != nil && !ignoredGenericMethodNames[m[1]] && len(m[1]) >= 4 {
			return m[1]
		}
	case "java":
		if m := javaSymHeaderRe.FindStringSubmatch(line); m != nil {
			cand := m[1]
			if cand == "" {
				cand = m[2]
			}
			if cand != "" && !polyglotKeywords[cand] && !ignoredGenericMethodNames[cand] && len(cand) >= 4 {
				return cand
			}
		}
	case "scala":
		if m := scalaSymHeaderRe.FindStringSubmatch(line); m != nil {
			cand := m[1]
			if cand == "" {
				cand = m[2]
			}
			if cand != "" && !polyglotKeywords[cand] && !ignoredGenericMethodNames[cand] && len(cand) >= 4 {
				return cand
			}
		}
	}
	return ""
}

func isWireContractKey(s string) bool {
	lower := strings.ToLower(s)
	if ignoredContractLiterals[lower] {
		return false
	}
	// REST/WebSocket API endpoints ("/api/auth/me", "/api/dashboard/plugins", "/v1/decisions")
	if strings.HasPrefix(s, "/api/") || strings.HasPrefix(s, "/v1/") || strings.HasPrefix(s, "/v1beta/") {
		return true
	}
	// Custom HTTP auth/protocol headers ("X-Hermes-Session-Token", "X-Forwarded-Prefix")
	if strings.HasPrefix(s, "X-") && len(s) >= 6 {
		return true
	}
	// Kubernetes / Portal label & annotation keys ("portal.dev/task", "k8s.io/...", "kubernetes.io/...")
	if strings.Contains(s, "portal.dev/") || strings.Contains(s, "k8s.io/") || strings.Contains(s, "kubernetes.io/") {
		return true
	}
	// Kafka / JVM dot-separated configuration & metric keys ("group.initial.rebalance.delay.ms", "cleanup.policy")
	if strings.Count(s, ".") >= 2 && s == lower && !strings.HasPrefix(s, "org.") && !strings.HasPrefix(s, "com.") && !strings.HasPrefix(s, "java.") && !strings.HasPrefix(s, "scala.") && !strings.Contains(s, "/") {
		return true
	}
	// Snake_case wire dictionary keys ("task_budget_usd", "over_budget", "session_expired", "login_url")
	if strings.Contains(s, "_") && !strings.HasPrefix(s, "_") && !strings.Contains(s, "/") &&
		!strings.HasSuffix(s, ".go") && !strings.HasSuffix(s, ".py") && !strings.HasSuffix(s, ".ts") {
		return true
	}
	return false
}

func parseStringListArg(v any) []string {
	switch x := v.(type) {
	case []string:
		return x
	case string:
		var out []string
		for _, part := range strings.Split(x, ",") {
			if p := strings.TrimSpace(part); p != "" {
				out = append(out, p)
			}
		}
		return out
	default:
		return nil
	}
}

func appendUniqueSorted(list []string, item string) []string {
	for _, x := range list {
		if x == item {
			return list
		}
	}
	list = append(list, item)
	sort.Strings(list)
	return list
}

func dedupAndCapConsumers(in []ImpactConsumer, limit int) []ImpactConsumer {
	sort.SliceStable(in, func(i, j int) bool {
		if in[i].CrossLanguage != in[j].CrossLanguage {
			return in[i].CrossLanguage
		}
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
		"[S3 BLAST-RADIUS SCANNER · %s (%d external file(s) across %d sibling package(s)/service(s), scanned %d files in %.1fms)]\n",
		r.Level, r.ExternalFilesCount, max(1, r.ExternalDirsCount), r.FilesScanned, r.ScanElapsedMs,
	))
	if r.CrossLanguageHits > 0 {
		b.WriteString(fmt.Sprintf(
			"  • CROSS-LANGUAGE WIRE BOUNDARY: %d hit(s) across polyglot frontend/backend/JVM stacks.\n",
			r.CrossLanguageHits,
		))
	}
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
			xlang := ""
			if c.CrossLanguage {
				xlang = " [cross-lang]"
			}
			b.WriteString(fmt.Sprintf("  • %s%s %q -> %s:%d: %s\n", c.Kind, xlang, c.Token, c.RelPath, c.Line, c.Snippet))
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
