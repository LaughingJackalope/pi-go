package sense

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

var (
	// Java / Scala / TypeScript class & object headers
	jvmTSClassRe = regexp.MustCompile(`^(?:public\s+|protected\s+|private\s+|abstract\s+|final\s+|sealed\s+|export\s+|default\s+|case\s+)*(?:class|object)\s+([A-Za-z_][A-Za-z0-9_]*)`)

	// Lock field type indicators across Java, Scala, and TypeScript
	polyLockTypeRe = regexp.MustCompile(`\b(?:ReentrantLock|ReentrantReadWriteLock|ReadWriteLock|StampedLock|Semaphore|Mutex|RWLock|AsyncLock)\b`)
	polyLockNameRe = regexp.MustCompile(`(?i)(?:^lock$|^mu$|^_lock$|^rwlock$|Lock$|Mutex$)`)

	// Java field declaration: e.g. `private final Deque<ByteBuffer> free;` or `private long nonPooledAvailableMemory = 0;`
	javaFieldDeclRe = regexp.MustCompile(`^(?:private|protected|public|volatile|transient|\s)+(?:final\s+)?(?:[A-Za-z_][A-Za-z0-9_<>.,?\s\[\]]*?)\s+([a-z_][A-Za-z0-9_]*)\s*(?:=|;)`)

	// Scala val/var field declaration (class body or constructor params)
	scalaFieldDeclRe = regexp.MustCompile(`(?:@volatile\s+)?(?:private(?:\[[A-Za-z0-9_]+\])?\s+|protected\s+|override\s+)*(?:val|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::\s*[^=;]+)?(?:\s*=|$|,|\))`)

	// TypeScript class field declaration: e.g. `private items: Map<string, any> = new Map();`
	tsClassFieldRe = regexp.MustCompile(`^(?:private|protected|public|readonly|\s)+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:\?|:|=)`)

	// TypeScript module-scoped stateful singletons: `let _profile = ""` or `const _registered: Map<...> = new Map()`
	tsModuleLetRe       = regexp.MustCompile(`^(?:export\s+)?let\s+([A-Za-z_$][A-Za-z0-9_$]*)\b`)
	tsModuleContainerRe = regexp.MustCompile(`^(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*new\s+(?:Map|Set|WeakMap|WeakSet|Array)\b`)

	// Method declarations (including multi-line parameter lists where `(` opens on header line)
	javaMethodDeclRe      = regexp.MustCompile(`^(?:public|protected|private|synchronized|final|abstract|\s)*(?:<[^>]+>\s+)?[A-Za-z_][A-Za-z0-9_<>.,?\[\]]*\s+([a-z_][A-Za-z0-9_]*)\s*\([^;]*$`)
	scalaMethodDeclRe     = regexp.MustCompile(`^(?:override\s+|private(?:\[[A-Za-z0-9_]+\])?\s+|protected\s+|final\s+)*def\s+([A-Za-z_][A-Za-z0-9_]*)\b`)
	tsClassMethodRe       = regexp.MustCompile(`^(?:public\s+|protected\s+|private\s+|async\s+)*([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<[^>]+>)?\s*\([^;]*\{\s*$`)
	tsModuleFuncRe        = regexp.MustCompile(`^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<[^>]+>)?\s*\(`)
	tsArrowFuncRe         = regexp.MustCompile(`^(?:export\s+)?const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::[^=]+)?=>\s*\{`)

	// Lock acquisition idioms across Java, Scala (Kafka LockUtils with optional type args), and TypeScript
	polyLockCallRe = regexp.MustCompile(`\b(?:synchronized\s*\(|\.synchronized\b|(?:inReadLock|inWriteLock|inLock)(?:\[[^\]]+\])?\s*\(|\.lock\s*\(|\.readLock\s*\(\)\.lock|\.writeLock\s*\(\)\.lock|\.runExclusive\s*\(|\.acquire\s*\(|navigator\.locks\.request\s*\()`)
	syncBlockArgRe = regexp.MustCompile(`\bsynchronized\s*\(\s*(?:this\.)?([A-Za-z_][A-Za-z0-9_]*)\s*\)`)
	inLockArgRe    = regexp.MustCompile(`\b(?:inReadLock|inWriteLock|inLock)(?:\[[^\]]+\])?\s*\(\s*(?:this\.)?([A-Za-z_][A-Za-z0-9_]*)\s*[,)]`)

	// Mutating method calls on collection/atomic fields across JVM & TS
	mutatingReceiverCallRe = regexp.MustCompile(`(?:this\.)?([A-Za-z_$][A-Za-z0-9_$]*)\s*\.\s*(?:put|putIfAbsent|remove|clear|add|addAll|addFirst|addLast|poll|pollFirst|pollLast|offer|set|getAndSet|incrementAndGet|decrementAndGet|addAndGet|compareAndSet|update|delete|push|pop|shift|unshift|splice)\s*\(`)
	assignFieldRe          = regexp.MustCompile(`(?:this\.)?([A-Za-z_$][A-Za-z0-9_$]*)(?:\s*\[[^\]]+\])?\s*(?:[+\-*/|&^]?=|\+\+|--)`)
	identTokenRe           = regexp.MustCompile(`\b([A-Za-z_$][A-Za-z0-9_$]*)\b`)
	intraHelperCallRe      = regexp.MustCompile(`(?:this\.)?([A-Za-z_$][A-Za-z0-9_$]*)\s*\(`)
)

var polyglotKeywords = map[string]bool{
	"if": true, "else": true, "for": true, "while": true, "do": true, "switch": true,
	"case": true, "return": true, "throw": true, "try": true, "catch": true, "finally": true,
	"new": true, "class": true, "interface": true, "enum": true, "object": true, "trait": true,
	"public": true, "private": true, "protected": true, "static": true, "final": true,
	"void": true, "int": true, "long": true, "boolean": true, "double": true, "float": true,
	"val": true, "var": true, "def": true, "override": true, "synchronized": true,
	"function": true, "const": true, "let": true, "async": true, "await": true, "export": true,
	"import": true, "package": true, "true": true, "false": true, "null": true, "undefined": true,
	"super": true, "this": true, "match": true, "yield": true, "typeof": true,
}

// AnalyzePolyglotDir extracts the implicit state-machine coupling graph (fields,
// lock guards, method reads/writes, and 1-hop intra-class helper calls) for
// Java (.java), Scala (.scala), and TypeScript (.ts / .tsx) directories.
func AnalyzePolyglotDir(dir, focusFile, lang string) (*PackageStateGraph, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	graph := &PackageStateGraph{
		PackageName:  filepath.Base(dir),
		StructFields: make(map[string][]string),
		LockFields:   make(map[string][]string),
	}

	type cleanFile struct {
		name  string
		lines []string
	}
	var files []cleanFile
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !matchesPolyglotLang(name, lang) || isPolyglotTestOrDeclFile(name) {
			continue
		}
		raw, rErr := os.ReadFile(filepath.Join(dir, name))
		if rErr != nil {
			continue
		}
		files = append(files, cleanFile{
			name:  name,
			lines: stripCStyleCommentsPreserveLines(string(raw)),
		})
	}

	knownFields := make(map[string]map[string]bool)
	lockFields := make(map[string]map[string]bool)

	ensureUnit := func(u string) {
		if knownFields[u] == nil {
			knownFields[u] = make(map[string]bool)
			lockFields[u] = make(map[string]bool)
		}
	}

	// Pass 1: Discover classes/objects (plus TS module-scoped singletons) and their state & lock fields.
	for _, f := range files {
		curClass := ""
		classBraceDepth := -1
		braceDepth := 0
		inScalaCtorHeader := false

		moduleUnit := "Module:" + strings.TrimSuffix(strings.TrimSuffix(f.name, ".tsx"), ".ts")

		for _, line := range f.lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}

			// Detect class/object entry when not already inside a method body
			if m := jvmTSClassRe.FindStringSubmatch(trimmed); m != nil && !strings.Contains(trimmed, ";") {
				if curClass == "" || braceDepth <= classBraceDepth+1 {
					curClass = m[1]
					classBraceDepth = braceDepth
					ensureUnit(curClass)
					if lang == "scala" && strings.Contains(trimmed, "(") && !strings.Contains(trimmed, "{") {
						inScalaCtorHeader = true
					}
				}
			}

			// Scala multi-line primary constructor `val`/`var` fields
			if lang == "scala" && curClass != "" && (inScalaCtorHeader || braceDepth == classBraceDepth+1) {
				for _, sm := range scalaFieldDeclRe.FindAllStringSubmatch(trimmed, -1) {
					fld := sm[1]
					if !polyglotKeywords[fld] {
						knownFields[curClass][fld] = true
						if polyLockTypeRe.MatchString(trimmed) || polyLockNameRe.MatchString(fld) {
							lockFields[curClass][fld] = true
						}
					}
				}
				if inScalaCtorHeader && strings.Contains(trimmed, "{") {
					inScalaCtorHeader = false
				}
			}

			// Java field declarations at class body depth
			if lang == "java" && curClass != "" && braceDepth == classBraceDepth+1 {
				if !strings.Contains(trimmed, "(") || strings.Contains(trimmed, "=") {
					if !strings.Contains(trimmed, " static ") && !strings.HasPrefix(trimmed, "static ") {
						if jm := javaFieldDeclRe.FindStringSubmatch(trimmed); jm != nil {
							fld := jm[1]
							if !polyglotKeywords[fld] {
								knownFields[curClass][fld] = true
								if polyLockTypeRe.MatchString(trimmed) || polyLockNameRe.MatchString(fld) {
									lockFields[curClass][fld] = true
								}
							}
						}
					}
				}
			}

			// TypeScript class fields OR top-level module singleton state
			if lang == "typescript" {
				if curClass != "" && braceDepth == classBraceDepth+1 && (!strings.Contains(trimmed, "(") || strings.Contains(trimmed, "=")) {
					if tm := tsClassFieldRe.FindStringSubmatch(trimmed); tm != nil {
						fld := tm[1]
						if !polyglotKeywords[fld] {
							knownFields[curClass][fld] = true
							if polyLockTypeRe.MatchString(trimmed) || polyLockNameRe.MatchString(fld) {
								lockFields[curClass][fld] = true
							}
						}
					}
				} else if curClass == "" && braceDepth == 0 {
					if lm := tsModuleLetRe.FindStringSubmatch(trimmed); lm != nil {
						ensureUnit(moduleUnit)
						knownFields[moduleUnit][lm[1]] = true
					} else if cm := tsModuleContainerRe.FindStringSubmatch(trimmed); cm != nil {
						ensureUnit(moduleUnit)
						knownFields[moduleUnit][cm[1]] = true
					}
				}
			}

			// Recognize fields used inside `synchronized(field)` or `inReadLock(field)` / `inWriteLock(field)` as lock fields!
			if curClass != "" {
				if sm := syncBlockArgRe.FindStringSubmatch(trimmed); sm != nil && sm[1] != "this" {
					if knownFields[curClass][sm[1]] {
						lockFields[curClass][sm[1]] = true
					}
				}
				if im := inLockArgRe.FindStringSubmatch(trimmed); im != nil {
					ensureUnit(curClass)
					knownFields[curClass][im[1]] = true
					lockFields[curClass][im[1]] = true
				}
			}

			braceDepth += countBraceDelta(line)
			if braceDepth < 0 {
				braceDepth = 0
			}
			if curClass != "" && braceDepth <= classBraceDepth && strings.Contains(line, "}") {
				curClass = ""
				classBraceDepth = -1
			}
		}
	}

	for u, fset := range knownFields {
		if len(fset) == 0 {
			continue
		}
		graph.StructFields[u] = sortedKeys(fset)
		graph.LockFields[u] = sortedKeys(lockFields[u])
	}

	// Pass 2: Extract per-method reads, writes, lock guards, visibility, and intra-class calls.
	for _, f := range files {
		curClass := ""
		classBraceDepth := -1
		braceDepth := 0

		curUnit := ""
		curMethod := ""
		curPrivate := false
		methodStartDepth := -1
		methodLine := 0
		awaitingMethodBody := false
		singleLineScalaDef := false
		var reads, writes, calls map[string]bool
		usesLock := false

		moduleUnit := "Module:" + strings.TrimSuffix(strings.TrimSuffix(f.name, ".tsx"), ".ts")
		hasModuleState := len(knownFields[moduleUnit]) > 0

		flushMethod := func() {
			if curUnit != "" && curMethod != "" && curMethod != curUnit && curMethod != "constructor" && curMethod != "apply" {
				graph.Methods = append(graph.Methods, MethodTouch{
					StructName: curUnit,
					MethodName: curMethod,
					File:       f.name,
					Line:       methodLine,
					Reads:      sortedKeys(reads),
					Writes:     sortedKeys(writes),
					Calls:      sortedKeys(calls),
					UsesLock:   usesLock,
					IsPrivate:  curPrivate,
				})
			}
			curMethod = ""
			methodStartDepth = -1
			awaitingMethodBody = false
			singleLineScalaDef = false
		}

		for idx, line := range f.lines {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" {
				continue
			}

			if m := jvmTSClassRe.FindStringSubmatch(trimmed); m != nil && !strings.Contains(trimmed, ";") {
				if curClass == "" || braceDepth <= classBraceDepth+1 {
					flushMethod()
					curClass = m[1]
					classBraceDepth = braceDepth
				}
			}

			// Detect method start at class body depth (or TS module top-level depth 0)
			if curMethod == "" {
				if curClass != "" && knownFields[curClass] != nil && braceDepth == classBraceDepth+1 {
					mName, isPriv, syncMod, singleExpr := matchPolyglotMethodHeader(trimmed, lang, curClass)
					if mName != "" {
						curUnit = curClass
						curMethod = mName
						curPrivate = isPriv
						methodStartDepth = braceDepth
						methodLine = idx + 1
						reads = make(map[string]bool)
						writes = make(map[string]bool)
						calls = make(map[string]bool)
						usesLock = syncMod
						awaitingMethodBody = !strings.Contains(trimmed, "{") && !strings.Contains(trimmed, "=")
						singleLineScalaDef = singleExpr && !awaitingMethodBody
					}
				} else if lang == "typescript" && curClass == "" && hasModuleState && braceDepth == 0 {
					mName, isPriv := matchTSModuleFuncHeader(trimmed)
					if mName != "" {
						curUnit = moduleUnit
						curMethod = mName
						curPrivate = isPriv
						methodStartDepth = braceDepth
						methodLine = idx + 1
						reads = make(map[string]bool)
						writes = make(map[string]bool)
						calls = make(map[string]bool)
						usesLock = false
						awaitingMethodBody = !strings.Contains(trimmed, "{")
					}
				}
			} else if awaitingMethodBody {
				if strings.Contains(trimmed, ";") && !strings.Contains(trimmed, "{") {
					// Abstract / interface signature with no body
					curMethod = ""
					awaitingMethodBody = false
				} else if strings.Contains(trimmed, "{") {
					awaitingMethodBody = false
					singleLineScalaDef = false
				} else if lang == "scala" && strings.Contains(trimmed, "=") && strings.Contains(trimmed, ")") {
					awaitingMethodBody = false
					singleLineScalaDef = true
				}
			}

			if curMethod != "" {
				uKnown := knownFields[curUnit]
				uLocks := lockFields[curUnit]

				if polyLockCallRe.MatchString(trimmed) {
					usesLock = true
				}
				if strings.Contains(trimmed, "synchronized ") || strings.HasPrefix(trimmed, "synchronized(") {
					usesLock = true
				}

				// Mutating method calls on fields: e.g. `free.pollFirst()`, `_registered.set(...)`
				for _, mm := range mutatingReceiverCallRe.FindAllStringSubmatch(trimmed, -1) {
					fld := mm[1]
					if uKnown[fld] && !uLocks[fld] {
						writes[fld] = true
					}
				}

				// Direct assignments / compound assignments / ++ / -- on fields
				// Avoid matching `==`, `!=`, `<=`, `>=`, `=>`
				sanitizedAssign := sanitizeComparisonOps(trimmed)
				// Skip matching the method header's own `def foo(...) =`
				if !(idx+1 == methodLine && lang == "scala") {
					for _, am := range assignFieldRe.FindAllStringSubmatch(sanitizedAssign, -1) {
						fld := am[1]
						if uKnown[fld] && !uLocks[fld] {
							writes[fld] = true
						}
					}
				}

				// Reads of known fields
				for _, im := range identTokenRe.FindAllStringSubmatch(trimmed, -1) {
					tok := im[1]
					if uKnown[tok] && !uLocks[tok] {
						reads[tok] = true
					}
				}

				// Intra-class / intra-module helper calls
				for _, cm := range intraHelperCallRe.FindAllStringSubmatch(trimmed, -1) {
					callee := cm[1]
					if callee != curMethod && !polyglotKeywords[callee] && !uKnown[callee] && !uLocks[callee] {
						calls[callee] = true
					}
				}
			}

			delta := countBraceDelta(line)
			braceDepth += delta
			if braceDepth < 0 {
				braceDepth = 0
			}

			if curMethod != "" && !awaitingMethodBody {
				if singleLineScalaDef {
					if delta == 0 && !strings.HasSuffix(trimmed, "=") {
						flushMethod()
					} else if delta > 0 {
						singleLineScalaDef = false
					}
				} else if braceDepth <= methodStartDepth && strings.Contains(line, "}") {
					flushMethod()
				}
			}

			if curClass != "" && braceDepth <= classBraceDepth && strings.Contains(line, "}") {
				flushMethod()
				curClass = ""
				classBraceDepth = -1
			}
		}
		flushMethod()
	}

	graph.Anomalies = detectCouplingAnomalies(graph, cleanFocusBase(focusFile))
	return graph, nil
}

func matchesPolyglotLang(name, lang string) bool {
	switch lang {
	case "java":
		return strings.HasSuffix(name, ".java")
	case "scala":
		return strings.HasSuffix(name, ".scala")
	case "typescript":
		return strings.HasSuffix(name, ".ts") || strings.HasSuffix(name, ".tsx")
	default:
		return false
	}
}

func isPolyglotTestOrDeclFile(name string) bool {
	return strings.HasSuffix(name, ".d.ts") ||
		strings.HasSuffix(name, ".test.ts") ||
		strings.HasSuffix(name, ".test.tsx") ||
		strings.HasSuffix(name, ".spec.ts") ||
		strings.HasSuffix(name, ".spec.tsx") ||
		strings.Contains(name, ".fixture.") ||
		strings.HasSuffix(name, "Test.java") ||
		strings.HasSuffix(name, "Tests.java") ||
		strings.HasSuffix(name, "Test.scala")
}

func matchPolyglotMethodHeader(trimmed, lang, curClass string) (name string, isPriv, syncMod, singleExpr bool) {
	if strings.Contains(trimmed, " static ") || strings.HasPrefix(trimmed, "static ") {
		return "", false, false, false
	}
	switch lang {
	case "java":
		if !strings.Contains(trimmed, "(") || strings.HasPrefix(trimmed, "if") || strings.HasPrefix(trimmed, "for") || strings.HasPrefix(trimmed, "while") || strings.HasPrefix(trimmed, "switch") || strings.HasPrefix(trimmed, "new ") {
			return "", false, false, false
		}
		if m := javaMethodDeclRe.FindStringSubmatch(trimmed); m != nil {
			n := m[1]
			if polyglotKeywords[n] || n == curClass {
				return "", false, false, false
			}
			priv := strings.Contains(trimmed, "private ")
			sync := strings.Contains(trimmed, "synchronized ")
			return n, priv, sync, false
		}
	case "scala":
		if m := scalaMethodDeclRe.FindStringSubmatch(trimmed); m != nil {
			n := m[1]
			if polyglotKeywords[n] {
				return "", false, false, false
			}
			priv := strings.Contains(trimmed, "private")
			sync := strings.Contains(trimmed, "synchronized")
			single := !strings.Contains(trimmed, "{")
			return n, priv, sync, single
		}
	case "typescript":
		if m := tsClassMethodRe.FindStringSubmatch(trimmed); m != nil {
			n := m[1]
			if polyglotKeywords[n] || n == "constructor" {
				return "", false, false, false
			}
			priv := strings.HasPrefix(trimmed, "private ") || strings.HasPrefix(n, "_") || strings.HasPrefix(n, "#")
			return n, priv, false, false
		}
	}
	return "", false, false, false
}

func matchTSModuleFuncHeader(trimmed string) (name string, isPriv bool) {
	if m := tsModuleFuncRe.FindStringSubmatch(trimmed); m != nil {
		n := m[1]
		// Ignore React PascalCase components when analyzing module singleton state
		if len(n) > 0 && unicode.IsUpper(rune(n[0])) {
			return "", false
		}
		priv := !strings.HasPrefix(trimmed, "export ") || strings.HasPrefix(n, "_")
		return n, priv
	}
	if m := tsArrowFuncRe.FindStringSubmatch(trimmed); m != nil {
		n := m[1]
		if len(n) > 0 && unicode.IsUpper(rune(n[0])) {
			return "", false
		}
		priv := !strings.HasPrefix(trimmed, "export ") || strings.HasPrefix(n, "_")
		return n, priv
	}
	return "", false
}

// stripCStyleCommentsPreserveLines strips // and /* ... */ comments and string literal
// interiors (including Scala s"...${fn("x")}..." and TS `...${fn("x")}...` nested
// interpolations) while preserving exact line counts and structural brace balance.
func stripCStyleCommentsPreserveLines(src string) []string {
	var out strings.Builder
	out.Grow(len(src))

	inBlockComment := false
	inLineComment := false
	type strFrame struct {
		quote       byte
		interpDepth int
	}
	var stack []strFrame

	for i := 0; i < len(src); i++ {
		ch := src[i]
		if ch == '\n' {
			inLineComment = false
			// Non-backtick unclosed single-line quotes reset at newline if not in ${...}
			if len(stack) > 0 && stack[len(stack)-1].quote != '`' && stack[len(stack)-1].interpDepth == 0 {
				stack = stack[:len(stack)-1]
			}
			out.WriteByte('\n')
			continue
		}
		if inLineComment {
			continue
		}
		if inBlockComment {
			if ch == '*' && i+1 < len(src) && src[i+1] == '/' {
				inBlockComment = false
				i++
			}
			continue
		}
		if len(stack) > 0 {
			top := &stack[len(stack)-1]
			if top.interpDepth == 0 {
				if ch == '\\' {
					i++
					continue
				}
				if (top.quote == '"' || top.quote == '`') && ch == '$' && i+1 < len(src) && src[i+1] == '{' {
					top.interpDepth = 1
					i++
					out.WriteString("  ")
					continue
				}
				if ch == top.quote {
					stack = stack[:len(stack)-1]
					out.WriteByte(ch)
				} else {
					out.WriteByte(' ')
				}
				continue
			}
			// Inside ${ ... } interpolation expression: keep braces balanced as spaces
			if ch == '\\' {
				i++
				continue
			}
			if ch == '"' || ch == '\'' || ch == '`' {
				stack = append(stack, strFrame{quote: ch, interpDepth: 0})
				out.WriteByte(' ')
				continue
			}
			if ch == '{' {
				top.interpDepth++
			} else if ch == '}' {
				top.interpDepth--
			}
			out.WriteByte(' ')
			continue
		}
		if ch == '/' && i+1 < len(src) {
			if src[i+1] == '/' {
				inLineComment = true
				i++
				continue
			}
			if src[i+1] == '*' {
				inBlockComment = true
				i++
				continue
			}
		}
		if ch == '"' || ch == '\'' || ch == '`' {
			stack = append(stack, strFrame{quote: ch, interpDepth: 0})
			out.WriteByte(ch)
			continue
		}
		out.WriteByte(ch)
	}
	return strings.Split(out.String(), "\n")
}

func countBraceDelta(cleanLine string) int {
	d := 0
	for i := 0; i < len(cleanLine); i++ {
		switch cleanLine[i] {
		case '{':
			d++
		case '}':
			d--
		}
	}
	return d
}

func sanitizeComparisonOps(s string) string {
	r := strings.NewReplacer("===", "   ", "!==", "   ", "==", "  ", "!=", "  ", "<=", "  ", ">=", "  ", "=>", "  ")
	return r.Replace(s)
}
