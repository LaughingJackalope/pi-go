package sense

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

var (
	pyClassRe     = regexp.MustCompile(`^(\s*)class\s+([A-Za-z_][A-Za-z0-9_]*)\b`)
	pyDefRe       = regexp.MustCompile(`^(\s*)def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*self\b`)
	pySelfLockRe  = regexp.MustCompile(`self\.([A-Za-z_][A-Za-z0-9_]*)\s*=\s*.*(?:Lock|RLock|Semaphore)\s*\(`)
	pyWithLockRe  = regexp.MustCompile(`\bwith\s+self\.([A-Za-z_][A-Za-z0-9_]*)\b|self\.([A-Za-z_][A-Za-z0-9_]*)\.acquire\s*\(`)
	pySelfWriteRe = regexp.MustCompile(`self\.([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[[^\]]+\])*\s*(?:[+\-*/|&^]?=|\.(?:pop|clear|update|append|extend|remove|discard|add|setdefault)\s*\()`)
	pyTupleLHSRe  = regexp.MustCompile(`^(\s*(?:self\.[A-Za-z_][A-Za-z0-9_]*\s*,\s*)+self\.[A-Za-z_][A-Za-z0-9_]*)\s*=`)
	pyDelSelfRe   = regexp.MustCompile(`\bdel\s+self\.([A-Za-z_][A-Za-z0-9_]*)\b`)
	pySelfCallRe  = regexp.MustCompile(`self\.([A-Za-z_][A-Za-z0-9_]*)\s*\(`)
	pySelfReadRe  = regexp.MustCompile(`self\.([A-Za-z_][A-Za-z0-9_]*)\b`)
)

// AnalyzePythonDir parses all non-test Python (.py) files in dir and extracts
// the implicit state-machine coupling graph across Python class self.<field>
// attributes, lock guards (with self.lock:), and method reads/writes.
func AnalyzePythonDir(dir string, focusFile string) (*PackageStateGraph, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	graph := &PackageStateGraph{
		PackageName:  filepath.Base(dir),
		StructFields: make(map[string][]string),
		LockFields:   make(map[string][]string),
	}

	type rawPyFile struct {
		name  string
		lines []string
	}
	var pyFiles []rawPyFile
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".py") || strings.HasPrefix(name, "test_") || strings.HasSuffix(name, "_test.py") {
			continue
		}
		blob, readErr := os.ReadFile(filepath.Join(dir, name))
		if readErr != nil {
			continue
		}
		pyFiles = append(pyFiles, rawPyFile{
			name:  name,
			lines: strings.Split(string(blob), "\n"),
		})
	}

	knownFieldsByClass := make(map[string]map[string]bool)
	lockFieldsByClass := make(map[string]map[string]bool)

	// Pass 1: Discover classes, self.<field> definitions, and lock attributes.
	for _, f := range pyFiles {
		curClass := ""
		classIndent := 0
		for _, rawLine := range f.lines {
			line := stripPyComment(rawLine)
			if strings.TrimSpace(line) == "" {
				continue
			}
			indent := countIndent(line)
			if m := pyClassRe.FindStringSubmatch(line); m != nil {
				curClass = m[2]
				classIndent = len(m[1])
				if knownFieldsByClass[curClass] == nil {
					knownFieldsByClass[curClass] = make(map[string]bool)
					lockFieldsByClass[curClass] = make(map[string]bool)
				}
				continue
			}
			if curClass != "" && indent <= classIndent {
				curClass = ""
			}
			if curClass == "" {
				continue
			}
			for _, lm := range pySelfLockRe.FindAllStringSubmatch(line, -1) {
				fld := lm[1]
				lockFieldsByClass[curClass][fld] = true
				knownFieldsByClass[curClass][fld] = true
			}
			for _, wm := range pySelfWriteRe.FindAllStringSubmatch(line, -1) {
				fld := wm[1]
				if fld == "lock" || fld == "mu" || fld == "_lock" {
					lockFieldsByClass[curClass][fld] = true
				}
				knownFieldsByClass[curClass][fld] = true
			}
			if tm := pyTupleLHSRe.FindStringSubmatch(line); tm != nil {
				for _, rm := range pySelfReadRe.FindAllStringSubmatch(tm[1], -1) {
					fld := rm[1]
					if fld == "lock" || fld == "mu" || fld == "_lock" {
						lockFieldsByClass[curClass][fld] = true
					}
					knownFieldsByClass[curClass][fld] = true
				}
			}
		}
	}

	for cls, fset := range knownFieldsByClass {
		graph.StructFields[cls] = sortedKeys(fset)
		graph.LockFields[cls] = sortedKeys(lockFieldsByClass[cls])
	}

	// Pass 2: Extract per-method self.<field> reads, writes, and lock usage.
	for _, f := range pyFiles {
		curClass := ""
		classIndent := 0
		curMethod := ""
		methodIndent := 0
		methodLine := 0
		var reads map[string]bool
		var writes map[string]bool
		var calls map[string]bool
		usesLock := false

		flushMethod := func() {
			if curClass != "" && curMethod != "" {
				if curMethod == "__init__" {
					// Record __init__'s intra-class helper calls (e.g. _reconcile_adopted) without
					// counting constructor field initializations as runtime state mutations.
					if len(calls) > 0 {
						graph.Methods = append(graph.Methods, MethodTouch{
							StructName: curClass,
							MethodName: "__init__",
							File:       f.name,
							Line:       methodLine,
							Calls:      sortedKeys(calls),
							UsesLock:   true,
							IsPrivate:  true,
						})
					}
				} else {
					graph.Methods = append(graph.Methods, MethodTouch{
						StructName: curClass,
						MethodName: curMethod,
						File:       f.name,
						Line:       methodLine,
						Reads:      sortedKeys(reads),
						Writes:     sortedKeys(writes),
						Calls:      sortedKeys(calls),
						UsesLock:   usesLock,
						IsPrivate:  strings.HasPrefix(curMethod, "_"),
					})
				}
			}
			curMethod = ""
		}

		for idx, rawLine := range f.lines {
			line := stripPyComment(rawLine)
			if strings.TrimSpace(line) == "" {
				continue
			}
			indent := countIndent(line)

			if m := pyClassRe.FindStringSubmatch(line); m != nil {
				flushMethod()
				curClass = m[2]
				classIndent = len(m[1])
				continue
			}
			if curClass != "" && indent <= classIndent {
				flushMethod()
				curClass = ""
			}
			if curClass == "" {
				continue
			}

			if m := pyDefRe.FindStringSubmatch(line); m != nil {
				if len(m[1]) > classIndent {
					flushMethod()
					curMethod = m[2]
					methodIndent = len(m[1])
					methodLine = idx + 1
					reads = make(map[string]bool)
					writes = make(map[string]bool)
					calls = make(map[string]bool)
					usesLock = false
					continue
				}
			}
			if curMethod != "" && indent <= methodIndent {
				flushMethod()
			}
			if curMethod == "" {
				continue
			}

			known := knownFieldsByClass[curClass]
			locks := lockFieldsByClass[curClass]

			if lm := pyWithLockRe.FindStringSubmatch(line); lm != nil {
				lf := lm[1]
				if lf == "" {
					lf = lm[2]
				}
				if locks[lf] || lf == "lock" || lf == "mu" || lf == "_lock" {
					usesLock = true
				}
			}

			for _, wm := range pySelfWriteRe.FindAllStringSubmatch(line, -1) {
				fld := wm[1]
				if known[fld] && !locks[fld] {
					writes[fld] = true
				}
			}
			if tm := pyTupleLHSRe.FindStringSubmatch(line); tm != nil {
				for _, rm := range pySelfReadRe.FindAllStringSubmatch(tm[1], -1) {
					fld := rm[1]
					if known[fld] && !locks[fld] {
						writes[fld] = true
					}
				}
			}
			for _, dm := range pyDelSelfRe.FindAllStringSubmatch(line, -1) {
				fld := dm[1]
				if known[fld] && !locks[fld] {
					writes[fld] = true
				}
			}
			for _, rm := range pySelfReadRe.FindAllStringSubmatch(line, -1) {
				fld := rm[1]
				if known[fld] && !locks[fld] {
					reads[fld] = true
				}
			}
			for _, cm := range pySelfCallRe.FindAllStringSubmatch(line, -1) {
				callee := cm[1]
				if !known[callee] && !locks[callee] {
					calls[callee] = true
				}
			}
		}
		flushMethod()
	}

	graph.Anomalies = detectCouplingAnomalies(graph, cleanFocusBase(focusFile))
	return graph, nil
}

func stripPyComment(line string) string {
	inSingle := false
	inDouble := false
	for i := 0; i < len(line); i++ {
		ch := line[i]
		if ch == '\\' {
			i++
			continue
		}
		if ch == '\'' && !inDouble {
			inSingle = !inSingle
		} else if ch == '"' && !inSingle {
			inDouble = !inDouble
		} else if ch == '#' && !inSingle && !inDouble {
			return line[:i]
		}
	}
	return line
}

func countIndent(line string) int {
	n := 0
	for _, ch := range line {
		if ch == ' ' {
			n++
		} else if ch == '\t' {
			n += 4
		} else {
			break
		}
	}
	return n
}
