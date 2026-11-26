package sense

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// StateSnapshot records a structural AST/token hash of a file at a specific edit step.
type StateSnapshot struct {
	Step    int    `json:"step"`
	Sig     string `json:"sig"`
	Summary string `json:"summary"`
}

// StructuralSignature computes a comment- and whitespace-invariant structural hash
// of a source file in <0.2ms:
//   - For Go (.go): streams all lexical tokens via go/scanner (skipping comments
//     and raw whitespace/formatting) into SHA-256.
//   - For Python (.py) and other languages: strips comments, collapses horizontal
//     whitespace, drops blank lines, and hashes the normalized statement stream.
func StructuralSignature(filePath, content string) string {
	h := sha256.New()
	if strings.HasSuffix(filePath, ".go") {
		fset := token.NewFileSet()
		file := fset.AddFile(filePath, fset.Base(), len(content))
		var s scanner.Scanner
		// Mode 0 omits scanner.ScanComments, so comments are stripped lexically.
		s.Init(file, []byte(content), nil, 0)
		for {
			_, tok, lit := s.Scan()
			if tok == token.EOF {
				break
			}
			_, _ = h.Write([]byte(tok.String()))
			_, _ = h.Write([]byte{0})
			_, _ = h.Write([]byte(lit))
			_, _ = h.Write([]byte{'\n'})
		}
		return hex.EncodeToString(h.Sum(nil))[:16]
	}

	// Python / generic normalized structural signature
	for _, rawLine := range strings.Split(content, "\n") {
		line := stripPyComment(rawLine)
		norm := strings.Join(strings.Fields(line), " ")
		if norm == "" {
			continue
		}
		_, _ = h.Write([]byte(norm))
		_, _ = h.Write([]byte{'\n'})
	}
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// ECCMAutoLockEdit inspects an incoming "edit" tool call's args["old_string"]
// before the edit tool runs. If old_string is not an exact byte match in the
// target file due to whitespace, tab-vs-space indentation drift, or double-escaped
// newlines, ECCM locates the unique matching code block in the file and snaps
// args["old_string"] (and indentation of args["new_string"]) onto the true file bytes.
func ECCMAutoLockEdit(args map[string]any) (locked bool, summary string) {
	if args == nil {
		return false, ""
	}
	filePath, _ := args["file_path"].(string)
	if filePath == "" {
		filePath, _ = args["path"].(string)
	}
	oldStr, _ := args["old_string"].(string)
	newStr, _ := args["new_string"].(string)
	if filePath == "" || oldStr == "" {
		return false, ""
	}

	rawBytes, err := os.ReadFile(filePath)
	if err != nil {
		return false, ""
	}
	content := string(rawBytes)

	// Fast path: exact match already exists in file.
	if strings.Count(content, oldStr) > 0 {
		return false, ""
	}

	// Pass 1: Check for literal double-escaped \n or \t from weak models.
	if strings.Contains(oldStr, `\n`) || strings.Contains(oldStr, `\t`) {
		unescapedOld := strings.ReplaceAll(strings.ReplaceAll(oldStr, `\n`, "\n"), `\t`, "\t")
		if strings.Count(content, unescapedOld) == 1 {
			args["old_string"] = unescapedOld
			if strings.Contains(newStr, `\n`) || strings.Contains(newStr, `\t`) {
				args["new_string"] = strings.ReplaceAll(strings.ReplaceAll(newStr, `\n`, "\n"), `\t`, "\t")
			}
			return true, fmt.Sprintf(
				"ECCM Auto-Lock (%s): repaired double-escaped \\n/\\t in old_string",
				filepath.Base(filePath),
			)
		}
	}

	// Pass 2: Whitespace- and indentation-normalized unique line-block lock.
	oldLines := trimBlankEdges(strings.Split(strings.ReplaceAll(oldStr, "\r\n", "\n"), "\n"))
	if len(oldLines) == 0 {
		return false, ""
	}
	normOld := make([]string, len(oldLines))
	for i, l := range oldLines {
		normOld[i] = normalizeCodeLine(l)
	}

	fileLines := strings.Split(content, "\n")
	type matchSpan struct {
		startIdx int // inclusive
		endIdx   int // inclusive
	}
	var matches []matchSpan

	for i := 0; i < len(fileLines); i++ {
		if normalizeCodeLine(fileLines[i]) != normOld[0] {
			continue
		}
		// Try to match remaining oldLines against fileLines starting at i,
		// allowing blank lines in either to be skipped only if needed.
		fIdx := i
		oIdx := 0
		ok := true
		for oIdx < len(normOld) {
			if fIdx >= len(fileLines) {
				ok = false
				break
			}
			if normOld[oIdx] == "" {
				oIdx++
				continue
			}
			fNorm := normalizeCodeLine(fileLines[fIdx])
			if fNorm == "" {
				fIdx++
				continue
			}
			if fNorm != normOld[oIdx] {
				ok = false
				break
			}
			oIdx++
			fIdx++
		}
		if ok {
			matches = append(matches, matchSpan{startIdx: i, endIdx: fIdx - 1})
		}
	}

	if len(matches) != 1 {
		return false, ""
	}

	m := matches[0]
	lockedOld := strings.Join(fileLines[m.startIdx:m.endIdx+1], "\n")
	args["old_string"] = lockedOld

	// If the matched file line uses tabs for indentation and new_string used spaces,
	// harmonize leading 4-space groups to tabs on Go files so gofmt stays clean.
	if strings.HasSuffix(filePath, ".go") && strings.HasPrefix(fileLines[m.startIdx], "\t") && strings.HasPrefix(newStr, "    ") {
		args["new_string"] = harmonizeGoTabs(newStr)
	}

	return true, fmt.Sprintf(
		"ECCM Auto-Lock (%s:%d-%d): snapped whitespace-drifted old_string onto unique target block",
		filepath.Base(filePath), m.startIdx+1, m.endIdx+1,
	)
}

func normalizeCodeLine(s string) string {
	return strings.Join(strings.Fields(s), " ")
}

func trimBlankEdges(lines []string) []string {
	start := 0
	for start < len(lines) && strings.TrimSpace(lines[start]) == "" {
		start++
	}
	end := len(lines)
	for end > start && strings.TrimSpace(lines[end-1]) == "" {
		end--
	}
	return lines[start:end]
}

func harmonizeGoTabs(s string) string {
	lines := strings.Split(s, "\n")
	for i, l := range lines {
		spaces := 0
		for spaces < len(l) && l[spaces] == ' ' {
			spaces++
		}
		if spaces >= 4 {
			tabs := spaces / 4
			rem := spaces % 4
			lines[i] = strings.Repeat("\t", tabs) + strings.Repeat(" ", rem) + l[spaces:]
		}
	}
	return strings.Join(lines, "\n")
}
