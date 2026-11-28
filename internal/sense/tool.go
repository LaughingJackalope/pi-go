package sense

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/tool"
	"google.golang.org/adk/v2/tool/functiontool"
)

// SenseMapInput defines the parameters for the sense_map Pre-Flight Radar tool.
type SenseMapInput struct {
	// Path to a Go (.go) file, Python (.py) file, or package directory to scan
	// for implicit state-machine invariants, coupled struct/class fields, and mutex guards.
	Path string `json:"path"`
}

// SenseMapOutput contains the zero-token S3 Pre-Flight Radar state-coupling map.
type SenseMapOutput struct {
	Target       string            `json:"target"`
	Language     string            `json:"language"`
	RadarSummary string            `json:"radar_summary"`
	StructFields map[string][]string `json:"struct_fields,omitempty"`
	LockFields   map[string][]string `json:"lock_fields,omitempty"`
	Anomalies    []CouplingAnomaly `json:"anomalies,omitempty"`
}

// NewSenseMapTool creates the zero-token S3 Pre-Flight Radar tool ("sense_map")
// so an agent can inspect the implicit state-machine coupling map of a Go, Python,
// TypeScript, Java, or Scala package/module before making an edit.
func NewSenseMapTool() (tool.Tool, error) {
	return functiontool.New(functiontool.Config{
		Name: "sense_map",
		Description: `S3 Pre-Flight Radar: Extracts the implicit state-machine coupling map (mutex-guarded fields, co-mutated coupled state fields, and active state desync anomalies) for Go (.go), Python (.py), TypeScript (.ts/.tsx), Java (.java), or Scala (.scala) files or directories in <2ms at 0 tokens. Call this before editing unfamiliar structs or stateful classes.`,
	}, func(_ agent.Context, input SenseMapInput) (SenseMapOutput, error) {
		p := strings.TrimSpace(input.Path)
		if p == "" {
			return SenseMapOutput{}, fmt.Errorf("path is required")
		}
		info, err := os.Stat(p)
		if err != nil {
			return SenseMapOutput{}, err
		}

		dir := p
		focusFile := ""
		if !info.IsDir() {
			dir = filepath.Dir(p)
			focusFile = p
		}

		graph, lang, err := AnalyzeAnyDir(dir, focusFile)
		if err != nil {
			return SenseMapOutput{}, err
		}

		radar := BuildPreFlightRadar(p, graph)
		if radar == "" {
			radar = fmt.Sprintf("[S3 PRE-FLIGHT RADAR · %s] No coupled state fields or mutex guards detected.", filepath.Base(p))
		}

		return SenseMapOutput{
			Target:       p,
			Language:     lang,
			RadarSummary: radar,
			StructFields: graph.StructFields,
			LockFields:   graph.LockFields,
			Anomalies:    graph.Anomalies,
		}, nil
	})
}

// BuildPreFlightRadar summarizes mutex-protected fields and coupled state invariants
// across all structs/classes in graph. Injected automatically into read tool results
// under result["s3_preflight_radar"] so even a narrow offset/limit read sees the
// full package's state invariants before the first edit.
func BuildPreFlightRadar(filePath string, g *PackageStateGraph) string {
	if g == nil || len(g.Methods) == 0 {
		return ""
	}
	expandIntraClassCalls(g)

	focusBase := ""
	if info, err := os.Stat(filePath); err == nil && !info.IsDir() {
		focusBase = filepath.Base(filePath)
	} else if ext := filepath.Ext(filePath); ext != "" {
		focusBase = filepath.Base(filePath)
	}

	structsInFocusFile := make(map[string]bool)
	byStruct := make(map[string][]MethodTouch)
	for _, m := range g.Methods {
		byStruct[m.StructName] = append(byStruct[m.StructName], m)
		if focusBase == "" || m.File == focusBase {
			structsInFocusFile[m.StructName] = true
		}
	}

	var lines []string
	structNames := make([]string, 0, len(byStruct))
	for sName := range byStruct {
		if len(structsInFocusFile) > 0 && !structsInFocusFile[sName] {
			continue
		}
		structNames = append(structNames, sName)
	}
	sort.Strings(structNames)

	for _, sName := range structNames {
		methods := byStruct[sName]
		var structLines []string
		if locks := g.LockFields[sName]; len(locks) > 0 {
			lockedFields := make(map[string]bool)
			for _, m := range methods {
				if m.UsesLock {
					for _, w := range m.Writes {
						lockedFields[w] = true
					}
				}
			}
			if len(lockedFields) > 0 {
				structLines = append(structLines, fmt.Sprintf(
					"  • Mutex guard: %s.%s protects writes to %v",
					sName, locks[0], sortedKeys(lockedFields),
				))
			}
		}

		// Find coupled field pairs (co-written in >=2 methods and co-read in >=2 methods)
		coWrite := make(map[[2]string][]string)
		coRead := make(map[[2]string]int)
		for _, m := range methods {
			for _, w1 := range m.Writes {
				for _, w2 := range m.Writes {
					if w1 < w2 {
						coWrite[[2]string{w1, w2}] = append(coWrite[[2]string{w1, w2}], m.MethodName)
					}
				}
			}
			allTouched := make(map[string]bool)
			for _, r := range m.Reads {
				allTouched[r] = true
			}
			for _, w := range m.Writes {
				allTouched[w] = true
			}
			tList := sortedKeys(allTouched)
			for _, r1 := range tList {
				for _, r2 := range tList {
					if r1 < r2 {
						coRead[[2]string{r1, r2}]++
					}
				}
			}
		}

		for pair, peers := range coWrite {
			if len(peers) >= 2 && coRead[pair] >= 2 {
				structLines = append(structLines, fmt.Sprintf(
					"  • Coupled state invariant: %s.%s <-> %s.%s (co-mutated in %s)",
					sName, pair[0], sName, pair[1], strings.Join(peers, ", "),
				))
			}
		}

		if len(structLines) > 0 {
			lines = append(lines, fmt.Sprintf("Struct/Class %s State Invariants:", sName))
			lines = append(lines, structLines...)
		}
	}

	if len(lines) == 0 {
		return ""
	}
	header := fmt.Sprintf("[S3 PRE-FLIGHT RADAR · %s]\n", filepath.Base(filePath))
	return header + strings.Join(lines, "\n")
}
