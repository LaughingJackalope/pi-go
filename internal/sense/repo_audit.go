package sense

import (
	"context"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"
)

// DiscoveredInvariant summarizes a mutex guard or coupled field pair extracted from a repo.
type DiscoveredInvariant struct {
	Language    string   `json:"language"`
	RelDir      string   `json:"rel_dir"`
	UnitName    string   `json:"unit_name"`
	Kind        string   `json:"kind"` // "MUTEX_GUARD" | "COUPLED_STATE_PAIR"
	Fields      []string `json:"fields"`
	GuardOrPeer string   `json:"guard_or_peer,omitempty"`
	Methods     []string `json:"methods,omitempty"`
}

// LanguageAuditStats tracks scanned files, classes/structs, invariants, and anomalies per language.
type LanguageAuditStats struct {
	FilesScanned      int `json:"files_scanned"`
	DirsAnalyzed      int `json:"dirs_analyzed"`
	UnitsAnalyzed     int `json:"units_analyzed"`
	MethodsAnalyzed   int `json:"methods_analyzed"`
	MutexGuards       int `json:"mutex_guards"`
	CoupledStatePairs int `json:"coupled_state_pairs"`
	AnomaliesCount    int `json:"anomalies_count"`
}

// RepoAnomalyFinding enriches a CouplingAnomaly with its repo-relative directory and language.
type RepoAnomalyFinding struct {
	Language string          `json:"language"`
	RelDir   string          `json:"rel_dir"`
	Anomaly  CouplingAnomaly `json:"anomaly"`
}

// RepoAuditReport holds the zero-token multi-language S3 structural audit of one or more paths.
type RepoAuditReport struct {
	Targets           []string                      `json:"targets"`
	ElapsedMs         float64                       `json:"elapsed_ms"`
	FilesScanned      int                           `json:"files_scanned"`
	DirsAnalyzed      int                           `json:"dirs_analyzed"`
	UnitsAnalyzed     int                           `json:"units_analyzed"`
	MethodsAnalyzed   int                           `json:"methods_analyzed"`
	MutexGuardsCount  int                           `json:"mutex_guards_count"`
	CoupledPairsCount int                           `json:"coupled_pairs_count"`
	ByLanguage        map[string]LanguageAuditStats `json:"by_language"`
	Invariants        []DiscoveredInvariant         `json:"invariants,omitempty"`
	Anomalies         []RepoAnomalyFinding          `json:"anomalies,omitempty"`
	S3Posture         string                        `json:"s3_posture,omitempty"`
	S3DesyncProb      float64                       `json:"s3_desync_prob,omitempty"`
	S3ActProb         float64                       `json:"s3_act_prob,omitempty"`
	S3LatencyMs       float64                       `json:"s3_latency_ms,omitempty"`
}

var auditSkippedDirs = map[string]bool{
	".git": true, "vendor": true, "node_modules": true, "__pycache__": true,
	".venv": true, "venv": true, "dist": true, "build": true, "target": true,
	"out": true, "testdata": true, "tests": true, "tests-js": true, "test": true,
	"testing": true, "website": true, "locales": true, "contributors": true,
	"optional-skills": true, "plugin-catalog": true, "assets": true,
}

// DetectFileLanguage maps a source filename to a supported S3 sensor language.
func DetectFileLanguage(name string) string {
	switch {
	case strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, "_test.go"):
		return "go"
	case strings.HasSuffix(name, ".py") && !strings.HasPrefix(name, "test_") && !strings.HasSuffix(name, "_test.py"):
		return "python"
	case strings.HasSuffix(name, ".java") && !isPolyglotTestOrDeclFile(name):
		return "java"
	case strings.HasSuffix(name, ".scala") && !isPolyglotTestOrDeclFile(name):
		return "scala"
	case (strings.HasSuffix(name, ".ts") || strings.HasSuffix(name, ".tsx")) && !isPolyglotTestOrDeclFile(name):
		return "typescript"
	default:
		return ""
	}
}

// AnalyzeAnyDir dispatches to the appropriate language sensor (Go, Python, Java, Scala, TypeScript).
func AnalyzeAnyDir(dir, focusFile string) (*PackageStateGraph, string, error) {
	lang := DetectFileLanguage(filepath.Base(focusFile))
	if lang == "" {
		entries, err := os.ReadDir(dir)
		if err != nil {
			return nil, "", err
		}
		counts := make(map[string]int)
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			if l := DetectFileLanguage(e.Name()); l != "" {
				counts[l]++
			}
		}
		bestCount := 0
		for _, cand := range []string{"go", "python", "java", "scala", "typescript"} {
			if counts[cand] > bestCount {
				bestCount = counts[cand]
				lang = cand
			}
		}
	}
	if lang == "" {
		lang = "go"
	}
	return AnalyzeDirForLanguage(dir, focusFile, lang)
}

// AnalyzeDirForLanguage runs the specific language state-coupling sensor on dir.
func AnalyzeDirForLanguage(dir, focusFile, lang string) (*PackageStateGraph, string, error) {
	switch lang {
	case "go":
		g, err := AnalyzePackageDir(dir, focusFile)
		return g, "go", err
	case "python":
		g, err := AnalyzePythonDir(dir, focusFile)
		return g, "python", err
	case "java", "scala", "typescript":
		g, err := AnalyzePolyglotDir(dir, focusFile, lang)
		return g, lang, err
	default:
		g, err := AnalyzePackageDir(dir, focusFile)
		return g, "go", err
	}
}

// AuditRepository walks target files/directories in parallel across Go, Python,
// TypeScript, Java, and Scala, extracting implicit state-machine invariants and
// state-desync anomalies, then evaluates the aggregate telemetry via Engine (s3-cortexd).
func (e *Engine) AuditRepository(ctx context.Context, targets []string, maxFindings, maxInvariants int) (*RepoAuditReport, error) {
	started := time.Now()
	if len(targets) == 0 {
		targets = []string{"."}
	}
	if maxFindings <= 0 {
		maxFindings = 25
	}
	if maxInvariants <= 0 {
		maxInvariants = 25
	}

	type dirJob struct {
		root      string
		dir       string
		focusFile string
		lang      string
		fileCount int
	}

	jobMap := make(map[string]*dirJob)
	var roots []string

	for _, t := range targets {
		info, err := os.Stat(t)
		if err != nil {
			continue
		}
		absT, _ := filepath.Abs(t)
		if !info.IsDir() {
			lang := DetectFileLanguage(filepath.Base(absT))
			if lang == "" {
				continue
			}
			dir := filepath.Dir(absT)
			roots = append(roots, dir)
			key := dir + "|" + lang + "|" + absT
			jobMap[key] = &dirJob{root: dir, dir: dir, focusFile: absT, lang: lang, fileCount: 1}
			continue
		}

		roots = append(roots, absT)
		_ = filepath.WalkDir(absT, func(p string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return nil
			}
			name := d.Name()
			if d.IsDir() {
				if p != absT && (strings.HasPrefix(name, ".") || auditSkippedDirs[name]) {
					return filepath.SkipDir
				}
				return nil
			}
			lang := DetectFileLanguage(name)
			if lang == "" {
				return nil
			}
			dir := filepath.Dir(p)
			key := dir + "|" + lang
			j := jobMap[key]
			if j == nil {
				j = &dirJob{root: absT, dir: dir, lang: lang}
				jobMap[key] = j
			}
			j.fileCount++
			return nil
		})
	}

	jobs := make([]*dirJob, 0, len(jobMap))
	for _, j := range jobMap {
		jobs = append(jobs, j)
	}
	sort.Slice(jobs, func(i, k int) bool {
		if jobs[i].dir != jobs[k].dir {
			return jobs[i].dir < jobs[k].dir
		}
		return jobs[i].lang < jobs[k].lang
	})

	type jobOut struct {
		job   *dirJob
		graph *PackageStateGraph
		invs  []DiscoveredInvariant
	}

	workers := min(max(2, runtime.NumCPU()), 8)
	jobCh := make(chan *dirJob, len(jobs))
	outCh := make(chan jobOut, len(jobs))
	for _, j := range jobs {
		jobCh <- j
	}
	close(jobCh)

	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobCh {
				g, _, err := AnalyzeDirForLanguage(j.dir, j.focusFile, j.lang)
				if err != nil || g == nil {
					continue
				}
				relDir, rErr := filepath.Rel(j.root, j.dir)
				if rErr != nil || relDir == "" {
					relDir = "."
				}
				invs := extractGraphInvariants(g, j.lang, relDir)
				outCh <- jobOut{job: j, graph: g, invs: invs}
			}
		}()
	}
	wg.Wait()
	close(outCh)

	report := &RepoAuditReport{
		Targets:    targets,
		ByLanguage: make(map[string]LanguageAuditStats),
	}

	var allInvs []DiscoveredInvariant
	var allAnoms []RepoAnomalyFinding

	for res := range outCh {
		lang := res.job.lang
		st := report.ByLanguage[lang]
		st.FilesScanned += res.job.fileCount
		st.DirsAnalyzed++
		st.UnitsAnalyzed += len(res.graph.StructFields)
		st.MethodsAnalyzed += len(res.graph.Methods)

		for _, inv := range res.invs {
			if inv.Kind == "MUTEX_GUARD" {
				st.MutexGuards++
				report.MutexGuardsCount++
			} else {
				st.CoupledStatePairs++
				report.CoupledPairsCount++
			}
		}
		st.AnomaliesCount += len(res.graph.Anomalies)
		report.ByLanguage[lang] = st

		report.FilesScanned += res.job.fileCount
		report.DirsAnalyzed++
		report.UnitsAnalyzed += len(res.graph.StructFields)
		report.MethodsAnalyzed += len(res.graph.Methods)

		allInvs = append(allInvs, res.invs...)
		relDir, rErr := filepath.Rel(res.job.root, res.job.dir)
		if rErr != nil || relDir == "" {
			relDir = "."
		}
		for _, a := range res.graph.Anomalies {
			allAnoms = append(allAnoms, RepoAnomalyFinding{
				Language: lang,
				RelDir:   relDir,
				Anomaly:  a,
			})
		}
	}

	sort.Slice(allInvs, func(i, k int) bool {
		if allInvs[i].Kind != allInvs[k].Kind {
			return allInvs[i].Kind == "COUPLED_STATE_PAIR"
		}
		if allInvs[i].RelDir != allInvs[k].RelDir {
			return allInvs[i].RelDir < allInvs[k].RelDir
		}
		return allInvs[i].UnitName < allInvs[k].UnitName
	})
	if len(allInvs) > maxInvariants {
		allInvs = allInvs[:maxInvariants]
	}
	report.Invariants = allInvs

	sort.Slice(allAnoms, func(i, k int) bool {
		if allAnoms[i].Anomaly.Kind != allAnoms[k].Anomaly.Kind {
			return allAnoms[i].Anomaly.Kind == "UNLOCKED_STATE_WRITE"
		}
		if allAnoms[i].RelDir != allAnoms[k].RelDir {
			return allAnoms[i].RelDir < allAnoms[k].RelDir
		}
		return allAnoms[i].Anomaly.MethodName < allAnoms[k].Anomaly.MethodName
	})
	if len(allAnoms) > maxFindings {
		allAnoms = allAnoms[:maxFindings]
	}
	report.Anomalies = allAnoms

	// Score aggregate audit posture through live Dual-Tier s3-cortexd if available
	if e != nil {
		s3Start := time.Now()
		hasAnom := len(report.Anomalies) > 0
		pref := PostureProceed
		if hasAnom {
			pref = PostureInspect
		}
		stateStr := "State desync detected: " + yesNo(hasAnom) + ". Static analysis incomplete: no. Durable engineering change: no."
		criteria := map[string]string{
			PostureProceed: "Suboptimal action.",
			PostureVerify:  "Suboptimal action.",
			PostureInspect: "Suboptimal action.",
			PostureBranch:  "Suboptimal action.",
		}
		criteria[pref] = "Safe. Best tactical coding sense action."
		if hasAnom {
			criteria[PostureProceed] = "Unsafe. Ignores active state desync or error."
		}
		questions := map[string]any{
			"posture": map[string]any{
				"type":         "choice",
				"instructions": "Choose the best coding agent sense action.",
				"criteria":     criteria,
			},
			"desync": map[string]any{
				"type":         "noul",
				"instructions": "Is a state desync detected?",
				"criteria": map[string]string{
					"true":  "state desync, compiler error, or missing coupled field write",
					"false": "no state desync, clean routine read or valid change",
				},
			},
		}
		probs, dProb, _, srr := e.callSystem1(ctx, stateStr, questions, pref, hasAnom, false)
		chosen := maxKey(probs, pref)
		if hasAnom {
			chosen = PostureInspect
		}
		report.S3Posture = chosen
		report.S3DesyncProb = dProb
		report.S3ActProb = srr.ActProb
		report.S3LatencyMs = float64(time.Since(s3Start).Microseconds()) / 1000.0
	}

	report.ElapsedMs = float64(time.Since(started).Microseconds()) / 1000.0
	return report, nil
}

func extractGraphInvariants(g *PackageStateGraph, lang, relDir string) []DiscoveredInvariant {
	if g == nil || len(g.Methods) == 0 {
		return nil
	}
	byStruct := make(map[string][]MethodTouch)
	for _, m := range g.Methods {
		byStruct[m.StructName] = append(byStruct[m.StructName], m)
	}

	var out []DiscoveredInvariant
	for sName, methods := range byStruct {
		if locks := g.LockFields[sName]; len(locks) > 0 {
			lockedFields := make(map[string]bool)
			var lockMethods []string
			for _, m := range methods {
				if m.UsesLock && len(m.Writes) > 0 {
					lockMethods = append(lockMethods, m.MethodName)
					for _, w := range m.Writes {
						lockedFields[w] = true
					}
				}
			}
			if len(lockedFields) > 0 {
				out = append(out, DiscoveredInvariant{
					Language:    lang,
					RelDir:      relDir,
					UnitName:    sName,
					Kind:        "MUTEX_GUARD",
					Fields:      sortedKeys(lockedFields),
					GuardOrPeer: locks[0],
					Methods:     lockMethods,
				})
			}
		}

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
			touched := make(map[string]bool)
			for _, r := range m.Reads {
				touched[r] = true
			}
			for _, w := range m.Writes {
				touched[w] = true
			}
			tList := sortedKeys(touched)
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
				out = append(out, DiscoveredInvariant{
					Language: lang,
					RelDir:   relDir,
					UnitName: sName,
					Kind:     "COUPLED_STATE_PAIR",
					Fields:   []string{pair[0], pair[1]},
					Methods:  peers,
				})
			}
		}
	}
	return out
}
