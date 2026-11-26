package sense

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/tool"

	"github.com/dimetron/pi-go/internal/notice"
)

// Posture represents the 4 single-word tactical sense-making actions scored by System 1.
const (
	PostureProceed = "proceed" // Clean progress; continue normal flow
	PostureVerify  = "verify"  // Code modified; run build/test next
	PostureInspect = "inspect" // BULLET TIME: Implicit state desync or LSP diagnostic regression
	PostureBranch  = "branch"  // Oscillating edit loop or high blast radius; checkpoint/branch
)

// SenseVerdict is injected into the tool result map under result["system1_sense"]
// and result["bullet_time_advisory"] when Bullet Time engages.
type SenseVerdict struct {
	Posture              string             `json:"posture"`
	Probabilities        map[string]float64 `json:"probabilities"`
	DesyncProb           float64            `json:"desync_prob"`
	ActProb              float64            `json:"act_prob"`
	RescueProb           float64            `json:"rescue_prob,omitempty"`
	HarmProb             float64            `json:"harm_prob,omitempty"`
	AmbigProb            float64            `json:"ambig_prob,omitempty"`
	SignedGain           float64            `json:"signed_gain,omitempty"`
	RiskTier             string             `json:"risk_tier,omitempty"`
	RecommendedRole      string             `json:"recommended_role"`
	RetainMemory         bool               `json:"retain_memory"`
	RetainProb           float64            `json:"retain_prob"`
	BulletTimeEngaged    bool               `json:"bullet_time_engaged"`
	ECCMAutoLocked       bool               `json:"eccm_auto_locked,omitempty"`
	ECCMSummary          string             `json:"eccm_summary,omitempty"`
	WarpScramblerEngaged bool               `json:"warp_scrambler_engaged,omitempty"`
	CycleSummary         string             `json:"cycle_summary,omitempty"`
	Anomalies            []CouplingAnomaly  `json:"anomalies,omitempty"`
	BlastRadius          *BlastRadiusReport `json:"blast_radius,omitempty"`
	Advisory             string             `json:"advisory,omitempty"`
	PreFlightRadar       string             `json:"preflight_radar,omitempty"`
	LatencyMs            float64            `json:"latency_ms"`
}

// Engine tracks coding session tool trajectories and queries a local System 1
// decision server (/v1/systemone) + Bullet-Time Go/Python AST State-Machine Explorer.
type Engine struct {
	endpoint   string
	httpClient *http.Client

	mu                 sync.Mutex
	editCounts         map[string]int             // file_path -> consecutive edits without a passing test
	stateHistory       map[string][]StateSnapshot // file_path -> structural AST signature history
	knownRadars        map[string]string          // file_path -> most recent Pre-Flight Radar summary
	lastECCM           map[string]string          // file_path -> most recent ECCM auto-lock summary
	lastCycle          map[string]string          // file_path -> most recent Warp Scrambler cycle summary
	lastBlast          *BlastRadiusReport         // most recent Cross-File Blast-Radius report
	extraSearchRoots   []string                   // additional search roots for Cross-File Blast-Radius scanning
	lastVerdict        SenseVerdict               // most recent tool-turn verdict for Capacitor Router
	smolModel          string                     // fast/free workhorse model ID
	slowModel          string                     // heavy reasoning model ID for Bullet Time
	maxHistoryContents int                        // threshold for 500-token Handover Capsule compaction
	routerStats        CapacitorRouterStats       // telemetry for role shifts & capsule compaction
	lastTool           string
	repeatCount        int
}

// NewEngine creates a System 1 + Bullet-Time sense-making engine pointing at a
// local Laya /v1/systemone HTTP endpoint (e.g. "http://127.0.0.1:8765").
func NewEngine(endpoint string) *Engine {
	return &Engine{
		endpoint: strings.TrimRight(endpoint, "/"),
		httpClient: &http.Client{
			Timeout: 1500 * time.Millisecond,
		},
		editCounts:         make(map[string]int),
		stateHistory:       make(map[string][]StateSnapshot),
		knownRadars:        make(map[string]string),
		lastECCM:           make(map[string]string),
		lastCycle:          make(map[string]string),
		maxHistoryContents: 8,
	}
}

// DefaultDaemonEndpoint is the local address where laya_system1d.py listens.
const DefaultDaemonEndpoint = "http://127.0.0.1:8765"

// NewEngineFromEnv resolves the System 1 endpoint from PI_SYSTEM1_URL or
// LAYA_SYSTEM1_URL. When neither env var is set, it performs a fast 15ms TCP
// probe against 127.0.0.1:8765 so an always-on local laya_system1d.py daemon
// is discovered automatically with zero configuration.
func NewEngineFromEnv() *Engine {
	ep := strings.TrimSpace(os.Getenv("PI_SYSTEM1_URL"))
	if ep == "" {
		ep = strings.TrimSpace(os.Getenv("LAYA_SYSTEM1_URL"))
	}
	if ep == "" {
		if conn, err := net.DialTimeout("tcp", "127.0.0.1:8765", 15*time.Millisecond); err == nil {
			_ = conn.Close()
			ep = DefaultDaemonEndpoint
		}
	}
	return NewEngine(ep)
}

// BuildBeforeToolCallback returns an ADK BeforeToolCallback that fits two
// pre-flight S3 subsystems before a tool executes:
//  1. ECCM Fuzzy Edit Auto-Lock: snaps whitespace/indentation-drifted old_string
//     targets onto the unique matching code block in the file so weak/free models
//     do not waste turns on "old_string not found".
//  2. Pre-Edit Baseline Snapshot: records the file's pre-edit structural AST
//     signature so the Warp Scrambler can detect A->B->A semantic reverts even
//     if the file was not read first.
func (e *Engine) BuildBeforeToolCallback() llmagent.BeforeToolCallback {
	return func(_ agent.Context, t tool.Tool, args map[string]any) (map[string]any, error) {
		if t.Name() != "edit" && t.Name() != "write" {
			return nil, nil
		}
		filePath, _ := args["file_path"].(string)
		if filePath == "" {
			filePath, _ = args["path"].(string)
		}
		if filePath != "" {
			if raw, err := os.ReadFile(filePath); err == nil {
				e.recordStructuralState(filePath, string(raw), "baseline")
			}
		}
		if t.Name() == "edit" {
			if locked, summary := ECCMAutoLockEdit(args); locked {
				e.mu.Lock()
				e.lastECCM[filePath] = summary
				e.mu.Unlock()
				notice.Notifyf("[S3 ECCM · AUTO-LOCK] %s", summary)
			}
		}
		return nil, nil
	}
}

func (e *Engine) recordStructuralState(filePath, content, label string) (cycleDetected bool, cycleSummary string) {
	if filePath == "" || content == "" {
		return false, ""
	}
	sig := StructuralSignature(filePath, content)

	e.mu.Lock()
	defer e.mu.Unlock()

	hist := e.stateHistory[filePath]
	if len(hist) > 0 && hist[len(hist)-1].Sig == sig {
		// Unchanged from current state (e.g. baseline read before edit, or comment-only edit)
		return false, ""
	}

	// Check if sig matches any non-current prior state (A -> B -> ... -> A cycle!)
	for i := 0; i < len(hist)-1; i++ {
		if hist[i].Sig == sig {
			summary := fmt.Sprintf(
				"Warp Scrambler: Semantic A->B->A AST cycle detected on %s (edit reverts to exact structural signature %s from step #%d [%s]; comment/whitespace differences ignored)",
				filepath.Base(filePath), sig[:8], hist[i].Step, hist[i].Summary,
			)
			e.stateHistory[filePath] = append(hist, StateSnapshot{
				Step:    len(hist) + 1,
				Sig:     sig,
				Summary: label + " (CYCLE_REVERT)",
			})
			e.lastCycle[filePath] = summary
			return true, summary
		}
	}

	e.stateHistory[filePath] = append(hist, StateSnapshot{
		Step:    len(hist) + 1,
		Sig:     sig,
		Summary: label,
	})
	delete(e.lastCycle, filePath)
	return false, ""
}

// BuildAfterToolCallback returns an ADK AfterToolCallback compatible with
// extension.ComposeAfterToolChain. Place it after BuildCompactorCallback
// and before memory.BuildAfterToolCallback / memRecorder.afterTool.
func (e *Engine) BuildAfterToolCallback() llmagent.AfterToolCallback {
	return func(ctx agent.Context, t tool.Tool, args, result map[string]any, toolErr error) (map[string]any, error) {
		if result == nil {
			return nil, nil
		}
		verdict := e.EvaluateToolTurn(ctx, t.Name(), args, result, toolErr)
		result["system1_sense"] = verdict
		result["system1_memory_retain"] = verdict.RetainMemory
		if verdict.PreFlightRadar != "" {
			result["s3_preflight_radar"] = verdict.PreFlightRadar
		}
		if verdict.BlastRadius != nil && verdict.BlastRadius.Summary != "" {
			result["s3_blast_radius"] = verdict.BlastRadius.Summary
		}
		if verdict.BulletTimeEngaged && verdict.Advisory != "" {
			result["bullet_time_advisory"] = verdict.Advisory
			filePath, _ := result["path"].(string)
			if filePath == "" {
				filePath, _ = args["file_path"].(string)
			}
			notice.Notifyf(
				"[BULLET TIME · %s] %s (%d state anomaly/anomalies, %.1fms)",
				strings.ToUpper(verdict.Posture),
				filepath.Base(filePath),
				len(verdict.Anomalies),
				verdict.LatencyMs,
			)
		}
		return result, nil
	}
}

// EvaluateToolTurn runs the fast sensor + System 1 ANE decision pass, and triggers
// Bullet-Time AST State-Machine exploration whenever a desync or loop is detected.
func (e *Engine) EvaluateToolTurn(
	ctx context.Context,
	toolName string,
	args, result map[string]any,
	toolErr error,
) SenseVerdict {
	started := time.Now()

	filePath, _ := result["path"].(string)
	if filePath == "" {
		filePath, _ = args["file_path"].(string)
	}
	lspDiags, _ := result["lsp_diagnostics"].(string)
	hasLSPError := strings.Contains(lspDiags, "Error") || strings.Contains(lspDiags, "error")

	exitCode := 0
	if ec, ok := result["exit_code"].(int); ok {
		exitCode = ec
	} else if ecFloat, ok := result["exit_code"].(float64); ok {
		exitCode = int(ecFloat)
	}
	hasToolFailure := toolErr != nil || exitCode != 0

	// Update trajectory loop counters
	e.mu.Lock()
	if toolName == e.lastTool && (toolName == "edit" || toolName == "grep" || toolName == "read") {
		e.repeatCount++
	} else {
		e.repeatCount = 1
		e.lastTool = toolName
	}
	if toolName == "edit" || toolName == "write" {
		e.editCounts[filePath]++
	} else if toolName == "bash" && exitCode == 0 {
		// Passing test/build clears consecutive unverified edit counter
		e.editCounts = make(map[string]int)
	}
	editsOnFile := e.editCounts[filePath]
	repeats := e.repeatCount
	e.mu.Unlock()

	// Fast AST sensor pass on read/edit/write of Go or Python files (<1 ms)
	var astGraph *PackageStateGraph
	var anomalies []CouplingAnomaly
	var preFlightRadar string
	var eccmLocked bool
	var eccmSummary string
	var warpScrambler bool
	var cycleSummary string
	var blastReport *BlastRadiusReport

	if toolName == "read" || toolName == "edit" || toolName == "write" {
		if raw, rErr := os.ReadFile(filePath); rErr == nil {
			if toolName == "read" {
				e.recordStructuralState(filePath, string(raw), "read")
			} else if !hasToolFailure {
				warpScrambler, cycleSummary = e.recordStructuralState(filePath, string(raw), toolName)
			}
		}
	}

	if (toolName == "edit" || toolName == "write") && !hasToolFailure {
		if br := e.ScanBlastRadius(filePath, args); br != nil && (len(br.Consumers) > 0 || br.UnsoundnessDetected) {
			blastReport = br
			e.mu.Lock()
			e.lastBlast = br
			e.mu.Unlock()
		}
	}

	e.mu.Lock()
	if s, ok := e.lastECCM[filePath]; ok && toolName == "edit" {
		eccmLocked = true
		eccmSummary = s
		delete(e.lastECCM, filePath)
	}
	e.mu.Unlock()

	if toolName == "edit" || toolName == "write" || toolName == "read" {
		dir := filepath.Dir(filePath)
		switch {
		case strings.HasSuffix(filePath, ".go"):
			if g, err := AnalyzePackageDir(dir, filePath); err == nil {
				astGraph = g
				r := BuildPreFlightRadar(filePath, g)
				if r != "" {
					e.mu.Lock()
					e.knownRadars[filePath] = r
					e.mu.Unlock()
				}
				if toolName == "read" {
					preFlightRadar = r
				} else {
					anomalies = g.Anomalies
				}
			}
		case strings.HasSuffix(filePath, ".py"):
			if g, err := AnalyzePythonDir(dir, filePath); err == nil {
				astGraph = g
				r := BuildPreFlightRadar(filePath, g)
				if r != "" {
					e.mu.Lock()
					e.knownRadars[filePath] = r
					e.mu.Unlock()
				}
				if toolName == "read" {
					preFlightRadar = r
				} else {
					anomalies = g.Anomalies
				}
			}
		}
	}

	hasUnsoundness := blastReport != nil && blastReport.UnsoundnessDetected
	hasDesync := len(anomalies) > 0 || hasLSPError || hasToolFailure || warpScrambler || hasUnsoundness
	isLooping := editsOnFile >= 3 || repeats >= 4 || warpScrambler
	isDurable := (toolName == "edit" || toolName == "write") && !hasLSPError && len(anomalies) == 0 && !warpScrambler

	// Determine recommended tactical posture & entailment-aligned prompt (Discovery 6 + Unsoundness Flag)
	pref := PostureProceed
	if warpScrambler {
		pref = PostureBranch
	} else if len(anomalies) > 0 || hasLSPError || hasUnsoundness {
		pref = PostureInspect
	} else if isLooping {
		pref = PostureBranch
	} else if toolName == "edit" || toolName == "write" {
		pref = PostureVerify
	}

	stateStr := fmt.Sprintf(
		"State desync detected: %s. Static analysis incomplete: %s. Durable engineering change: %s.",
		yesNo(hasDesync),
		yesNo(hasUnsoundness),
		yesNo(isDurable),
	)

	criteria := map[string]string{
		PostureProceed: "Suboptimal action.",
		PostureVerify:  "Suboptimal action.",
		PostureInspect: "Suboptimal action.",
		PostureBranch:  "Suboptimal action.",
	}
	for k := range criteria {
		if k == pref {
			criteria[k] = "Safe. Best tactical coding sense action."
		} else if k == PostureProceed && hasDesync {
			criteria[k] = "Unsafe. Ignores active state desync or error."
		}
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
		"retain": map[string]any{
			"type":         "noul",
			"instructions": "Is this a durable engineering change?",
			"criteria": map[string]string{
				"true":  "durable code change or verified bugfix worth storing",
				"false": "ephemeral file read, directory listing, or broken edit",
			},
		},
	}

	probs, desyncProb, retainProb, srr := e.callSystem1(ctx, stateStr, questions, pref, hasDesync, isDurable)
	executed := maxKey(probs, pref)

	// Policy Shield: always enforce PostureBranch on Warp Scrambler AST cycles,
	// and PostureInspect when AST state-coupling anomalies or dynamic unsoundness exist.
	if warpScrambler {
		executed = PostureBranch
	} else if len(anomalies) > 0 || hasUnsoundness {
		executed = PostureInspect
	} else if isLooping && executed == PostureProceed {
		executed = PostureBranch
	}

	// Upgrade 5: Role-Stratified Conformal Risk Control (arXiv:2607.24343 & SHIFT-JEV)
	//   - Tier 0 (read/grep/ls/sense_map): risk ceiling tau = 0.20
	//   - Tier 1 (bounded local edit):     risk ceiling tau = 0.45
	//   - Tier 2 (cross-package/service edit, unsound dynamic dispatch, or bash): strict tau = 0.65
	riskTier := "TIER0_RECON"
	actThreshold := 0.20
	if toolName == "edit" || toolName == "write" || toolName == "bash" {
		riskTier = "TIER1_LOCAL_MUTATION"
		actThreshold = 0.45
		if hasUnsoundness || (blastReport != nil && (blastReport.Level == BlastLevelCrossPackage || blastReport.Level == BlastLevelCrossService)) {
			riskTier = "TIER2_HIGH_BLAST_OR_DYNAMIC"
			actThreshold = 0.65
		}
	}

	bulletTime := executed == PostureInspect || executed == PostureBranch || len(anomalies) > 0 || warpScrambler || hasUnsoundness || (hasDesync && desyncProb >= 0.55)
	advisory := ""
	if bulletTime {
		advisory = buildBulletTimeAdvisory(toolName, filePath, executed, anomalies, astGraph, lspDiags, editsOnFile, cycleSummary, blastReport)
	} else if blastReport != nil && blastReport.Summary != "" {
		advisory = blastReport.Summary
	}

	// Upgrade 1: Signed Rescue Routing (SRR) + Role-Stratified Risk Control
	recRole := CloneRoleSmol
	if bulletTime || warpScrambler || executed == PostureInspect || executed == PostureBranch || srr.SignedGain > 0.0 || srr.ActProb < actThreshold {
		recRole = CloneRoleSlow
	}

	verdict := SenseVerdict{
		Posture:              executed,
		Probabilities:        probs,
		DesyncProb:           desyncProb,
		ActProb:              srr.ActProb,
		RescueProb:           srr.RescueProb,
		HarmProb:             srr.HarmProb,
		AmbigProb:            srr.AmbigProb,
		SignedGain:           srr.SignedGain,
		RiskTier:             riskTier,
		RecommendedRole:      recRole,
		RetainMemory:         retainProb >= 0.50 && isDurable,
		RetainProb:           retainProb,
		BulletTimeEngaged:    bulletTime,
		ECCMAutoLocked:       eccmLocked,
		ECCMSummary:          eccmSummary,
		WarpScramblerEngaged: warpScrambler,
		CycleSummary:         cycleSummary,
		Anomalies:            anomalies,
		BlastRadius:          blastReport,
		Advisory:             advisory,
		PreFlightRadar:       preFlightRadar,
		LatencyMs:            float64(time.Since(started).Microseconds()) / 1000.0,
	}
	e.mu.Lock()
	e.lastVerdict = verdict
	e.mu.Unlock()
	return verdict
}

func buildBulletTimeAdvisory(
	toolName, filePath, posture string,
	anomalies []CouplingAnomaly,
	graph *PackageStateGraph,
	lspDiags string,
	editsOnFile int,
	cycleSummary string,
	blastReport *BlastRadiusReport,
) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("[BULLET TIME · SYSTEM 2 SENSE-MAKER (%s on %s -> %s)]\n", toolName, filepath.Base(filePath), strings.ToUpper(posture)))

	if cycleSummary != "" {
		b.WriteString("  • " + cycleSummary + "\n")
		b.WriteString("  • Directive: Stop oscillating between previously visited code states. Re-read constraints or branch session.\n")
	}
	if len(anomalies) > 0 {
		b.WriteString("Implicit State-Machine Hazards Detected by Go AST Sensor:\n")
		for _, a := range anomalies {
			b.WriteString("  • " + a.Summary + "\n")
		}
	}
	if blastReport != nil && blastReport.Summary != "" {
		b.WriteString(blastReport.Summary + "\n")
	}
	if lspDiags != "" {
		b.WriteString("LSP Diagnostics:\n  • " + strings.ReplaceAll(lspDiags, "\n", "\n  • ") + "\n")
	}
	if editsOnFile >= 3 {
		b.WriteString(fmt.Sprintf("Oscillation Warning: %d consecutive unverified edits on %s. Consider session branching or switching to 'plan' role.\n", editsOnFile, filepath.Base(filePath)))
	}
	if graph != nil && len(anomalies) > 0 {
		sName := anomalies[0].StructName
		b.WriteString(fmt.Sprintf("State Coupling Map for struct %s:\n", sName))
		for _, m := range graph.Methods {
			if m.StructName == sName && len(m.Writes) > 0 {
				lockTag := "unlocked"
				if m.UsesLock {
					lockTag = "mu.Lock"
				}
				b.WriteString(fmt.Sprintf("  - %s() [%s]: writes=%v reads=%v\n", m.MethodName, lockTag, m.Writes, m.Reads))
			}
		}
	}
	return strings.TrimSpace(b.String())
}

type srrActionMetrics struct {
	ActProb    float64
	RescueProb float64
	HarmProb   float64
	AmbigProb  float64
	SignedGain float64
}

func (e *Engine) callSystem1(
	ctx context.Context,
	state string,
	questions map[string]any,
	fallbackChoice string,
	fallbackDesync, fallbackRetain bool,
) (map[string]float64, float64, float64, srrActionMetrics) {
	defaultProbs := map[string]float64{
		PostureProceed: 0.05,
		PostureVerify:  0.05,
		PostureInspect: 0.05,
		PostureBranch:  0.05,
	}
	defaultProbs[fallbackChoice] = 0.85
	dProb := 0.05
	srr := srrActionMetrics{ActProb: 0.88, RescueProb: 0.04, HarmProb: 0.25, AmbigProb: 0.02, SignedGain: -0.33}
	if fallbackDesync {
		dProb = 0.95
		srr = srrActionMetrics{ActProb: 0.12, RescueProb: 0.91, HarmProb: 0.03, AmbigProb: 0.04, SignedGain: 0.76}
	}
	rProb := 0.05
	if fallbackRetain {
		rProb = 0.95
	}

	if e.endpoint == "" {
		return defaultProbs, dProb, rProb, srr
	}

	payload, err := json.Marshal(map[string]any{
		"state":     state,
		"questions": questions,
	})
	if err != nil {
		return defaultProbs, dProb, rProb, srr
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint+"/v1/systemone", bytes.NewReader(payload))
	if err != nil {
		return defaultProbs, dProb, rProb, srr
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := e.httpClient.Do(req)
	if err != nil {
		return defaultProbs, dProb, rProb, srr
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return defaultProbs, dProb, rProb, srr
	}

	var body struct {
		Answers map[string]struct {
			Choice        string             `json:"choice"`
			Probabilities map[string]float64 `json:"probabilities"`
			Noul          float64            `json:"noul"`
			Action        struct {
				ActProbability float64 `json:"act_probability"`
				RescueProb     float64 `json:"rescue_prob"`
				HarmProb       float64 `json:"harm_prob"`
				AmbigProb      float64 `json:"ambig_prob"`
				SignedGain     float64 `json:"signed_gain"`
			} `json:"action"`
		} `json:"answers"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return defaultProbs, dProb, rProb, srr
	}

	if pAns, ok := body.Answers["posture"]; ok && len(pAns.Probabilities) > 0 {
		defaultProbs = pAns.Probabilities
		if pAns.Action.ActProbability > 0 {
			srr.ActProb = pAns.Action.ActProbability
			srr.RescueProb = pAns.Action.RescueProb
			srr.HarmProb = pAns.Action.HarmProb
			srr.AmbigProb = pAns.Action.AmbigProb
			srr.SignedGain = pAns.Action.SignedGain
		}
	}
	if dAns, ok := body.Answers["desync"]; ok {
		dProb = dAns.Noul
	}
	if rAns, ok := body.Answers["retain"]; ok {
		rProb = rAns.Noul
	}
	return defaultProbs, dProb, rProb, srr
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func maxKey(m map[string]float64, fallback string) string {
	bestK := fallback
	bestV := -1.0
	for k, v := range m {
		if v > bestV {
			bestV = v
			bestK = k
		}
	}
	return bestK
}
