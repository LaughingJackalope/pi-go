package sense

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"google.golang.org/adk/v2/agent"
	"google.golang.org/adk/v2/agent/llmagent"
	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"

	"github.com/dimetron/pi-go/internal/notice"
)

// CloneRole represents the target clone tier selected by the S3 Capacitor Router.
const (
	CloneRoleSmol = "smol" // Fast/free high-RPM workhorse clone for routine PROCEED/VERIFY turns
	CloneRoleSlow = "slow" // High-reasoning heavy clone for BULLET TIME (INSPECT/BRANCH) turns
)

// CapacitorRouterStats tracks dynamic role shifts and transcript compaction savings.
type CapacitorRouterStats struct {
	TotalSteps          int    `json:"total_steps"`
	SmolSteps           int    `json:"smol_steps"`
	SlowSteps           int    `json:"slow_steps"`
	RoleShifts          int    `json:"role_shifts"`
	CapsulesInjected    int    `json:"capsules_injected"`
	ContentsPruned      int    `json:"contents_pruned"`
	ActiveRole          string `json:"active_role"`
	ActiveModel         string `json:"active_model"`
	LastHandoverCapsule string `json:"last_handover_capsule,omitempty"`
}

// ConfigureCapacitorRouter sets the workhorse (smol) and heavy (slow) model IDs
// for dynamic per-step role shifting, plus the transcript length threshold at
// which stale middle tool-churn turns are replaced by the 500-token S3 Clone
// Handover Capsule.
//
// Environment variables PI_S3_SMOL_MODEL and PI_S3_SLOW_MODEL override the
// passed model names when set.
func (e *Engine) ConfigureCapacitorRouter(smolModel, slowModel string, maxHistoryContents int) {
	if envSmol := strings.TrimSpace(os.Getenv("PI_S3_SMOL_MODEL")); envSmol != "" {
		smolModel = envSmol
	}
	if envSlow := strings.TrimSpace(os.Getenv("PI_S3_SLOW_MODEL")); envSlow != "" {
		slowModel = envSlow
	}
	if maxHistoryContents <= 0 {
		maxHistoryContents = 8
	}
	e.mu.Lock()
	e.smolModel = smolModel
	e.slowModel = slowModel
	e.maxHistoryContents = maxHistoryContents
	e.mu.Unlock()
}

// RouterStats returns a snapshot of the Capacitor Router telemetry.
func (e *Engine) RouterStats() CapacitorRouterStats {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.routerStats
}

// SynthesizeHandoverCapsule builds a deterministic ~300-500 token S3 Clone
// Handover Capsule summarizing the session's active tactical posture, touched
// files and their lexical AST hashes, discovered struct/class state-coupling
// invariants, and unresolved Bullet-Time / Warp-Scrambler advisories.
//
// Used both when hot-swapping between clones mid-session (e.g. smol <-> slow or
// after a 429 rate-limit) and when compacting stale middle tool turns out of
// req.Contents.
func (e *Engine) SynthesizeHandoverCapsule() string {
	e.mu.Lock()
	defer e.mu.Unlock()

	var b strings.Builder
	posture := e.lastVerdict.Posture
	if posture == "" {
		posture = PostureProceed
	}
	role := e.lastVerdict.RecommendedRole
	if role == "" {
		role = CloneRoleSmol
	}

	b.WriteString(fmt.Sprintf(
		"[S3 CLONE HANDOVER CAPSULE · POSTURE=%s · TARGET_CLONE=%s · ACT_PROB=%.2f · DESYNC_PROB=%.2f]\n",
		strings.ToUpper(posture), strings.ToUpper(role), e.lastVerdict.ActProb, e.lastVerdict.DesyncProb,
	))

	// 1. Touched files & structural AST signatures
	if len(e.stateHistory) > 0 {
		files := make([]string, 0, len(e.stateHistory))
		for f := range e.stateHistory {
			files = append(files, f)
		}
		sort.Strings(files)
		b.WriteString("Tracked Files & Lexical AST Signatures:\n")
		for _, f := range files {
			hist := e.stateHistory[f]
			if len(hist) == 0 {
				continue
			}
			latest := hist[len(hist)-1]
			edits := e.editCounts[f]
			b.WriteString(fmt.Sprintf(
				"  • %s: ast_sig=%s (steps=%d, unverified_edits=%d, last=%s)\n",
				filepath.Base(f), latest.Sig[:8], len(hist), edits, latest.Summary,
			))
		}
	}

	// 2. Discovered Pre-Flight State-Coupling Invariants across read/edited packages
	if len(e.knownRadars) > 0 {
		rFiles := make([]string, 0, len(e.knownRadars))
		for f := range e.knownRadars {
			rFiles = append(rFiles, f)
		}
		sort.Strings(rFiles)
		b.WriteString("Active Package/Module State-Coupling Invariants:\n")
		for _, f := range rFiles {
			for _, line := range strings.Split(e.knownRadars[f], "\n") {
				if strings.HasPrefix(line, "[S3 PRE-FLIGHT RADAR") {
					continue
				}
				if strings.TrimSpace(line) != "" {
					b.WriteString("  " + line + "\n")
				}
			}
		}
	}

	// 3. Active Cross-Package / Cross-Service Blast Radius
	if e.lastBlast != nil && e.lastBlast.Summary != "" {
		b.WriteString("Active Cross-File Blast-Radius Map:\n")
		for _, line := range strings.Split(e.lastBlast.Summary, "\n") {
			b.WriteString("  " + line + "\n")
		}
	}

	// 4. Active Bullet-Time / Warp Scrambler Advisories
	if e.lastVerdict.BulletTimeEngaged && e.lastVerdict.Advisory != "" {
		b.WriteString("Active Bullet-Time Directive:\n")
		for _, line := range strings.Split(e.lastVerdict.Advisory, "\n") {
			b.WriteString("  " + line + "\n")
		}
	}

	return strings.TrimSpace(b.String())
}

// BuildBeforeModelCallback returns an ADK BeforeModelCallback that implements:
//  1. The S3 Capacitor Router: dynamically shifts req.Model between the fast/free
//     workhorse clone ("smol") on routine PROCEED/VERIFY steps and the heavy
//     reasoning clone ("slow") when Bullet Time or the Warp Scrambler engages.
//  2. The 500-Token Clone Hot-Swap Handover Capsule: when req.Contents grows
//     beyond maxHistoryContents (or on a clone role shift), prunes stale middle
//     tool-churn turns while preserving valid user/model turn pairing and
//     injects the deterministic S3 Handover Capsule.
func (e *Engine) BuildBeforeModelCallback() llmagent.BeforeModelCallback {
	return func(_ agent.Context, req *model.LLMRequest) (*model.LLMResponse, error) {
		if req == nil {
			return nil, nil
		}

		capsule := e.SynthesizeHandoverCapsule()

		e.mu.Lock()
		verdict := e.lastVerdict
		targetRole := verdict.RecommendedRole
		if targetRole == "" {
			targetRole = CloneRoleSmol
		}
		smolModel := e.smolModel
		slowModel := e.slowModel
		maxContents := e.maxHistoryContents
		if maxContents <= 0 {
			maxContents = 8
		}

		e.routerStats.TotalSteps++
		prevRole := e.routerStats.ActiveRole
		shifted := false

		if targetRole == CloneRoleSlow {
			e.routerStats.SlowSteps++
			if slowModel != "" && req.Model != slowModel {
				req.Model = slowModel
				shifted = prevRole != "" && prevRole != CloneRoleSlow
			} else if prevRole != "" && prevRole != CloneRoleSlow {
				shifted = true
			}
		} else {
			e.routerStats.SmolSteps++
			if smolModel != "" && req.Model != smolModel {
				req.Model = smolModel
				shifted = prevRole != "" && prevRole != CloneRoleSmol
			} else if prevRole != "" && prevRole != CloneRoleSmol {
				shifted = true
			}
		}

		if shifted {
			e.routerStats.RoleShifts++
		}
		e.routerStats.ActiveRole = targetRole
		e.routerStats.ActiveModel = req.Model
		e.routerStats.LastHandoverCapsule = capsule
		e.mu.Unlock()

		if shifted {
			notice.Notifyf(
				"[S3 CAPACITOR ROUTER · %s] Shifted clone role -> %s (model=%s, posture=%s, act_prob=%.2f)",
				strings.ToUpper(targetRole),
				targetRole,
				req.Model,
				verdict.Posture,
				verdict.ActProb,
			)
		}

		// Inject the 500-token Handover Capsule and prune middle tool-churn turns
		// when the transcript exceeds maxContents or when hot-swapping clones.
		if len(req.Contents) > maxContents || (shifted && len(req.Contents) > 4) {
			pruned := compactContentsWithCapsule(req, capsule, maxContents)
			if pruned > 0 {
				e.mu.Lock()
				e.routerStats.CapsulesInjected++
				e.routerStats.ContentsPruned += pruned
				e.mu.Unlock()
			}
		} else if verdict.BulletTimeEngaged && capsule != "" {
			// Even on short transcripts, attach the capsule to SystemInstruction
			// during Bullet Time so the escalated clone sees full package invariants.
			appendCapsuleToSystemInstruction(req, capsule)
			e.mu.Lock()
			e.routerStats.CapsulesInjected++
			e.mu.Unlock()
		}

		return nil, nil
	}
}

// compactContentsWithCapsule prunes stale intermediate (model FunctionCall +
// user FunctionResponse) turn pairs from the middle of req.Contents, keeping
// req.Contents[0] (the initial user prompt) and the most recent tailContents
// turns, and attaches the 500-token S3 Handover Capsule to SystemInstruction.
func compactContentsWithCapsule(req *model.LLMRequest, capsule string, maxContents int) int {
	n := len(req.Contents)
	if n <= maxContents || n <= 3 {
		return 0
	}

	// Keep req.Contents[0] at the head, and find a safe tail boundary of length
	// at most (maxContents - 1) that starts on a "model" turn (or a non-FunctionResponse
	// "user" turn) so we never leave an orphaned FunctionResponse after Contents[0].
	tailStart := n - (maxContents - 1)
	if tailStart < 1 {
		tailStart = 1
	}
	for tailStart < n-1 && hasFunctionResponse(req.Contents[tailStart]) {
		tailStart++
	}
	if tailStart <= 1 {
		return 0
	}

	pruned := tailStart - 1
	compacted := make([]*genai.Content, 0, 1+(n-tailStart))
	compacted = append(compacted, req.Contents[0])
	compacted = append(compacted, req.Contents[tailStart:]...)
	req.Contents = compacted

	appendCapsuleToSystemInstruction(req, capsule)
	return pruned
}

func hasFunctionResponse(c *genai.Content) bool {
	if c == nil {
		return false
	}
	for _, p := range c.Parts {
		if p != nil && p.FunctionResponse != nil {
			return true
		}
	}
	return false
}

func appendCapsuleToSystemInstruction(req *model.LLMRequest, capsule string) {
	if req == nil || capsule == "" {
		return
	}
	if req.Config == nil {
		req.Config = &genai.GenerateContentConfig{}
	}
	if req.Config.SystemInstruction == nil {
		req.Config.SystemInstruction = &genai.Content{
			Role:  "system",
			Parts: []*genai.Part{{Text: capsule}},
		}
		return
	}
	// Avoid duplicating the capsule if already present on this step's config
	for _, p := range req.Config.SystemInstruction.Parts {
		if p != nil && strings.Contains(p.Text, "[S3 CLONE HANDOVER CAPSULE") {
			p.Text = capsule
			return
		}
	}
	req.Config.SystemInstruction.Parts = append(
		req.Config.SystemInstruction.Parts,
		&genai.Part{Text: "\n\n" + capsule},
	)
}
