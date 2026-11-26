package sense

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

func TestAnalyzeGuardrailImplicitStateMachine(t *testing.T) {
	// Point the AST sensor at pi-go's real internal/guardrail package
	guardrailDir := filepath.Join("..", "guardrail")
	graph, err := AnalyzePackageDir(guardrailDir, "guardrail.go")
	if err != nil {
		t.Fatalf("AnalyzePackageDir failed: %v", err)
	}
	if len(graph.StructFields["Tracker"]) == 0 {
		t.Fatalf("expected Tracker struct fields to be extracted")
	}
	// Clean upstream guardrail.go should have 0 anomalies
	if len(graph.Anomalies) != 0 {
		t.Fatalf("expected 0 anomalies on clean upstream guardrail.go, got: %+v", graph.Anomalies)
	}
}

func TestBulletTimeCatchesImplicitStateDesyncOnEdit(t *testing.T) {
	// Copy guardrail.go into a temp dir and simulate an agent `edit` tool call that adds
	// a buggy method mutating `lastPromptTokens` WITHOUT acquiring `t.mu.Lock()` and
	// WITHOUT resetting coupled field `cachePrefixTokens` or `lastCachedTokens`!
	origBytes, err := os.ReadFile(filepath.Join("..", "guardrail", "guardrail.go"))
	if err != nil {
		t.Fatalf("read guardrail.go: %v", err)
	}

	tmpDir := t.TempDir()
	buggyFile := filepath.Join(tmpDir, "guardrail.go")
	buggyCode := string(origBytes) + `

// ForcePromptBaseline simulates a buggy agent edit that mutates lastPromptTokens
// without locking t.mu and without updating coupled fields cachePrefixTokens / lastCachedTokens.
func (t *Tracker) ForcePromptBaseline(tokens int64) {
	t.lastPromptTokens = tokens
}
`
	if err := os.WriteFile(buggyFile, []byte(buggyCode), 0o644); err != nil {
		t.Fatalf("write buggy guardrail.go: %v", err)
	}

	engine := NewEngineFromEnv()

	// 1. Simulate a routine `read` tool call -> should PostureProceed and NOT retain in Memory Palace
	readVerdict := engine.EvaluateToolTurn(
		context.Background(),
		"read",
		map[string]any{"file_path": buggyFile},
		map[string]any{"content": "package guardrail..."},
		nil,
	)
	if readVerdict.RetainMemory {
		t.Errorf("expected routine read tool call to be filtered out of Memory Palace (RetainMemory=false)")
	}
	if readVerdict.BulletTimeEngaged {
		t.Errorf("expected routine read not to trigger Bullet Time")
	}
	if !strings.Contains(readVerdict.PreFlightRadar, "Tracker.mu") || !strings.Contains(readVerdict.PreFlightRadar, "cachePrefixTokens") {
		t.Fatalf("expected S3 Pre-Flight Radar on read to report Tracker.mu and cachePrefixTokens invariant, got:\n%s", readVerdict.PreFlightRadar)
	}
	t.Logf("\n--- S3 PRE-FLIGHT RADAR INJECTED ON READ ---\n%s\n--------------------------------------------", readVerdict.PreFlightRadar)

	// 2. Simulate the buggy `edit` tool call -> must trigger PostureInspect + Bullet Time AST State Map!
	editVerdict := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{"file_path": buggyFile},
		map[string]any{"path": buggyFile, "replacements": 1},
		nil,
	)
	if !editVerdict.BulletTimeEngaged {
		t.Fatalf("expected Bullet Time to engage on buggy edit, got verdict: %+v", editVerdict)
	}
	if editVerdict.Posture != PostureInspect {
		t.Fatalf("expected posture=%q, got %q", PostureInspect, editVerdict.Posture)
	}
	if len(editVerdict.Anomalies) < 2 {
		t.Fatalf("expected both UNLOCKED_STATE_WRITE and MISSING_COUPLED_WRITE anomalies, got: %+v", editVerdict.Anomalies)
	}
	if !strings.Contains(editVerdict.Advisory, "cachePrefixTokens") || !strings.Contains(editVerdict.Advisory, "without acquiring Tracker.mu") {
		t.Fatalf("expected Bullet Time advisory to name coupled field cachePrefixTokens and missing mutex, got:\n%s", editVerdict.Advisory)
	}
	t.Logf("\n--- BULLET TIME ADVISORY INJECTED INTO PI-GO TOOL RESULT ---\n%s\n------------------------------------------------------------", editVerdict.Advisory)
}

func TestBulletTimeCatchesPythonImplicitStateDesync(t *testing.T) {
	// Simulate portal's apps/taskbot/files/taskbot.py where Bot.tasks and Bot.watching
	// are coupled state dicts protected by `with self.lock:`, and an agent edit adds
	// a buggy `force_purge_task` method that mutates `self.tasks` without `with self.lock:`
	// and without updating `self.watching`.
	pyCode := `import threading

class Bot:
    def __init__(self):
        self.lock = threading.Lock()
        self.tasks = {}
        self.watching = {}

    def spawn_task(self, tid, spec):
        with self.lock:
            self.tasks[tid] = spec
            self.watching[tid] = True

    def finish_task(self, tid):
        with self.lock:
            self.tasks.pop(tid, None)
            self.watching.pop(tid, None)
            _ = self.tasks.get(tid) or self.watching.get(tid)

    def inspect_status(self, tid):
        return self.tasks.get(tid), self.watching.get(tid)

    def force_purge_task(self, tid):
        # Buggy agent edit: mutates self.tasks without self.lock and leaves self.watching stale!
        self.tasks[tid] = None
`
	tmpDir := t.TempDir()
	pyFile := filepath.Join(tmpDir, "taskbot.py")
	if err := os.WriteFile(pyFile, []byte(pyCode), 0o644); err != nil {
		t.Fatalf("write taskbot.py: %v", err)
	}

	engine := NewEngineFromEnv()
	verdict := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{"file_path": pyFile},
		map[string]any{"path": pyFile, "replacements": 1},
		nil,
	)
	if !verdict.BulletTimeEngaged {
		t.Fatalf("expected Bullet Time to engage on buggy Python edit, got: %+v", verdict)
	}
	if verdict.Posture != PostureInspect {
		t.Fatalf("expected posture=%q, got %q", PostureInspect, verdict.Posture)
	}
	if len(verdict.Anomalies) < 2 {
		t.Fatalf("expected both UNLOCKED_STATE_WRITE and MISSING_COUPLED_WRITE on Python class Bot, got: %+v", verdict.Anomalies)
	}
	if !strings.Contains(verdict.Advisory, "watching") || !strings.Contains(verdict.Advisory, "Bot.lock") {
		t.Fatalf("expected Python Bullet Time advisory to name coupled field 'watching' and missing 'Bot.lock', got:\n%s", verdict.Advisory)
	}
	t.Logf("\n--- PYTHON BULLET TIME ADVISORY (taskbot.py) ---\n%s\n------------------------------------------------", verdict.Advisory)
}

func TestS3HullCombatTrial_RealPortalAndPiGoMissions(t *testing.T) {
	engine := NewEngineFromEnv()

	// Mission 1: Real ~/portal/apps/taskbot/files/taskbot.py
	// Alpha Clone adds `cancel_task(self, task_id)` mutating `self.tasks[task_id]`
	// without `self.watching.discard(task_id)` (co-mutated in task, resume).
	portalTaskbotPath := "/Users/mp/portal/apps/taskbot/files/taskbot.py"
	if orig, err := os.ReadFile(portalTaskbotPath); err == nil {
		tmpDir := t.TempDir()
		fPath := filepath.Join(tmpDir, "taskbot.py")
		// Alpha clone's myopic edit inside class Bot: mutates self.tasks without touching self.watching
		alphaEdit := strings.Replace(
			string(orig),
			"if __name__ == \"__main__\":",
			"    def cancel_task(self, task_id):\n        self.tasks[task_id] = {\"requester\": \"\", \"state\": \"closed\"}\n\n\nif __name__ == \"__main__\":",
			1,
		)
		if err := os.WriteFile(fPath, []byte(alphaEdit), 0o644); err != nil {
			t.Fatalf("write taskbot.py: %v", err)
		}
		v1 := engine.EvaluateToolTurn(
			context.Background(),
			"edit",
			map[string]any{"file_path": fPath},
			map[string]any{"path": fPath, "replacements": 1},
			nil,
		)
		if !v1.BulletTimeEngaged || v1.Posture != PostureInspect {
			t.Fatalf("Mission 1 (portal taskbot.py): expected Bullet Time INSPECT on Alpha Clone edit, got: %+v", v1)
		}
		if !strings.Contains(v1.Advisory, "Bot.cancel_task") || !strings.Contains(v1.Advisory, "watching") {
			t.Fatalf("Mission 1 (portal taskbot.py): expected S3 Target Painter to highlight Bot.cancel_task and coupled field 'watching', got:\n%s", v1.Advisory)
		}
		t.Logf("\n[MISSION 1 · CAPITAL SHIP PORTAL taskbot.py]\n%s", v1.Advisory)

		// Now apply the S3-guided repair (updating both self.tasks and self.watching in cancel_task)
		repairedEdit := strings.Replace(
			string(orig),
			"if __name__ == \"__main__\":",
			"    def cancel_task(self, task_id):\n        self.tasks[task_id] = {\"requester\": \"\", \"state\": \"closed\"}\n        self.watching.discard(task_id)\n\n\nif __name__ == \"__main__\":",
			1,
		)
		_ = os.WriteFile(fPath, []byte(repairedEdit), 0o644)
		v1Fixed := engine.EvaluateToolTurn(
			context.Background(),
			"edit",
			map[string]any{"file_path": fPath},
			map[string]any{"path": fPath, "replacements": 1},
			nil,
		)
		for _, a := range v1Fixed.Anomalies {
			if a.MethodName == "cancel_task" {
				t.Fatalf("Mission 1 repaired: expected cancel_task anomaly to be resolved, still got: %+v", a)
			}
		}
	}

	// Mission 2: Real ~/portal/apps/model-gateway/files/gateway.py
	// Alpha Clone adds `hot_reload_data(self, raw_dict)` to `MemoryConfigMap`
	// mutating `self.data` WITHOUT `with self.lock:` and WITHOUT bumping coupled `self.version`!
	portalGatewayPath := "/Users/mp/portal/apps/model-gateway/files/gateway.py"
	if orig, err := os.ReadFile(portalGatewayPath); err == nil {
		tmpDir := t.TempDir()
		fPath := filepath.Join(tmpDir, "gateway.py")
		// Add a second locked method on MemoryConfigMap so locked/coupled threshold (>=2) is active,
		// plus the Alpha Clone's buggy unlocked/uncoupled `hot_reload_data` method.
		alphaEdit := strings.Replace(
			string(orig),
			"            self.data, self.version = dict(data), self.version + 1\n            return True\n",
			"            self.data, self.version = dict(data), self.version + 1\n            return True\n\n    def reset_store(self):\n        with self.lock:\n            self.data, self.version = {}, self.version + 1\n\n    def hot_reload_data(self, raw_dict):\n        self.data = dict(raw_dict)\n",
			1,
		)
		if err := os.WriteFile(fPath, []byte(alphaEdit), 0o644); err != nil {
			t.Fatalf("write gateway.py: %v", err)
		}
		v2 := engine.EvaluateToolTurn(
			context.Background(),
			"edit",
			map[string]any{"file_path": fPath},
			map[string]any{"path": fPath, "replacements": 1},
			nil,
		)
		if !v2.BulletTimeEngaged || v2.Posture != PostureInspect {
			t.Fatalf("Mission 2 (portal gateway.py): expected Bullet Time INSPECT, got: %+v", v2)
		}
		if !strings.Contains(v2.Advisory, "MemoryConfigMap.lock") || !strings.Contains(v2.Advisory, "version") {
			t.Fatalf("Mission 2 (portal gateway.py): expected advisory to name MemoryConfigMap.lock and coupled field 'version', got:\n%s", v2.Advisory)
		}
		t.Logf("\n[MISSION 2 · CAPITAL SHIP PORTAL gateway.py]\n%s", v2.Advisory)
	}

	// Mission 3: Real pi-go internal/lsp/manager.go
	// Alpha Clone adds `EvictServer(key string)` to `Manager` mutating `m.servers`
	// WITHOUT acquiring `m.mu.Lock()`.
	lspMgrPath := filepath.Join("..", "lsp", "manager.go")
	if orig, err := os.ReadFile(lspMgrPath); err == nil {
		tmpDir := t.TempDir()
		fPath := filepath.Join(tmpDir, "manager.go")
		alphaEdit := string(orig) + `
func (m *Manager) EvictServer(key string) {
	delete(m.servers, key)
}
`
		if err := os.WriteFile(fPath, []byte(alphaEdit), 0o644); err != nil {
			t.Fatalf("write manager.go: %v", err)
		}
		v3 := engine.EvaluateToolTurn(
			context.Background(),
			"edit",
			map[string]any{"file_path": fPath},
			map[string]any{"path": fPath, "replacements": 1},
			nil,
		)
		if !v3.BulletTimeEngaged || v3.Posture != PostureInspect {
			t.Fatalf("Mission 3 (pi-go internal/lsp/manager.go): expected Bullet Time INSPECT, got: %+v", v3)
		}
		if !strings.Contains(v3.Advisory, "Manager.mu") {
			t.Fatalf("Mission 3 (pi-go internal/lsp/manager.go): expected advisory to name Manager.mu, got:\n%s", v3.Advisory)
		}
		t.Logf("\n[MISSION 3 · FRIGATE PI-GO internal/lsp/manager.go]\n%s", v3.Advisory)
	}
}

func TestSenseMapPreFlightRadarTool(t *testing.T) {
	senseTool, err := NewSenseMapTool()
	if err != nil {
		t.Fatalf("NewSenseMapTool failed: %v", err)
	}
	if senseTool.Name() != "sense_map" {
		t.Fatalf("expected tool name 'sense_map', got %q", senseTool.Name())
	}

	// Verify BuildPreFlightRadar on pi-go's internal/guardrail and portal's taskbot.py
	gGo, err := AnalyzePackageDir(filepath.Join("..", "guardrail"), "guardrail.go")
	if err != nil {
		t.Fatalf("AnalyzePackageDir failed: %v", err)
	}
	radarGo := BuildPreFlightRadar("guardrail.go", gGo)
	if !strings.Contains(radarGo, "Tracker.mu") || !strings.Contains(radarGo, "cachePrefixTokens <-> Tracker.lastPromptTokens") {
		t.Fatalf("unexpected Go Pre-Flight Radar:\n%s", radarGo)
	}

	if _, err := os.Stat("/Users/mp/portal/apps/taskbot/files/taskbot.py"); err == nil {
		gPy, err := AnalyzePythonDir("/Users/mp/portal/apps/taskbot/files", "taskbot.py")
		if err != nil {
			t.Fatalf("AnalyzePythonDir failed: %v", err)
		}
		radarPy := BuildPreFlightRadar("taskbot.py", gPy)
		if !strings.Contains(radarPy, "Bot.tasks <-> Bot.watching") {
			t.Fatalf("expected Python Pre-Flight Radar to report Bot.tasks <-> Bot.watching, got:\n%s", radarPy)
		}
		t.Logf("\n--- S3 PRE-FLIGHT RADAR (portal taskbot.py) ---\n%s\n-----------------------------------------------", radarPy)
	}
}

func TestECCMFuzzyEditAutoLockAndWarpScramblerCycleBreaker(t *testing.T) {
	tmpDir := t.TempDir()
	goFile := filepath.Join(tmpDir, "calc.go")

	// Initial State A (uses tabs for Go indentation)
	stateA := `package calc

func ComputeQuota(used, limit int) int {
	if used >= limit {
		return 0
	}
	return limit - used
}
`
	if err := os.WriteFile(goFile, []byte(stateA), 0o644); err != nil {
		t.Fatalf("write calc.go: %v", err)
	}

	engine := NewEngineFromEnv()
	beforeCB := engine.BuildBeforeToolCallback()

	// 1. Test ECCM Fuzzy Edit Auto-Lock:
	// Simulate an Alpha Clone sending 4-space indentation instead of tabs in old_string!
	driftedOld := "    if used >= limit {\n        return 0\n    }"
	driftedNew := "    if used > limit {\n        return -1\n    }"
	editArgs := map[string]any{
		"file_path":  goFile,
		"old_string": driftedOld,
		"new_string": driftedNew,
	}

	if _, err := beforeCB(nil, nilToolStub("edit"), editArgs); err != nil {
		t.Fatalf("beforeCB failed: %v", err)
	}

	lockedOld, _ := editArgs["old_string"].(string)
	if !strings.Contains(lockedOld, "\tif used >= limit {") {
		t.Fatalf("expected ECCM Auto-Lock to snap 4-space old_string onto tab-indented file target, got: %q", lockedOld)
	}
	// Apply the auto-locked edit to transition to State B
	stateB := strings.Replace(stateA, lockedOld, editArgs["new_string"].(string), 1)
	if err := os.WriteFile(goFile, []byte(stateB), 0o644); err != nil {
		t.Fatalf("write stateB: %v", err)
	}
	vB := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		editArgs,
		map[string]any{"path": goFile, "replacements": 1},
		nil,
	)
	if !vB.ECCMAutoLocked {
		t.Fatalf("expected ECCMAutoLocked=true on verdict B, got: %+v", vB)
	}
	if vB.WarpScramblerEngaged {
		t.Fatalf("expected WarpScramblerEngaged=false on first transition A->B")
	}
	t.Logf("\n--- ECCM AUTO-LOCK SUMMARY ---\n%s\n------------------------------", vB.ECCMSummary)

	// 2. Test Warp Scrambler (Semantic A -> B -> A Cycle Detector):
	// Now the Alpha Clone oscillates back to State A's logic, even adding a new comment
	// and extra blank lines so a naive byte-hash or string compare would think it's new!
	stateARevertedWithComments := `package calc

// ComputeQuota calculates remaining quota (oscillated revert with extra comment!)
func ComputeQuota(used, limit int) int {

	if used >= limit {
		return 0
	}

	return limit - used
}
`
	if err := os.WriteFile(goFile, []byte(stateARevertedWithComments), 0o644); err != nil {
		t.Fatalf("write stateA revert: %v", err)
	}
	vRevert := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{"file_path": goFile},
		map[string]any{"path": goFile, "replacements": 1},
		nil,
	)
	if !vRevert.WarpScramblerEngaged {
		t.Fatalf("expected Warp Scrambler to catch semantic A->B->A AST cycle despite added comments/blank lines, got: %+v", vRevert)
	}
	if vRevert.Posture != PostureBranch || !vRevert.BulletTimeEngaged {
		t.Fatalf("expected PostureBranch and BulletTimeEngaged=true on A->B->A cycle, got posture=%s", vRevert.Posture)
	}
	t.Logf("\n--- WARP SCRAMBLER BULLET TIME ADVISORY ---\n%s\n-------------------------------------------", vRevert.Advisory)
}

type stubNamedTool string

func nilToolStub(name string) stubNamedTool        { return stubNamedTool(name) }
func (s stubNamedTool) Name() string               { return string(s) }
func (s stubNamedTool) Description() string        { return "" }
func (s stubNamedTool) IsLongRunning() bool        { return false }

func TestCapacitorRouterAndCloneHandoverCapsule(t *testing.T) {
	// Copy guardrail.go to a temp dir so we can simulate a multi-turn session
	// that starts on a routine `read` (smol clone), hits a Bullet-Time state
	// desync on `edit` (escalates to slow clone + injects 500-token Handover Capsule),
	// and repairs the desync (drops back to smol clone).
	origBytes, err := os.ReadFile(filepath.Join("..", "guardrail", "guardrail.go"))
	if err != nil {
		t.Fatalf("read guardrail.go: %v", err)
	}
	tmpDir := t.TempDir()
	targetFile := filepath.Join(tmpDir, "guardrail.go")
	if err := os.WriteFile(targetFile, origBytes, 0o644); err != nil {
		t.Fatalf("write guardrail.go: %v", err)
	}

	engine := NewEngineFromEnv()
	const smolClone = "qwen/qwen3-coder:free"
	const slowClone = "deepseek/deepseek-r1:free"
	engine.ConfigureCapacitorRouter(smolClone, slowClone, 6)
	beforeModelCB := engine.BuildBeforeModelCallback()

	// Step 1: Routine `read` -> ANE act_head recommends smol workhorse clone
	vRead := engine.EvaluateToolTurn(
		context.Background(),
		"read",
		map[string]any{"file_path": targetFile},
		map[string]any{"content": "package guardrail..."},
		nil,
	)
	if vRead.RecommendedRole != CloneRoleSmol {
		t.Fatalf("expected RecommendedRole=%q on routine read, got %q (act_prob=%.3f)", CloneRoleSmol, vRead.RecommendedRole, vRead.ActProb)
	}

	req1 := &model.LLMRequest{
		Model: slowClone, // even if started on slow, Capacitor Router drops to smol for cruise turns
		Contents: []*genai.Content{
			{Role: "user", Parts: []*genai.Part{{Text: "Add ForcePromptBaseline to Tracker"}}},
		},
	}
	if _, err := beforeModelCB(nil, req1); err != nil {
		t.Fatalf("beforeModelCB step 1: %v", err)
	}
	if req1.Model != smolClone {
		t.Fatalf("expected Capacitor Router to route routine read step to smol clone %q, got %q", smolClone, req1.Model)
	}

	// Step 2: Alpha clone makes buggy edit (unlocked + missing coupled fields)
	// AND we simulate a bloated 12-turn transcript (to test 500-token Handover Capsule compaction!)
	buggyCode := string(origBytes) + `
func (t *Tracker) ForcePromptBaseline(tokens int64) {
	t.lastPromptTokens = tokens
}
`
	if err := os.WriteFile(targetFile, []byte(buggyCode), 0o644); err != nil {
		t.Fatalf("write buggy guardrail.go: %v", err)
	}
	vBuggy := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{"file_path": targetFile},
		map[string]any{"path": targetFile, "replacements": 1},
		nil,
	)
	if vBuggy.RecommendedRole != CloneRoleSlow {
		t.Fatalf("expected RecommendedRole=%q on Bullet-Time state desync, got %q", CloneRoleSlow, vBuggy.RecommendedRole)
	}

	// Build a 13-element bloated transcript: [user goal] + 6 pairs of (model FunctionCall, user FunctionResponse)
	bloatedContents := []*genai.Content{
		{Role: "user", Parts: []*genai.Part{{Text: "Initial user goal: add ForcePromptBaseline safely"}}},
	}
	for i := 0; i < 6; i++ {
		bloatedContents = append(bloatedContents,
			&genai.Content{Role: "model", Parts: []*genai.Part{{FunctionCall: &genai.FunctionCall{Name: "read"}}}},
			&genai.Content{Role: "user", Parts: []*genai.Part{{FunctionResponse: &genai.FunctionResponse{Name: "read"}}}},
		)
	}

	req2 := &model.LLMRequest{
		Model:    smolClone,
		Contents: bloatedContents,
	}
	if _, err := beforeModelCB(nil, req2); err != nil {
		t.Fatalf("beforeModelCB step 2: %v", err)
	}
	// Verify Capacitor Router shifted model from smol -> slow!
	if req2.Model != slowClone {
		t.Fatalf("expected Capacitor Router to escalate to slow clone %q on Bullet Time, got %q", slowClone, req2.Model)
	}
	// Verify bloated 13-content transcript was compacted to <= 6 contents while preserving valid turn pairing!
	if len(req2.Contents) > 6 {
		t.Fatalf("expected bloated 13-content transcript to be compacted to <= 6 contents, got %d", len(req2.Contents))
	}
	if hasFunctionResponse(req2.Contents[1]) {
		t.Fatalf("expected compacted Contents[1] not to be an orphaned FunctionResponse")
	}
	if req2.Config == nil || req2.Config.SystemInstruction == nil {
		t.Fatalf("expected 500-token S3 Clone Handover Capsule to be injected into SystemInstruction")
	}
	sysText := req2.Config.SystemInstruction.Parts[0].Text
	if !strings.Contains(sysText, "[S3 CLONE HANDOVER CAPSULE") || !strings.Contains(sysText, "Tracker.mu") || !strings.Contains(sysText, "cachePrefixTokens") {
		t.Fatalf("expected Handover Capsule to preserve Tracker.mu and cachePrefixTokens invariants, got:\n%s", sysText)
	}

	// Step 3: Heavy clone repairs the coupled invariant -> Capacitor Router drops back to smol clone!
	fixedCode := string(origBytes) + `
func (t *Tracker) ForcePromptBaseline(tokens int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.lastPromptTokens = tokens
	t.lastCachedTokens = 0
	t.cachePrefixTokens = tokens
}
`
	if err := os.WriteFile(targetFile, []byte(fixedCode), 0o644); err != nil {
		t.Fatalf("write fixed guardrail.go: %v", err)
	}
	vFixed := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{"file_path": targetFile},
		map[string]any{"path": targetFile, "replacements": 1},
		nil,
	)
	if vFixed.RecommendedRole != CloneRoleSmol {
		t.Fatalf("expected RecommendedRole=%q after clean repair, got %q", CloneRoleSmol, vFixed.RecommendedRole)
	}
	req3 := &model.LLMRequest{Model: slowClone, Contents: req1.Contents}
	_, _ = beforeModelCB(nil, req3)
	if req3.Model != smolClone {
		t.Fatalf("expected Capacitor Router to shift back down to smol clone %q after repair, got %q", smolClone, req3.Model)
	}

	stats := engine.RouterStats()
	t.Logf("\n--- S3 CAPACITOR ROUTER & CLONE HANDOVER CAPSULE ---\nStats: steps=%d smol=%d slow=%d shifts=%d capsules=%d pruned_contents=%d\n%s\n----------------------------------------------------",
		stats.TotalSteps, stats.SmolSteps, stats.SlowSteps, stats.RoleShifts, stats.CapsulesInjected, stats.ContentsPruned, stats.LastHandoverCapsule)
}

func TestCrossFileBlastRadiusScanner_GoInterfacesAndPortalWireContracts(t *testing.T) {
	engine := NewEngineFromEnv()

	// 1. Test Go Cross-Package Symbol & Duck-Typed Interface Impact (guardrail.SetLastPromptTokens)
	guardrailOrig, err := os.ReadFile(filepath.Join("..", "guardrail", "guardrail.go"))
	if err != nil {
		t.Fatalf("read guardrail.go: %v", err)
	}
	tmpDir := t.TempDir()
	tmpGuardrail := filepath.Join(tmpDir, "guardrail.go")
	if err := os.WriteFile(tmpGuardrail, guardrailOrig, 0o644); err != nil {
		t.Fatalf("write tmp guardrail.go: %v", err)
	}

	// Register pi-go's internal/ directory as search root for the temp copy
	absInternal, _ := filepath.Abs("..")
	engine.AddSearchRoot(absInternal)

	vGo := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{
			"file_path":  tmpGuardrail,
			"old_string": "t.lastPromptTokens = n",
			"new_string": "t.lastPromptTokens = n // modified inside SetLastPromptTokens",
		},
		map[string]any{"path": tmpGuardrail, "replacements": 1},
		nil,
	)
	if vGo.BlastRadius == nil {
		t.Fatalf("expected BlastRadius report on guardrail.SetLastPromptTokens edit")
	}
	if vGo.BlastRadius.Level != BlastLevelCrossPackage {
		t.Fatalf("expected BlastLevelCrossPackage on SetLastPromptTokens edit, got %s", vGo.BlastRadius.Level)
	}
	if !strings.Contains(vGo.BlastRadius.Summary, "autocompact/hook.go") || !strings.Contains(vGo.BlastRadius.Summary, "tui/types.go") {
		t.Fatalf("expected BlastRadius summary to surface autocompact/hook.go and tui/types.go interface consumers, got:\n%s", vGo.BlastRadius.Summary)
	}
	t.Logf("\n--- S3 BLAST-RADIUS SCANNER (pi-go guardrail.SetLastPromptTokens) ---\n%s\n---------------------------------------------------------------------", vGo.BlastRadius.Summary)

	// 2. Test Cross-Service Wire-Contract Impact on ~/portal (gateway.py -> runner.py, task.sh, taskbot.py)
	portalGatewayPath := "/Users/mp/portal/apps/model-gateway/files/gateway.py"
	if gwOrig, err := os.ReadFile(portalGatewayPath); err == nil {
		tmpGw := filepath.Join(tmpDir, "gateway.py")
		if err := os.WriteFile(tmpGw, gwOrig, 0o644); err != nil {
			t.Fatalf("write tmp gateway.py: %v", err)
		}
		vPy := engine.EvaluateToolTurn(
			context.Background(),
			"edit",
			map[string]any{
				"file_path":   tmpGw,
				"search_root": "/Users/mp/portal/apps",
				"old_string":  `out["over_budget"] = ("task" if out["task_usd"] >= self.task_usd else`,
				"new_string":  `out["over_budget"] = ("task" if out["task_usd"] >= self.task_usd else`,
			},
			map[string]any{"path": tmpGw, "replacements": 1},
			nil,
		)
		if vPy.BlastRadius == nil {
			t.Fatalf("expected BlastRadius report on gateway.py Ledger.status wire-contract edit")
		}
		if vPy.BlastRadius.Level != BlastLevelCrossService {
			t.Fatalf("expected BlastLevelCrossService on gateway.py wire-contract edit, got %s", vPy.BlastRadius.Level)
		}
		if !strings.Contains(vPy.BlastRadius.Summary, "runner.py") || !strings.Contains(vPy.BlastRadius.Summary, "taskbot.py") {
			t.Fatalf("expected BlastRadius summary to identify cross-service wire consumers in runner.py and taskbot.py, got:\n%s", vPy.BlastRadius.Summary)
		}
		t.Logf("\n--- S3 BLAST-RADIUS SCANNER (portal gateway.py -> runner.py & taskbot.py) ---\n%s\n-----------------------------------------------------------------------------", vPy.BlastRadius.Summary)
	}

	// 3. Test Upgrade 4 & 5: Unsoundness-Aware Symbolic Predicates (CodeAnchor/CaMeLoT)
	// & Role-Stratified Conformal Risk Control (TIER2_HIGH_BLAST_OR_DYNAMIC)
	dynamicPy := `class DynamicRouter:
    def dispatch(self, action_name, **kwargs):
        handler = getattr(self, "handle_" + action_name, None)
        return handler(**kwargs)
`
	tmpDyn := filepath.Join(tmpDir, "dynamic_router.py")
	if err := os.WriteFile(tmpDyn, []byte(dynamicPy), 0o644); err != nil {
		t.Fatalf("write dynamic_router.py: %v", err)
	}
	vDyn := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{
			"file_path":  tmpDyn,
			"old_string": "return handler(**kwargs)",
			"new_string": "return handler(**kwargs)",
		},
		map[string]any{"path": tmpDyn, "replacements": 1},
		nil,
	)
	if vDyn.BlastRadius == nil || !vDyn.BlastRadius.UnsoundnessDetected {
		t.Fatalf("expected UnsoundnessDetected=true on Python getattr/**kwargs dynamic dispatch, got: %+v", vDyn.BlastRadius)
	}
	if vDyn.RiskTier != "TIER2_HIGH_BLAST_OR_DYNAMIC" || vDyn.RecommendedRole != CloneRoleSlow || !vDyn.BulletTimeEngaged {
		t.Fatalf("expected TIER2_HIGH_BLAST_OR_DYNAMIC and escalation to slow clone on unsound dynamic dispatch, got tier=%s role=%s", vDyn.RiskTier, vDyn.RecommendedRole)
	}
	t.Logf("\n--- S3 v2 UNSOUNDNESS-AWARE PREDICATE & ROLE-STRATIFIED RISK CONTROL ---\nRiskTier=%s RecommendedRole=%s SignedGain=%.2f (Rescue=%.2f Harm=%.2f)\n%s\n------------------------------------------------------------------------",
		vDyn.RiskTier, vDyn.RecommendedRole, vDyn.SignedGain, vDyn.RescueProb, vDyn.HarmProb, vDyn.Advisory)
}







