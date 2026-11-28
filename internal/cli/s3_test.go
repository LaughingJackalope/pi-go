package cli

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestS3CLISubcommands_AuditBlastAndHook(t *testing.T) {
	tmpDir := t.TempDir()
	goFile := filepath.Join(tmpDir, "store.go")
	srcA := `package store

import "sync"

type Cache struct {
	mu    sync.Mutex
	items map[string]string
	index map[string]int
}

func (c *Cache) Put(k, v string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[k] = v
	c.index[k] = len(v)
}

func (c *Cache) Delete(k string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, k)
	delete(c.index, k)
}

func (c *Cache) Get(k string) (string, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.items[k], c.index[k]
}
`
	if err := os.WriteFile(goFile, []byte(srcA), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. Test `pi s3 audit --json`
	root := newRootCmd()
	var auditBuf bytes.Buffer
	root.SetOut(&auditBuf)
	root.SetArgs([]string{"s3", "audit", "--json", tmpDir})
	if err := root.Execute(); err != nil {
		t.Fatalf("s3 audit failed: %v", err)
	}
	var auditOut map[string]any
	if err := json.Unmarshal(auditBuf.Bytes(), &auditOut); err != nil {
		t.Fatalf("unmarshal audit JSON: %v\nraw: %s", err, auditBuf.String())
	}
	if int(auditOut["coupled_pairs_count"].(float64)) < 1 {
		t.Fatalf("expected >=1 coupled pair in audit JSON, got: %v", auditOut)
	}

	// 2. Test `pi s3 hook pre-tool` ECCM auto-lock + `post-tool` A->B->A Warp Scrambler
	stateFile := filepath.Join(tmpDir, "hook_state.json")

	// Pre-tool with 4-space indentation instead of Go tabs -> ECCM snaps old_string!
	prePayload := map[string]any{
		"tool_name": "Edit",
		"tool_input": map[string]any{
			"file_path":  goFile,
			"old_string": "    return c.items[k], c.index[k]",
			"new_string": "    return c.items[k], 0",
		},
	}
	preBytes, _ := json.Marshal(prePayload)
	rootPre := newRootCmd()
	var preOut bytes.Buffer
	rootPre.SetIn(bytes.NewReader(preBytes))
	rootPre.SetOut(&preOut)
	rootPre.SetArgs([]string{"s3", "hook", "pre-tool", "--state-file", stateFile})
	if err := rootPre.Execute(); err != nil {
		t.Fatalf("s3 hook pre-tool failed: %v", err)
	}
	if !strings.Contains(preOut.String(), `"eccm_auto_locked":true`) {
		t.Fatalf("expected ECCM auto-lock on pre-tool hook, got: %s", preOut.String())
	}

	// Transition file to State B and run post-tool
	srcB := strings.Replace(srcA, "return c.items[k], c.index[k]", "return c.items[k], 0", 1)
	_ = os.WriteFile(goFile, []byte(srcB), 0o644)
	postPayload := map[string]any{
		"tool_name":  "Edit",
		"tool_input": map[string]any{"file_path": goFile},
	}
	postBytes, _ := json.Marshal(postPayload)
	rootPostB := newRootCmd()
	var postOutB bytes.Buffer
	rootPostB.SetIn(bytes.NewReader(postBytes))
	rootPostB.SetOut(&postOutB)
	rootPostB.SetArgs([]string{"s3", "hook", "post-tool", "--state-file", stateFile})
	if err := rootPostB.Execute(); err != nil {
		t.Fatalf("s3 hook post-tool B failed: %v", err)
	}

	// Revert file to State A (with extra comments) -> post-tool must detect A->B->A Warp Scrambler cycle!
	srcARevert := "// extra comment\n" + srcA
	_ = os.WriteFile(goFile, []byte(srcARevert), 0o644)
	rootPostA := newRootCmd()
	var postOutA bytes.Buffer
	rootPostA.SetIn(bytes.NewReader(postBytes))
	rootPostA.SetOut(&postOutA)
	rootPostA.SetArgs([]string{"s3", "hook", "post-tool", "--state-file", stateFile})
	if err := rootPostA.Execute(); err != nil {
		t.Fatalf("s3 hook post-tool A-revert failed: %v", err)
	}
	if !strings.Contains(postOutA.String(), "Warp Scrambler") {
		t.Fatalf("expected post-tool hook to detect A->B->A Warp Scrambler cycle across invocations, got: %s", postOutA.String())
	}
}

func TestS3HookAdaptiveReaderPolicy(t *testing.T) {
	t.Setenv("PI_SYSTEM1_URL", "off")
	tmpDir := t.TempDir()
	goFile := filepath.Join(tmpDir, "store.go")
	src := `package store

import "sync"

type Cache struct {
	mu    sync.Mutex
	items map[string]int
	index map[string]int
}

func (c *Cache) Put(k string, v int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items[k] = v
	c.index[k] = v
}

func (c *Cache) Delete(k string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.items, k)
	delete(c.index, k)
}
`
	if err := os.WriteFile(goFile, []byte(src), 0o644); err != nil {
		t.Fatalf("write store.go: %v", err)
	}
	stateFile := filepath.Join(tmpDir, "adaptive_state.json")

	callRead := func(start, end int) map[string]any {
		payload := map[string]any{
			"tool_name": "Read",
			"tool_input": map[string]any{
				"file_path":  goFile,
				"start_line": start,
				"end_line":   end,
			},
		}
		raw, _ := json.Marshal(payload)
		cmd := newRootCmd()
		var out bytes.Buffer
		cmd.SetIn(bytes.NewReader(raw))
		cmd.SetOut(&out)
		cmd.SetArgs([]string{"s3", "hook", "post-tool", "--state-file", stateFile})
		if err := cmd.Execute(); err != nil {
			t.Fatalf("hook read failed: %v", err)
		}
		var resp map[string]any
		if err := json.Unmarshal(out.Bytes(), &resp); err != nil {
			t.Fatalf("unmarshal hook resp: %v", err)
		}
		return resp
	}

	// 1. First narrow surgical read (lines 1..5 of ~24 = 20%) MUST emit Pre-Flight Radar
	r1 := callRead(1, 5)
	if ctx, _ := r1["additionalContext"].(string); !strings.Contains(ctx, "PRE-FLIGHT RADAR") {
		t.Fatalf("expected surgical first read to emit Pre-Flight Radar, got: %+v", r1)
	}

	// 2. Second read of the same file MUST suppress duplicate radar ("already_emitted")
	r2 := callRead(6, 12)
	if reason, _ := r2["radar_suppressed_reason"].(string); reason != "already_emitted" {
		t.Fatalf("expected second read of same file to suppress radar with already_emitted, got: %+v", r2)
	}
	if _, hasCtx := r2["additionalContext"]; hasCtx {
		t.Fatalf("expected no additionalContext when radar suppressed, got: %+v", r2)
	}

	// 3. Fresh state with wide exhaustive read (>=60% of file) MUST suppress radar immediately
	stateFile2 := filepath.Join(tmpDir, "adaptive_state_wide.json")
	payloadWide, _ := json.Marshal(map[string]any{
		"tool_name": "Read",
		"tool_input": map[string]any{
			"file_path":  goFile,
			"start_line": 1,
			"end_line":   22,
		},
	})
	cmdWide := newRootCmd()
	var outWide bytes.Buffer
	cmdWide.SetIn(bytes.NewReader(payloadWide))
	cmdWide.SetOut(&outWide)
	cmdWide.SetArgs([]string{"s3", "hook", "post-tool", "--state-file", stateFile2})
	if err := cmdWide.Execute(); err != nil {
		t.Fatalf("hook wide read failed: %v", err)
	}
	var rWide map[string]any
	_ = json.Unmarshal(outWide.Bytes(), &rWide)
	if reason, _ := rWide["radar_suppressed_reason"].(string); reason != "exhaustive_reader_coverage" {
		t.Fatalf("expected wide read to suppress radar with exhaustive_reader_coverage, got: %+v", rWide)
	}
}
