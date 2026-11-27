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
