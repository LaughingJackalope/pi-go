package sense

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPolyglotJavaBufferPoolAndBuggyEdit(t *testing.T) {
	javaSrc := `package org.apache.kafka.clients.producer.internals;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.concurrent.locks.ReentrantLock;

public class BufferPool {
    private final ReentrantLock lock = new ReentrantLock();
    private final Deque<Object> free = new ArrayDeque<>();
    private long nonPooledAvailableMemory = 1024;

    public Object allocate(int size) {
        this.lock.lock();
        try {
            freeUp(size);
            this.nonPooledAvailableMemory -= size;
            return this.free.pollFirst();
        } finally {
            this.lock.unlock();
        }
    }

    public void deallocate(Object buf, int size) {
        this.lock.lock();
        try {
            this.free.addLast(buf);
            this.nonPooledAvailableMemory += size;
        } finally {
            this.lock.unlock();
        }
    }

    private void freeUp(int size) {
        while (!this.free.isEmpty() && this.nonPooledAvailableMemory < size) {
            this.free.pollLast();
            this.nonPooledAvailableMemory += 256;
        }
    }
}
`
	tmpDir := t.TempDir()
	javaFile := filepath.Join(tmpDir, "BufferPool.java")
	if err := os.WriteFile(javaFile, []byte(javaSrc), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. Clean Java BufferPool (including private helper freeUp) must report 0 anomalies!
	gClean, lang, err := AnalyzeAnyDir(tmpDir, javaFile)
	if err != nil {
		t.Fatalf("AnalyzeAnyDir java failed: %v", err)
	}
	if lang != "java" {
		t.Fatalf("expected lang=java, got %q", lang)
	}
	if len(gClean.Anomalies) != 0 {
		t.Fatalf("expected 0 anomalies on clean BufferPool.java (private helper freeUp called under lock), got: %+v", gClean.Anomalies)
	}

	// 2. Inject a myopic agent edit: public forceReclaim() mutating `free` without `lock` or `nonPooledAvailableMemory`
	buggyJava := strings.Replace(javaSrc, "    private void freeUp", `    public void forceReclaim() {
        this.free.clear();
    }

    private void freeUp`, 1)
	if err := os.WriteFile(javaFile, []byte(buggyJava), 0o644); err != nil {
		t.Fatal(err)
	}

	engine := NewEngineFromEnv()
	verdict := engine.EvaluateToolTurn(
		context.Background(),
		"edit",
		map[string]any{"file_path": javaFile},
		map[string]any{"path": javaFile, "replacements": 1},
		nil,
	)
	if !verdict.BulletTimeEngaged || verdict.Posture != PostureInspect {
		t.Fatalf("expected Bullet Time INSPECT on buggy Java edit, got: %+v", verdict)
	}
	if !strings.Contains(verdict.Advisory, "BufferPool.forceReclaim") ||
		!strings.Contains(verdict.Advisory, "BufferPool.lock") ||
		!strings.Contains(verdict.Advisory, "nonPooledAvailableMemory") {
		t.Fatalf("expected Java advisory to flag BufferPool.forceReclaim missing lock & nonPooledAvailableMemory, got:\n%s", verdict.Advisory)
	}
}

func TestPolyglotScalaPartitionInWriteLockAndDesync(t *testing.T) {
	scalaSrc := `package kafka.cluster

import java.util.concurrent.locks.ReentrantReadWriteLock
import org.apache.kafka.server.util.LockUtils.{inReadLock, inWriteLock}

class Partition {
  private val leaderIsrUpdateLock = new ReentrantReadWriteLock
  @volatile private var leaderEpoch: Int = 0
  @volatile private var partitionState: String = "init"

  def makeLeader(newEpoch: Int, state: String): Unit = {
    inWriteLock(leaderIsrUpdateLock) {
      leaderEpoch = newEpoch
      partitionState = state
    }
  }

  def makeFollower(newEpoch: Int, state: String): Unit = {
    inWriteLock(leaderIsrUpdateLock) {
      leaderEpoch = newEpoch
      partitionState = state
    }
  }

  def currentLeaderAndState(): String = {
    inReadLock(leaderIsrUpdateLock) {
      s"$leaderEpoch:$partitionState"
    }
  }
}
`
	tmpDir := t.TempDir()
	scalaFile := filepath.Join(tmpDir, "Partition.scala")
	if err := os.WriteFile(scalaFile, []byte(scalaSrc), 0o644); err != nil {
		t.Fatal(err)
	}

	gClean, lang, err := AnalyzeAnyDir(tmpDir, scalaFile)
	if err != nil || lang != "scala" {
		t.Fatalf("AnalyzeAnyDir scala failed: lang=%q err=%v", lang, err)
	}
	if len(gClean.Anomalies) != 0 {
		t.Fatalf("expected 0 anomalies on clean Scala Partition, got: %+v", gClean.Anomalies)
	}

	// Inject buggy Scala method mutating leaderEpoch without inWriteLock or partitionState
	buggyScala := strings.Replace(scalaSrc, "  def currentLeaderAndState", `  def forceEpochBump(nextEpoch: Int): Unit = {
    leaderEpoch = nextEpoch
  }

  def currentLeaderAndState`, 1)
	if err := os.WriteFile(scalaFile, []byte(buggyScala), 0o644); err != nil {
		t.Fatal(err)
	}

	gBuggy, _, err := AnalyzeAnyDir(tmpDir, scalaFile)
	if err != nil {
		t.Fatal(err)
	}
	if len(gBuggy.Anomalies) < 2 {
		t.Fatalf("expected both UNLOCKED_STATE_WRITE and MISSING_COUPLED_WRITE on Scala Partition.forceEpochBump, got: %+v", gBuggy.Anomalies)
	}
}

func TestPolyglotTypeScriptModuleSingletonAndCrossLanguageBlastRadius(t *testing.T) {
	tmpRoot := t.TempDir()
	pyDir := filepath.Join(tmpRoot, "backend")
	tsDir := filepath.Join(tmpRoot, "web", "src", "plugins")
	if err := os.MkdirAll(pyDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(tsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	// 1. TypeScript module-scoped singleton registry (modeled on Hermes web/src/plugins/registry.ts)
	tsRegistry := `const _registered: Map<string, unknown> = new Map();
const _loadErrors: Map<string, string> = new Map();

export function selectPluginProfile(): void {
  _registered.clear();
  _loadErrors.clear();
}

export function registerPlugin(name: string, comp: unknown): void {
  _loadErrors.delete(name);
  _registered.set(name, comp);
}

export function getPluginStatus(name: string): boolean {
  return _registered.has(name) && !_loadErrors.has(name);
}

export function buggyEvictPlugin(name: string): void {
  _registered.delete(name);
}
`
	tsFile := filepath.Join(tsDir, "registry.ts")
	if err := os.WriteFile(tsFile, []byte(tsRegistry), 0o644); err != nil {
		t.Fatal(err)
	}

	gTS, lang, err := AnalyzeAnyDir(tsDir, tsFile)
	if err != nil || lang != "typescript" {
		t.Fatalf("AnalyzeAnyDir typescript failed: lang=%q err=%v", lang, err)
	}
	foundCoupled := false
	for _, a := range gTS.Anomalies {
		if a.MethodName == "buggyEvictPlugin" && a.PeerField == "_loadErrors" {
			foundCoupled = true
		}
	}
	if !foundCoupled {
		t.Fatalf("expected TypeScript module-singleton sensor to flag buggyEvictPlugin missing _loadErrors, got: %+v", gTS.Anomalies)
	}

	// 2. Cross-language Python FastAPI <-> TypeScript fetch wire-contract blast radius
	pyRoutes := `from fastapi import APIRouter
router = APIRouter()

@router.get("/api/dashboard/plugins")
def list_plugins():
    return {"session_expired": False}
`
	pyFile := filepath.Join(pyDir, "routes.py")
	if err := os.WriteFile(pyFile, []byte(pyRoutes), 0o644); err != nil {
		t.Fatal(err)
	}

	tsConsumer := "export async function loadPlugins(profile: string) {\n  const res = await fetch(`/api/dashboard/plugins${profile ? '?p=1' : ''}`);\n  const body = await res.json();\n  if (body.error === 'session_expired') return [];\n  return body;\n}\n"
	if err := os.WriteFile(filepath.Join(tsDir, "api.ts"), []byte(tsConsumer), 0o644); err != nil {
		t.Fatal(err)
	}

	engine := NewEngineFromEnv()
	br := engine.ScanBlastRadius(pyFile, map[string]any{
		"search_root": tmpRoot,
		"old_string":  `return {"session_expired": False}`,
		"new_string":  `return {"session_expired": True}`,
	})
	if br == nil || br.Level != BlastLevelCrossService || br.CrossLanguageHits == 0 {
		t.Fatalf("expected CROSS_SERVICE cross-language Python->TS blast radius, got: %+v", br)
	}
	if !strings.Contains(br.Summary, "web/src/plugins/api.ts") || !strings.Contains(br.Summary, "/api/dashboard/plugins") {
		t.Fatalf("expected blast summary to pinpoint TS template-literal consumer in web/src/plugins/api.ts, got:\n%s", br.Summary)
	}
}
