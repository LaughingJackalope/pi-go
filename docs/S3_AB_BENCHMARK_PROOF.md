# Reproducible Public Multi-Repo, 5-Language Controlled A/B Proof (`pi-s3` + `s3-cortexd`)

> **Summary:** Across **32 non-toy coding tasks (`64` full agent trajectories)** drawn from **4 public repositories (`kubernetes/kubernetes`, `apache/kafka`, `NousResearch/hermes-agent`, `dimetron/pi-go`)** and **5 languages (`Go`, `Java`, `Scala`, `Python`, `TypeScript`)**, augmenting the exact same model (`gemma4:latest`, `temperature = 0.0`, `seed = 42`) with `pi-s3 s3 hook` + `s3-cortexd` lifted **verified bug-free completion from `43.8%` (`14/32`) to `65.6%` (`21/32`, `+21.8 pp` / `+50.0%` relative)** and reduced **silent structural bugs escaping into the final diff by `63.4%` (`41` -> `15`)** at **`0.67%` S3 engine latency overhead** (`434 ms/task`) and **`1.69–4.34 MB` dirty daemon RAM** across `42+ hours` of continuous Apple M2 MacBook Air uptime.

---

## 1. Experimental Protocol

- **Controlled Variable:** Only the subcortical hook layer changes between arms:
  1. **Arm A (`baseline`):** Standard coding-agent tool loop (`read_file`, `edit_file`, `grep_repo`, `finish_task`, max 6 turns) with symmetric atomic syntax rollback (`gofmt -e`, `ast.parse`, brace balance).
  2. **Arm B (`s3_hooked`):** Identical model (`gemma4:latest` `8.0B` Q4\_K\_M), temperature (`0.0`), seed (`42`), system prompt, task prompts, and tools, augmented *only* by `pi-s3 s3 hook pre-tool` + `post-tool` + a Stop-Hook gate (`pi-s3 s3 audit` + unresolved `s3_blast_radius` downstream consumer check). Zero ground-truth grader assertions are exposed to `s3_hooked`.
- **Hardware:** Fanless **Apple M2 MacBook Air (`24 GB` unified memory)** running both the `9.6 GB` LLM on Metal GPU and the native Swift Dual-Tier `s3-cortexd` daemon (`mmBERT-base` 322M L96 on ANE -> `ModernBERT-large` 420M L512 on GPU) concurrently.

---

## 2. Overall Results (`N = 32` Tasks, `64` Trajectories)

| Metric | Arm A: `baseline` | Arm B: `s3_hooked` | Delta / Impact |
| :--- | :---: | :---: | :---: |
| **Apparent Functional Completion** | `27/32` (`84.4%`) | `27/32` (`84.4%`) | `0.0 pp` (same primary-file pass rate) |
| **Verified Clean Completion (0 Escaped Bugs)** | **`14/32` (`43.8%`)** | **`21/32` (`65.6%`)** | **`+21.8 pp` (`+50.0%` relative lift)** |
| **Silent Structural Bugs Escaping to Final Diff** | **`41` bugs** (`13` tasks) | **`15` bugs** (`6` tasks) | **`-63.4%` escaped bugs (`-53.8%` buggy PRs)** |
| **Total Turns / Avg Turns per Task** | `113` (`3.53` / task) | `138` (`4.31` / task) | `+0.78` turns/task (multi-file & repair work) |
| **Total Tool Calls** | `111` | `137` | `+26` tool calls |
| **Wasted Turns (`old_string` miss / syntax / dup read)** | `22` | `32` | `+10` (see Section 5 8B repair-loop analysis) |
| **ECCM Pre-Tool Whitespace Auto-Locks** | `0` | `2` | Prevented Turn-1 space-vs-tab edit failures |
| **S3 Stop-Hook Vetoes Triggered** | `0` | `4` | Blocked premature exit on broken state/consumers |
| **Total Input + Output Tokens** | `141,382` (`126.9k` / `14.5k`) | `234,522` (`217.4k` / `17.1k`) | `+65.9%` tokens (`$0.60` -> `$0.91` ref API eq.) |
| **S3 Engine Latency Overhead** | `0.0 ms` | **`434.0 ms / task`** | **`0.67%` of LLM wall-clock runtime** |
| **End-to-End Subprocess CLI Hook Overhead** | `0.0 ms` | `948.4 ms / task` | `1.46%` of LLM wall-clock runtime |

---

## 3. Stratified Results by Category & Repository

### 3.1 By Failure-Mode Category (Including Stateless Negative Controls)

| Task Category (`N`) | `baseline` Clean Pass | `s3_hooked` Clean Pass | Escaped Bugs (`B` -> `S3`) | Avg Turns (`B` -> `S3`) | Wasted Turns (`B` -> `S3`) | Mechanistic Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`stateful_narrow_window` (`16`)** | `7/16` (`43.8%`) | **`11/16` (`68.8%`)** | **`19` -> `7` (`-63.2%`)** | `3.38` -> `4.31` | `9` -> `23` | `[S3 PRE-FLIGHT RADAR]` exposed off-screen locks & coupled maps on `read_file`, rescuing 4 concurrency/desync tasks in **zero extra turns** (`3` turns each). |
| **`cross_file_contract` (`8`)** | `0/8` (`0.0%`) | **`3/8` (`37.5%`)** | **`22` -> `8` (`-63.6%`)** | `3.88` -> `5.50` | `7` -> `7` | `baseline` edited only the primary file (`0/8` clean). `s3_blast_radius` surfaced downstream Go/TS callers & cross-lang REST routes, cutting broken consumers `22` -> `8`. |
| **`edit_drift_eccm` (`2`)** | `1/2` (`50.0%`) | `1/2` (`50.0%`) | `0` -> `0` | **`5.00` -> `3.50`** | **`6` -> `2` (`-66.7%`)** | `ECCMAutoLockEdit` aligned space-indented prompt snippets to tab-indented Go ASTs (`17,609` -> `11,222` tokens, `-36.3%`). |
| **`stateless_control` (`6`)** | `6/6` (`100.0%`) | **`6/6` (`100.0%`)** | `0` -> `0` | **`3.00` -> `3.00`** | **`0` -> `0`** | **Zero false positives / zero regressions:** pure helpers across Go, Java, Scala, Python, and TS ran in 3.0 turns with 0 wasted turns and 0 false vetoes. |

### 3.2 By Public Upstream Repository

| Repository (`Languages`, `N`) | `baseline` Clean Pass | `s3_hooked` Clean Pass | Escaped Bugs (`B` -> `S3`) | S3 Engine Overhead / Task |
| :--- | :---: | :---: | :---: | :---: |
| **`kubernetes/kubernetes`** (`Go`, `9`) | `4/9` (`44.4%`) | **`6/9` (`66.7%`)** | **`18` -> `7` (`-61.1%`)** | `266.2 ms` |
| **`NousResearch/hermes-agent`** (`Python` + `TypeScript`, `9`) | `3/9` (`33.3%`) | **`6/9` (`66.7%`)** | **`14` -> `4` (`-71.4%`)** | `651.7 ms` |
| **`dimetron/pi-go`** (`Go`, `5`) | `1/5` (`20.0%`) | **`3/5` (`60.0%`)** | **`5` -> `0` (`-100.0%`)** | `245.2 ms` |
| **`apache/kafka`** (`Java` + `Scala`, `9`) | `6/9` (`66.7%`) | `6/9` (`66.7%`) | `4` -> `4` (`0.0%`) | `489.0 ms` |

---

## 4. Verbatim Diffs From Raw Trajectory Logs

### Example 1: Kubernetes `pod_manager.go` — Preventing a 5-Map Coupled-State Desync (`T02`)
```diff
# Arm A: baseline (3 turns, 6 ESCAPED COUPLED-STATE BUGS — podByUID, mirrorPodByUID, translationByUID left stale)
+func (pm *basicManager) ResetAllPods() { 
+	pm.lock.Lock()
+	defer pm.lock.Unlock()
+	pm.podByFullName = make(map[string]*v1.Pod)
+	pm.mirrorPodByFullName = make(map[string]*v1.Pod)
+}

# Arm B: s3_hooked (3 turns, 0 ESCAPED BUGS — Pre-Flight Radar surfaced all 5 co-mutated maps in 210 ms)
+func (pm *basicManager) ResetAllPods() {
+	pm.lock.Lock()
+	defer pm.lock.Unlock()
+
+	pm.mirrorPodByFullName = make(map[string]*v1.Pod)
+	pm.mirrorPodByUID = make(map[string]*v1.Pod)
+	pm.podByFullName = make(map[string]*v1.Pod)
+	pm.podByUID = make(map[string]*v1.Pod)
+	pm.translationByUID = make(map[string]string)
+}
```

### Example 2: Hermes Agent `bot_loop_guard.py` — Preventing Unlocked Mutex Races (`T20`)
```diff
# Arm A: baseline (3 turns, 4 ESCAPED UNLOCKED_STATE_WRITE RACES)
+    def reset_all_guards(self) -> None:
+        """Clears all tracked state and resets the sweep timer."""
+        self._events.clear()
+        self._cooldown_until.clear()
+        self._chains.clear()
+        self._last_sweep = 0.0

# Arm B: s3_hooked (3 turns, 0 ESCAPED BUGS)
+    def reset_all_guards(self) -> None:
+        """Clears all tracked state and resets the guard's internal timers."""
+        with self._lock:
+            self._chains.clear()
+            self._events.clear()
+            self._cooldown_until.clear()
+            self._last_sweep = 0.0
```

### Example 3: Hermes Agent Cross-Language Python FastAPI <-> TypeScript Frontend (`T21`)
```diff
# Arm A: baseline — updated ONLY Python dashboard_ui.py, leaving TypeScript web/src/lib/api.ts broken on 404!
-@router.get("/api/dashboard/plugins/rescan")
+@router.get("/api/dashboard/plugins/reload")

# Arm B: s3_hooked — s3_blast_radius flagged [cross-lang] consumer in web/src/lib/api.ts:1079 and updated both
--- a/web/src/lib/api.ts
+++ b/web/src/lib/api.ts
-  rescanPlugins: () =>
-    fetchJSON<{ ok: boolean; count: number }>("/api/dashboard/plugins/rescan"),
+  reloadPlugins: () =>
+    fetchJSON<{ ok: boolean; count: number }>("/api/dashboard/plugins/reload"),
```

---

## 5. Failure-Mode & Trade-Off Analysis

1. **Productive multi-file edits (`+13` turns in `cross_file_contract`):** `baseline` stopped prematurely after editing 1 file (`0/8` clean pass), whereas `s3_hooked` discovered and edited 2–4 downstream consumer files across packages and languages (`0%` -> `37.5%` clean pass, `22` -> `8` broken consumers).
2. **8B model repair-loop friction (`9` -> `23` wasted turns in `stateful_narrow_window`):** When `post-tool` `Edit` surfaced a partial coupled-state warning (`T01`, `T05`, `T15`, `T24`), `gemma4:latest` (`8B` Q4\_K\_M) attempted a follow-up repair edit and occasionally missed `old_string` or brace boundaries (`T05` regressed on syntax rollback).
3. **Zero tax on clean stateless code (`6/6` `stateless_control`):** Across all 5 languages (`Go`, `Java`, `Scala`, `Python`, `TypeScript`), `s3_hooked` added **0 extra turns (`3.00` vs. `3.00`), 0 wasted turns, and 0 false vetoes**.
