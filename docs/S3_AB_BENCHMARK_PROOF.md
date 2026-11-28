# Reproducible Public Multi-Repo, 5-Language Controlled A/B Proof (`pi-s3` + `s3-cortexd`)

> **Summary:** Across **32 non-toy coding tasks (`352` full agent trajectories)** drawn from **4 public repositories (`kubernetes/kubernetes`, `apache/kafka`, `NousResearch/hermes-agent`, `dimetron/pi-go`)** and **5 languages (`Go`, `Java`, `Scala`, `Python`, `TypeScript`)** across **5 models + S3 Capacitor Router**, augmenting agents with `pi-s3 s3 hook` + `s3-cortexd` lifted **verified bug-free completion across local and frontier flagships (`+9.4 pp` to `+21.8 pp`)** and eliminated **`100%` of narrow-window concurrency and coupled-state bugs on both `openai/gpt-5.3-codex` (`18` -> `0`) and `x-ai/grok-4.7` (`19` -> `0`)** at **`234–434 ms/task` S3 engine overhead** and **`1.69 MB` idle dirty daemon RAM** across `43.6 hours` of continuous Apple M2 MacBook Air uptime.

---

## 1. Multi-Model Headline Matrix (`N = 32` Tasks · `352` Trajectories)

| Model Tier & ID (`32` Tasks / Arm) | `baseline` Functional -> **Clean Pass** | `s3_hooked` Functional -> **Clean Pass** | Clean Lift | Escaped Silent Bugs (`B` -> `S3`) | Avg Turns (`B` -> `S3`) | Wasted Turns (`B` -> `S3`) | Total Cost (`B` -> `S3`) | S3 Engine Overhead |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Local Edge:** `ollama/gemma4:latest` (`8B` Q4) | `84.4%` -> **`43.8%`** (`14/32`) | `84.4%` -> **`65.6%`** (`21/32`) | **`+21.8 pp`** | **`41` -> `15` (`-63.4%`)** | `3.53` -> `4.31` | `22` -> `32` | `$0.00` -> `$0.00` | `0.67%` (`434 ms`) |
| **Frontier Daily-Driver:** `anthropic/claude-sonnet-5.5` | `87.5%` -> **`62.5%`** (`20/32`) | **`100.0%`** -> **`75.0%`** (`24/32`) | **`+12.5 pp`** | `26` -> `30`* | `5.00` -> `5.12` | **`45` -> `42`** | `$0.72` -> `$0.98` | `2.10%` (`234 ms`) |
| **RL Repo Specialist:** `openai/gpt-5.3-codex` | `93.8%` -> **`65.6%`** (`21/32`) | `87.5%` -> **`78.1%`** (`25/32`) | **`+12.5 pp`** | **`34` -> `10` (`-70.6%`)** | `4.12` -> `4.50` | `3` -> `9` | `$0.73` -> `$0.84` | `1.12%` (`258 ms`) |
| **Frontier Flagship:** `x-ai/grok-4.7` | `100.0%` -> **`75.0%`** (`24/32`) | `93.8%` -> **`84.4%`** (`27/32`) | **`+9.4 pp`** | **`27` -> `10` (`-63.0%`)** | `4.62` -> `4.66` | **`1` -> `0`** | `$0.44` -> `$0.54` | `2.39%` (`245 ms`) |
| **Ultra-Cheap Flash:** `deepseek/deepseek-v4.1-flash` | `96.9%` -> **`78.1%`** (`25/32`) | **`100.0%`** -> **`75.0%`** (`24/32`) | `-3.1 pp` | `19` -> `30`* | **`4.97` -> `4.88`** | `0` -> `1` | `$0.03` -> `$0.04` | `4.22%` (`257 ms`) |
| **Capacitor Router (`s3_routed`):** `deepseek-v4.1-flash` -> `claude-sonnet-5.5` (`17.1%` slow) | *(vs. unhooked Sonnet `62.5%` @ `$0.72`)* | `96.9%` -> **`71.9%`** (`23/32`) | **`+9.4 pp` vs. Sonnet B** | `30` bugs (`8` tasks) | `5.12` (`28/164` slow) | `14` | **`$0.45` (`-38.3%` vs. Sonnet B)** | `3.75%` (`256 ms`) |

*\*When a `baseline` trajectory fails to implement the target method at all (`func=0`, e.g. `claude-sonnet-5.5` on `T03` or `deepseek-v4.1-flash` on `T15`), `0` bugs are tallied on the missing method; when `s3_hooked` achieves `100%` (`32/32`) functional completion, partial coupled-field omissions inside newly completed `T03`/`T15` are counted.*

---

## 2. Category Stratification Across All 5 Models

| Model | `stateful_narrow_window` (`16` Tasks)<br/>Clean Pass & Escaped Bugs (`B` -> `S3`) | `cross_file_contract` (`8` Tasks)<br/>Clean Pass & Escaped Bugs (`B` -> `S3`) | `stateless_control` (`6` Tasks)<br/>Clean Pass (`B` -> `S3`) |
| :--- | :---: | :---: | :---: |
| **`ollama/gemma4:latest`** | `43.8%` -> **`68.8%`** (`19` -> `7` bugs, **`-63.2%`**) | `0.0%` -> **`37.5%`** (`22` -> `8` bugs, **`-63.6%`**) | `100%` -> **`100%`** (`0` false vetoes) |
| **`anthropic/claude-sonnet-5.5`** | `62.5%` -> **`81.2%`** (`+18.7 pp` clean lift) | `25.0%` -> **`37.5%`** (`+12.5 pp` clean lift) | `100%` -> **`100%`** (`0` false vetoes) |
| **`openai/gpt-5.3-codex`** | `68.8%` -> **`87.5%`** (**`18` -> `0` bugs, `-100.0%`!**) | `37.5%` -> `37.5%` (**`16` -> `10` bugs, `-37.5%`**) | `83.3%` -> **`100%`** (`0` false vetoes) |
| **`x-ai/grok-4.7`** | `75.0%` -> **`93.8%`** (**`19` -> `0` bugs, `-100.0%`!**) | `50.0%` -> `50.0%` (`8` -> `10` bugs) | `100%` -> **`100%`** (`0` false vetoes) |
| **`deepseek/deepseek-v4.1-flash`** | `81.2%` -> `81.2%` (`7` -> `16`* via `T15` `func 0->1`) | `50.0%` -> `37.5%` (`12` -> `14` bugs) | `100%` -> **`100%`** (`0` false vetoes) |

---

## 3. Four High-Signal Takeaways

1. **Frontier IQ Does Not Replace Structural Observability:** Even `anthropic/claude-sonnet-5.5` (`62.5%` clean), `openai/gpt-5.3-codex` (`65.6%` clean, `34` escaped bugs), and `x-ai/grok-4.7` (`75.0%` clean, `27` escaped bugs) ship silent races and half-reset coupled state maps when editing large files through bounded read windows (`T03`, `T05`, `T19`) or cross-language FastAPI<->TypeScript routes (`T21` on `gpt-5.3-codex`). Injecting `<2 ms` S3 sensor telemetry eliminated **`100%` of narrow-window concurrency/coupled-state bugs on both `gpt-5.3-codex` (`18` -> `0`) and `grok-4.7` (`19` -> `0`)**.
2. **Frontier Edit Fidelity Eliminates the 8B Retry Penalty:** Whereas `gemma4:latest` (`8B` Q4) incurred extra `old_string` retries during post-hook repairs (`22` -> `32` wasted turns), `x-ai/grok-4.7` dropped wasted turns **`1` -> `0`** across all 32 tasks (`84.4%` clean pass) and `anthropic/claude-sonnet-5.5` dropped wasted turns **`45` -> `42`** (`100.0%` functional, `75.0%` clean).
3. **Capacitor Router (`s3_routed`) Economics:** Escalating only **`17.1%` (`28/164`)** of turns from `deepseek-v4.1-flash` to `claude-sonnet-5.5` beat unhooked `claude-sonnet-5.5` by **`+9.4 pp` (`71.9%` vs. `62.5%` clean)** while cutting API spend by **`38.3%` (`$0.45` vs. `$0.72`)** and **`54.4%` vs. full `claude-sonnet-5.5 + pi-s3` (`$0.98`)**.
4. **Honest Boundary (`deepseek-v4.1-flash` solo):** Because `deepseek-v4.1-flash` (`$0.05/M` input) eagerly reads broad file spans upfront (`4.97` turns/task), it already observes most single-file locks without Pre-Flight Radar (`78.1%` baseline clean), and hitting Stop-Hook vetoes on turn 5 of a 6-turn cap (`T03`, `T21`) leaves too few turns for a Flash model to finish multi-site repairs without Capacitor Router `slow` escalation.
