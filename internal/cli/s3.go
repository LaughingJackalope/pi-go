package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/spf13/cobra"

	"github.com/dimetron/pi-go/internal/sense"
)

func newS3Cmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "s3",
		Short: "S3 neuro-symbolic subcortex: polyglot state-machine audit, blast-radius distillation, and <20ms agent hooks",
		Long: `S3 (System 1 + Bullet-Time System 2) local subcortical control plane.

Provides zero-token static state-coupling & blast-radius sensors across Go, Python,
TypeScript, Java, and Scala fused with the local Apple Silicon Dual-Tier S3 daemon
(s3-cortexd on http://127.0.0.1:8765).`,
	}

	cmd.AddCommand(newS3AuditCmd())
	cmd.AddCommand(newS3BlastCmd())
	cmd.AddCommand(newS3HookCmd())
	return cmd
}

func newS3AuditCmd() *cobra.Command {
	var (
		asJSON        bool
		maxFindings   int
		maxInvariants int
	)

	cmd := &cobra.Command{
		Use:   "audit [path...]",
		Short: "Scan repositories for implicit coupled-state invariants, mutex asymmetries, and state-desync bugs",
		Example: `  pi-s3 s3 audit .
  pi-s3 s3 audit ./bench_repos/kubernetes ./bench_repos/kafka ./bench_repos/hermes-agent
  pi-s3 s3 audit --json ./pkg/kubelet`,
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			engine := sense.NewEngineFromEnv()
			rep, err := engine.AuditRepository(cmd.Context(), args, maxFindings, maxInvariants)
			if err != nil {
				return err
			}
			out := cmd.OutOrStdout()
			if asJSON {
				enc := json.NewEncoder(out)
				enc.SetIndent("", "  ")
				return enc.Encode(rep)
			}
			renderS3AuditText(out, rep)
			return nil
		},
	}

	cmd.Flags().BoolVar(&asJSON, "json", false, "Emit machine-readable JSON report")
	cmd.Flags().IntVar(&maxFindings, "max-findings", 20, "Maximum number of anomaly findings to display")
	cmd.Flags().IntVar(&maxInvariants, "max-invariants", 15, "Maximum number of discovered mutex/coupled invariants to display")
	return cmd
}

func renderS3AuditText(out io.Writer, rep *sense.RepoAuditReport) {
	fmt.Fprintf(out, "[S3 POLYGLOT STATE-MACHINE AUDIT · %.1fms · posture=%s (desync=%.2f act=%.2f s3=%.1fms)]\n",
		rep.ElapsedMs, strings.ToUpper(rep.S3Posture), rep.S3DesyncProb, rep.S3ActProb, rep.S3LatencyMs)
	fmt.Fprintf(out, "Scanned %d files across %d packages/modules (%d structs/classes, %d methods)\n",
		rep.FilesScanned, rep.DirsAnalyzed, rep.UnitsAnalyzed, rep.MethodsAnalyzed)
	fmt.Fprintf(out, "Discovered %d mutex-guarded field sets & %d coupled state-field pairs | %d anomalies flagged\n",
		rep.MutexGuardsCount, rep.CoupledPairsCount, len(rep.Anomalies))

	if len(rep.ByLanguage) > 0 {
		langs := make([]string, 0, len(rep.ByLanguage))
		for k := range rep.ByLanguage {
			langs = append(langs, k)
		}
		sort.Strings(langs)
		fmt.Fprintln(out, "\nLanguage Breakdown:")
		for _, l := range langs {
			st := rep.ByLanguage[l]
			fmt.Fprintf(out, "  • %-10s files=%-4d dirs=%-3d units=%-4d methods=%-5d locks=%-3d coupled_pairs=%-3d anomalies=%d\n",
				l+":", st.FilesScanned, st.DirsAnalyzed, st.UnitsAnalyzed, st.MethodsAnalyzed, st.MutexGuards, st.CoupledStatePairs, st.AnomaliesCount)
		}
	}

	if len(rep.Invariants) > 0 {
		fmt.Fprintln(out, "\nSample Extracted State-Coupling & Lock Invariants:")
		for _, inv := range rep.Invariants {
			if inv.Kind == "COUPLED_STATE_PAIR" && len(inv.Fields) == 2 {
				fmt.Fprintf(out, "  • [%s] %s (%s): %s.%s <-> %s.%s (co-mutated in %s)\n",
					inv.Language, inv.UnitName, inv.RelDir, inv.UnitName, inv.Fields[0], inv.UnitName, inv.Fields[1], strings.Join(inv.Methods, ", "))
			} else if inv.Kind == "MUTEX_GUARD" {
				fmt.Fprintf(out, "  • [%s] %s (%s): lock %q guards %v\n",
					inv.Language, inv.UnitName, inv.RelDir, inv.GuardOrPeer, inv.Fields)
			}
		}
	}

	if len(rep.Anomalies) > 0 {
		fmt.Fprintln(out, "\nImplicit State-Machine Hazards Flagged:")
		for _, f := range rep.Anomalies {
			fmt.Fprintf(out, "  ! [%s · %s] (%s) %s\n", f.Language, f.Anomaly.Kind, f.RelDir, f.Anomaly.Summary)
		}
	}
}

func newS3BlastCmd() *cobra.Command {
	var (
		asJSON     bool
		searchRoot string
		oldStr     string
		newStr     string
		symbols    string
		contracts  string
	)

	cmd := &cobra.Command{
		Use:   "blast <file>",
		Short: "Distill cross-package and cross-language blast radius for a source file or edit",
		Example: `  pi-s3 s3 blast ./internal/guardrail/guardrail.go --symbol SetLastPromptTokens
  pi-s3 s3 blast ./bench_repos/hermes-agent/hermes_cli/dashboard_auth/routes.py --contract /api/auth/me`,
		Args:         cobra.ExactArgs(1),
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			target := args[0]
			engine := sense.NewEngineFromEnv()
			if searchRoot != "" {
				engine.AddSearchRoot(searchRoot)
			}
			scanArgs := map[string]any{
				"file_path":   target,
				"search_root": searchRoot,
				"old_string":  oldStr,
				"new_string":  newStr,
				"symbols":     symbols,
				"contracts":   contracts,
			}
			verdict := engine.EvaluateToolTurn(
				cmd.Context(),
				"edit",
				scanArgs,
				map[string]any{"path": target, "replacements": 1},
				nil,
			)
			out := cmd.OutOrStdout()
			if asJSON {
				enc := json.NewEncoder(out)
				enc.SetIndent("", "  ")
				return enc.Encode(verdict)
			}
			if verdict.BlastRadius != nil && verdict.BlastRadius.Summary != "" {
				fmt.Fprintln(out, verdict.BlastRadius.Summary)
			} else {
				fmt.Fprintf(out, "[S3 BLAST-RADIUS SCANNER · LOCAL] No external callers or wire-contract consumers found for %s\n", filepath.Base(target))
			}
			fmt.Fprintf(out, "\n[S3 CORTEX VERDICT · posture=%s · risk_tier=%s · role=%s · act_prob=%.2f · latency=%.1fms]\n",
				strings.ToUpper(verdict.Posture), verdict.RiskTier, verdict.RecommendedRole, verdict.ActProb, verdict.LatencyMs)
			return nil
		},
	}

	cmd.Flags().BoolVar(&asJSON, "json", false, "Emit full SenseVerdict + BlastRadiusReport as JSON")
	cmd.Flags().StringVar(&searchRoot, "root", "", "Explicit search root directory for cross-file/cross-service sweep")
	cmd.Flags().StringVar(&oldStr, "old", "", "Optional old_string snippet being replaced")
	cmd.Flags().StringVar(&newStr, "new", "", "Optional new_string snippet being introduced")
	cmd.Flags().StringVar(&symbols, "symbol", "", "Comma-separated symbol name(s) to trace across the workspace")
	cmd.Flags().StringVar(&contracts, "contract", "", "Comma-separated wire-contract or API route literal(s) to trace")
	return cmd
}

func newS3HookCmd() *cobra.Command {
	var sessionFile string

	cmd := &cobra.Command{
		Use:   "hook <pre-tool|post-tool>",
		Short: "Sub-20ms PreToolUse / PostToolUse JSON hook for Claude Code, Cursor, or external agent harnesses",
		Example: `  echo '{"tool_name":"Edit","tool_input":{"file_path":"foo.go","old_string":"a","new_string":"b"}}' | pi-s3 s3 hook post-tool`,
		Args:         cobra.ExactArgs(1),
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			phase := strings.ToLower(strings.TrimSpace(args[0]))
			raw, err := io.ReadAll(cmd.InOrStdin())
			if err != nil {
				return err
			}
			var payload map[string]any
			if len(strings.TrimSpace(string(raw))) > 0 {
				if err := json.Unmarshal(raw, &payload); err != nil {
					return fmt.Errorf("invalid hook JSON on stdin: %w", err)
				}
			} else {
				payload = make(map[string]any)
			}

			return runS3ExternalHook(cmd.Context(), cmd.OutOrStdout(), phase, payload, sessionFile)
		},
	}

	cmd.Flags().StringVar(&sessionFile, "state-file", "", "Optional path to persist structural AST cycle hashes across hook calls")
	return cmd
}

func runS3ExternalHook(ctx context.Context, out io.Writer, phase string, payload map[string]any, stateFile string) error {
	rawTool, _ := payload["tool_name"].(string)
	if rawTool == "" {
		rawTool, _ = payload["tool"].(string)
	}
	toolName := normalizeExternalToolName(rawTool)

	toolInput, _ := payload["tool_input"].(map[string]any)
	if toolInput == nil {
		toolInput, _ = payload["args"].(map[string]any)
	}
	if toolInput == nil {
		toolInput = make(map[string]any)
	}

	toolResp, _ := payload["tool_response"].(map[string]any)
	if toolResp == nil {
		toolResp, _ = payload["result"].(map[string]any)
	}
	if toolResp == nil {
		toolResp = make(map[string]any)
	}

	sessionID, _ := payload["session_id"].(string)
	if sessionID == "" {
		sessionID = "default"
	}
	if stateFile == "" {
		safeID := filepath.Base(sessionID)
		stateFile = filepath.Join(os.TempDir(), "pi-s3-hook-"+safeID+".json")
	}

	engine := sense.NewEngineFromEnv()
	loadHookHistory(engine, stateFile)

	resp := map[string]any{"continue": true}

	if phase == "pre-tool" || phase == "pretooluse" {
		filePath := resolveHookFilePath(toolInput, toolResp)
		if filePath != "" && (toolName == "edit" || toolName == "write") {
			if rawBytes, err := os.ReadFile(filePath); err == nil {
				recordHookSnapshot(stateFile, filePath, sense.StructuralSignature(filePath, string(rawBytes)), "pre-edit")
			}
			if toolName == "edit" {
				if locked, summary := sense.ECCMAutoLockEdit(toolInput); locked {
					resp["updatedInput"] = toolInput
					resp["eccm_auto_locked"] = true
					resp["systemMessage"] = "[S3 ECCM · AUTO-LOCK] " + summary
				}
			}
		}
	} else {
		filePath := resolveHookFilePath(toolInput, toolResp)
		if filePath != "" && toolResp["path"] == nil {
			toolResp["path"] = filePath
		}
		verdict := engine.EvaluateToolTurn(ctx, toolName, toolInput, toolResp, nil)
		if filePath != "" && (toolName == "edit" || toolName == "write" || toolName == "read") {
			if rawBytes, err := os.ReadFile(filePath); err == nil {
				if cyc, cycMsg := recordHookSnapshot(stateFile, filePath, sense.StructuralSignature(filePath, string(rawBytes)), toolName); cyc && (toolName == "edit" || toolName == "write") {
					verdict.WarpScramblerEngaged = true
					verdict.BulletTimeEngaged = true
					verdict.Posture = sense.PostureBranch
					verdict.CycleSummary = cycMsg
					if verdict.Advisory == "" {
						verdict.Advisory = cycMsg
					} else if !strings.Contains(verdict.Advisory, cycMsg) {
						verdict.Advisory = cycMsg + "\n" + verdict.Advisory
					}
				}
			}
		}
		resp["s3_verdict"] = verdict
		var ctxBlocks []string
		if verdict.PreFlightRadar != "" {
			ctxBlocks = append(ctxBlocks, verdict.PreFlightRadar)
		}
		if verdict.Advisory != "" {
			ctxBlocks = append(ctxBlocks, verdict.Advisory)
		}
		if len(ctxBlocks) > 0 {
			joined := strings.Join(ctxBlocks, "\n\n")
			resp["additionalContext"] = joined
			resp["hookSpecificOutput"] = map[string]any{
				"hookEventName":     "PostToolUse",
				"additionalContext": joined,
			}
		}
	}

	enc := json.NewEncoder(out)
	return enc.Encode(resp)
}

func normalizeExternalToolName(name string) string {
	switch strings.ToLower(strings.TrimSpace(name)) {
	case "edit", "multiedit", "str_replace_editor", "replace_file_content":
		return "edit"
	case "write", "write_file", "create_file":
		return "write"
	case "read", "read_file", "view_file":
		return "read"
	case "bash", "run_command", "shell":
		return "bash"
	default:
		return strings.ToLower(strings.TrimSpace(name))
	}
}

func resolveHookFilePath(in, res map[string]any) string {
	for _, m := range []map[string]any{in, res} {
		for _, k := range []string{"file_path", "path", "TargetFile"} {
			if s, ok := m[k].(string); ok && s != "" {
				return s
			}
		}
	}
	return ""
}

type hookPersistedState struct {
	History map[string][]sense.StateSnapshot `json:"history"`
}

func loadHookHistory(_ *sense.Engine, _ string) {}

func recordHookSnapshot(stateFile, filePath, sig, label string) (bool, string) {
	if stateFile == "" || filePath == "" || sig == "" {
		return false, ""
	}
	var st hookPersistedState
	if raw, err := os.ReadFile(stateFile); err == nil {
		_ = json.Unmarshal(raw, &st)
	}
	if st.History == nil {
		st.History = make(map[string][]sense.StateSnapshot)
	}
	hist := st.History[filePath]
	if len(hist) > 0 && hist[len(hist)-1].Sig == sig {
		return false, ""
	}
	cycle := false
	msg := ""
	for i := 0; i < len(hist)-1; i++ {
		if hist[i].Sig == sig {
			cycle = true
			msg = fmt.Sprintf("Warp Scrambler: Semantic A->B->A AST cycle detected on %s (reverts to structural hash %s from step #%d)",
				filepath.Base(filePath), sig[:8], hist[i].Step)
			break
		}
	}
	st.History[filePath] = append(hist, sense.StateSnapshot{
		Step:    len(hist) + 1,
		Sig:     sig,
		Summary: label,
	})
	if encoded, err := json.Marshal(st); err == nil {
		_ = os.WriteFile(stateFile, encoded, 0o600)
	}
	return cycle, msg
}
