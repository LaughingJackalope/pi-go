#!/usr/bin/env python3
"""
`export_snake_s3_heads.py` — Extracts M2 ANE `[CLS]` (`pooled` 768-dim) vectors from the
`move` forward pass across all canonical Snake states and fits a zero-extra-ANE-call
linear-logit readout (`snake_aux_head.fp32.bin`) that predicts the `risk` and `food`
`noul` probabilities directly from `move`'s `[CLS]` embedding in <5 microseconds via `vDSP`.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import laya_coreml as laya
import numpy as np
from laya_coreml.snake.game import DIRECTIONS, SnakeGame
from rich.console import Console

DECISION_DIR = Path(__file__).resolve().parent
MODEL_DIR = DECISION_DIR / "models" / "snake"
BUNDLE_DIR = MODEL_DIR / "swift_bundle"

console = Console()


def build_snake_questions(game: SnakeGame, prompt: str = "compact") -> tuple[str, dict]:
    moves = game.moves()
    safe = [m for m in moves if m.safe]
    preferred = max(safe, key=lambda m: m.advance).direction if safe else "NONE"
    reachable, space = game.food_reachability()

    if prompt == "compact":
        state = (
            f"Safe route: {'yes' if safe else 'no'}. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}."
        )
        criteria = {
            m.direction: (
                "Blocked. Collision."
                if not m.legal
                else "Unsafe. Traps the snake."
                if not m.safe
                else "Safe. Eat food now. Best."
                if m.eats
                else "Safe. Best route to food."
                if m.direction == preferred
                else "Safe. Slower route."
            )
            for m in moves
        }
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Choose the best safe move toward food.",
                "criteria": criteria,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is a safe route available?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through empty cells?",
            },
        }
    else:
        descriptions = {}
        for move in moves:
            if not move.legal:
                descriptions[move.direction] = f"Collision: {move.reason}. Unsafe."
            elif not move.safe:
                descriptions[move.direction] = "Unsafe route. Risk of trapping the snake."
            elif move.eats:
                descriptions[move.direction] = "Safe. Eat the food immediately. Best move."
            elif move.direction == preferred:
                descriptions[move.direction] = "Safe. Best progress toward food."
            else:
                descriptions[move.direction] = "Safe but less progress toward food."
        state = (
            f"Snake game. {len(safe)} safe directions available. "
            f"Food reachable through empty cells: {'yes' if reachable else 'no'}. "
            f"Open cells: {space}. Snake length: {len(game.body)}. "
            f"{'There is a safe route forward.' if safe else 'The snake is trapped.'}"
        )
        questions = {
            "move": {
                "type": "choice",
                "instructions": "Select the safest move with best progress toward food. Avoid collisions.",
                "criteria": descriptions,
            },
            "risk": {
                "type": "noul",
                "instructions": "Is there a safe route forward for the snake?",
            },
            "food": {
                "type": "noul",
                "instructions": "Is food reachable through the currently empty cells?",
            },
        }
    return state, questions


def main() -> None:
    t0 = time.perf_counter()
    agent = laya.load(str(MODEL_DIR), local_files_only=True)
    pooled_by_qtype: dict[int, np.ndarray] = {}
    orig_forward = agent.forward

    def tapped_forward(batch):
        outputs = agent.model.predict(agent.model_inputs(batch))
        pooled = next(v for v in outputs.values() if v.shape[1] != 1).reshape(1, -1).astype(np.float32)
        qt = int(batch["qtype"][0])
        if qt not in pooled_by_qtype:
            pooled_by_qtype[qt] = pooled[0].copy()
        return orig_forward(batch)

    agent.forward = tapped_forward

    X_list = []
    Y_prob_list = []

    # Collect real game states across multiple seeds + synthetic edge cases (safe=yes/no, reachable=yes/no)
    game = SnakeGame(24, 16, seed=7)
    for step in range(45):
        for prompt_mode in ("compact", "detailed"):
            state, questions = build_snake_questions(game, prompt=prompt_mode)
            pooled_by_qtype.clear()
            out = agent.predict(state, questions)
            h_cls = pooled_by_qtype[0]  # from `move` (qtype=0)
            p_risk_safe = float(out["answers"]["risk"]["noul"])
            p_food_reach = float(out["answers"]["food"]["noul"])
            X_list.append(h_cls)
            Y_prob_list.append([p_risk_safe, p_food_reach])

        moves = [m for m in game.moves() if m.safe]
        if not moves:
            break
        best = max(moves, key=lambda m: m.advance)
        game.step(best.direction)

    # Also include all 4 combinations of (safe=yes/no, reachable=yes/no) in compact & detailed mode
    _, sample_q = build_snake_questions(game, prompt="compact")
    for safe_flag in ("yes", "no"):
        for reach_flag in ("yes", "no"):
            st_c = f"Safe route: {safe_flag}. Food reachable through empty cells: {reach_flag}."
            pooled_by_qtype.clear()
            out_c = agent.predict(st_c, sample_q)
            X_list.append(pooled_by_qtype[0])
            Y_prob_list.append([float(out_c["answers"]["risk"]["noul"]), float(out_c["answers"]["food"]["noul"])])

    X = np.asarray(X_list, dtype=np.float32)  # (N, 768)
    Y_p = np.clip(np.asarray(Y_prob_list, dtype=np.float32), 1e-4, 1.0 - 1e-4)  # (N, 2)
    Y_logit = np.log(Y_p / (1.0 - Y_p))  # (N, 2)

    # Ridge regression in logit space: [X || 1] @ W_aug = Y_logit
    N, D = X.shape
    X_aug = np.concatenate([X, np.ones((N, 1), dtype=np.float32)], axis=1)  # (N, 769)
    reg = 1e-2 * np.eye(D + 1, dtype=np.float32)
    reg[-1, -1] = 0.0
    W_aug = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y_logit).astype(np.float32)
    W = W_aug[:D, :]  # (768, 2)
    b = W_aug[D, :]   # (2,)

    pred_logits = X @ W + b
    pred_p = 1.0 / (1.0 + np.exp(-pred_logits))
    mae = float(np.mean(np.abs(pred_p - Y_p)))
    max_err = float(np.max(np.abs(pred_p - Y_p)))

    # Save [W[:, 0] (768), W[:, 1] (768), b (2)] as flat Float32 (1538 floats = 6152 bytes)
    out_path = BUNDLE_DIR / "snake_aux_head.fp32.bin"
    with open(out_path, "wb") as fp:
        fp.write(np.ascontiguousarray(W[:, 0], dtype=np.float32).tobytes())
        fp.write(np.ascontiguousarray(W[:, 1], dtype=np.float32).tobytes())
        fp.write(np.ascontiguousarray(b, dtype=np.float32).tobytes())

    dt = (time.perf_counter() - t0) * 1000.0
    console.print(
        f"[bold green]✓ Exported `snake_aux_head.fp32.bin` ({out_path.stat().st_size} B) in {dt:.0f} ms | "
        f"N={N} states | MAE={mae:.5f} | Max |Δp|={max_err:.5f}[/bold green]"
    )


if __name__ == "__main__":
    main()
