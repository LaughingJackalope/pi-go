#!/usr/bin/env python3
"""
`export_swift_bundle.py` — Exports a zero-copy `mmap()` + pre-compiled `.mlmodelc` bundle
for the native Swift S3 Cortex daemon (`s3-cortexd`).

Produces `/Users/mp/decision/models/snake/swift_bundle/`:
  1. `model.mlmodelc/`         — Pre-compiled Apple Neural Engine graph (via `xcrun coremlcompiler compile`)
  2. `tok_embeddings.fp16.bin` — Page-aligned `[256000, 768]` Float16 token table (375 MB, mapped via `mmap()`)
  3. `type_emb.fp16.bin`       — `[3, 768]` Float16 question-type vectors (`choice=0`, `score=1`, `noul=2`)
  4. `heads.fp32.bin`          — Custom `act_head` MLP + S3 v2 Hyperspherical RMSNorm + Bounded FiLM + SRR head weights
  5. `tokenizer.bin`           — Binary BPE vocabulary (256,000 tokens) + 580,604 merge ranks + added_tokens table
  6. `bundle_config.json`      — Special token IDs, calibration temperatures, and binning maps
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table
from safetensors import safe_open

from laya_coreml.tokenizer import Tokenizer

DECISION_DIR = Path(__file__).resolve().parent
MODEL_DIR = DECISION_DIR / "models" / "snake"
BUNDLE_DIR = MODEL_DIR / "swift_bundle"
CUSTOM_ACT_HEAD = DECISION_DIR / "portal_custom_act_head.npz"
V2_FILM_SRR_HEAD = DECISION_DIR / "portal_s3_v2_film_srr_head.npz"
CALIBRATION_JSON = DECISION_DIR / "portal_calibration.json"

console = Console()


def compile_mlpackage() -> Path:
    mlpackage = MODEL_DIR / "model.mlpackage"
    mlmodelc = BUNDLE_DIR / "model.mlmodelc"
    if mlmodelc.exists():
        shutil.rmtree(mlmodelc)
    t0 = time.perf_counter()
    subprocess.run(
        ["/usr/bin/xcrun", "coremlcompiler", "compile", str(mlpackage), str(BUNDLE_DIR)],
        check=True,
        capture_output=True,
        text=True,
    )
    dt = (time.perf_counter() - t0) * 1000.0
    console.print(f"[green]✓ Compiled `model.mlpackage` -> `model.mlmodelc` in {dt:.1f} ms[/green]")
    return mlmodelc


def export_embeddings_and_heads() -> dict:
    host_st = MODEL_DIR / "host_weights.safetensors"
    with safe_open(str(host_st), framework="numpy") as f:
        tok_emb = f.get_tensor("encoder.embeddings.tok_embeddings.weight")
        type_emb = f.get_tensor("type_emb.weight")
        act_w = {
            k: f.get_tensor("act_head." + k).astype(np.float32)
            for k in ("0.weight", "0.bias", "2.weight", "2.bias")
        }

    if CUSTOM_ACT_HEAD.is_file():
        cw = np.load(CUSTOM_ACT_HEAD)
        for k in ("0.weight", "0.bias", "2.weight", "2.bias"):
            if k in cw:
                act_w[k] = cw[k].astype(np.float32)

    tok_emb_fp16 = np.ascontiguousarray(tok_emb, dtype=np.float16)
    tok_path = BUNDLE_DIR / "tok_embeddings.fp16.bin"
    tok_emb_fp16.tofile(tok_path)

    type_emb_fp16 = np.ascontiguousarray(type_emb, dtype=np.float16)
    type_path = BUNDLE_DIR / "type_emb.fp16.bin"
    type_emb_fp16.tofile(type_path)

    srr = dict(np.load(V2_FILM_SRR_HEAD))
    # Pack all FP32 tensors in a deterministic flat order:
    # 1. act_head: 0.weight (256, 772), 0.bias (256,), 2.weight (2, 256), 2.bias (2,)
    # 2. srr_head: mu_h (768,), sigma_h (768,), W_s (768, 32), b_s (32,),
    #              W_u (4, 32), b_u (32,), W_gamma (32, 32), b_gamma (32,),
    #              W_beta (32, 32), b_beta (32,), W_out (32, 3), b_out (3,), alpha (1,)
    ordered_tensors = [
        ("act_0_weight", act_w["0.weight"], (256, 772)),
        ("act_0_bias", act_w["0.bias"], (256,)),
        ("act_2_weight", act_w["2.weight"], (2, 256)),
        ("act_2_bias", act_w["2.bias"], (2,)),
        ("srr_mu_h", srr["mu_h"].reshape(768), (768,)),
        ("srr_sigma_h", srr["sigma_h"].reshape(768), (768,)),
        ("srr_W_s", srr["W_s"], (768, 32)),
        ("srr_b_s", srr["b_s"].reshape(32), (32,)),
        ("srr_W_u", srr["W_u"], (4, 32)),
        ("srr_b_u", srr["b_u"].reshape(32), (32,)),
        ("srr_W_gamma", srr["W_gamma"], (32, 32)),
        ("srr_b_gamma", srr["b_gamma"].reshape(32), (32,)),
        ("srr_W_beta", srr["W_beta"], (32, 32)),
        ("srr_b_beta", srr["b_beta"].reshape(32), (32,)),
        ("srr_W_out", srr["W_out"], (32, 3)),
        ("srr_b_out", srr["b_out"].reshape(3), (3,)),
        ("srr_alpha", srr["alpha"].reshape(1), (1,)),
    ]

    heads_path = BUNDLE_DIR / "heads.fp32.bin"
    layout = {}
    float_offset = 0
    with open(heads_path, "wb") as fp:
        for name, arr, expected_shape in ordered_tensors:
            assert arr.shape == expected_shape, f"{name}: expected {expected_shape}, got {arr.shape}"
            arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
            fp.write(arr_f32.tobytes())
            layout[name] = {
                "float_offset": float_offset,
                "float_count": int(arr_f32.size),
                "shape": list(expected_shape),
            }
            float_offset += int(arr_f32.size)

    console.print(
        f"[green]✓ Exported `tok_embeddings.fp16.bin` ({tok_path.stat().st_size / (1024*1024):.1f} MB) "
        f"& `heads.fp32.bin` ({heads_path.stat().st_size / 1024:.1f} KB)[/green]"
    )
    return layout


def export_tokenizer() -> dict:
    tok_json_path = MODEL_DIR / "tokenizer" / "tokenizer.json"
    t = json.loads(tok_json_path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = t["model"]["vocab"]
    merges: list[list[str]] = t["model"]["merges"]
    added_tokens: list[dict] = t.get("added_tokens", [])

    non_unused = [a for a in added_tokens if not a["content"].startswith("<unused")]
    console.print(f"[cyan]Tokenizer added_tokens: {len(added_tokens)} total ({len(non_unused)} non-unused)[/cyan]")
    for a in non_unused:
        console.print(
            f"  id={a['id']}: {a['content']!r} (special={a['special']}, lstrip={a['lstrip']}, norm={a['normalized']})"
        )

    vocab_size = len(vocab)
    id_to_piece = [""] * vocab_size
    for piece, idx in vocab.items():
        id_to_piece[idx] = piece
    for a in added_tokens:
        idx = a["id"]
        if 0 <= idx < vocab_size and not id_to_piece[idx]:
            id_to_piece[idx] = a["content"]

    # Convert each merge pair [left_str, right_str] to (left_id, right_id, merged_id, rank)
    merge_quads: list[tuple[int, int, int, int]] = []
    for rank, pair in enumerate(merges):
        if isinstance(pair, str):
            left_s, right_s = pair.split(" ", 1)
        else:
            left_s, right_s = pair[0], pair[1]
        left_id = vocab[left_s]
        right_id = vocab[right_s]
        merged_id = vocab[left_s + right_s]
        merge_quads.append((left_id, right_id, merged_id, rank))

    # Write compact binary tokenizer.bin:
    # Header: magic b"S3TK" (4B), version=2 (u32), vocab_size (u32), num_merges (u32), num_added (u32)
    # Section 1: Vocab pieces: for i in 0..<vocab_size: u16 byte_len + UTF-8 bytes
    # Section 2: Merges: num_merges * (u32 left_id, u32 right_id, u32 merged_id) in rank order
    # Section 3: Added tokens: num_added * (u32 id, u8 flags [bit0=special, bit1=lstrip, bit2=rstrip, bit3=normalized], u16 byte_len, UTF-8 bytes)
    tok_bin_path = BUNDLE_DIR / "tokenizer.bin"
    with open(tok_bin_path, "wb") as fp:
        fp.write(struct.pack("<4sIIII", b"S3TK", 2, vocab_size, len(merge_quads), len(added_tokens)))
        for piece in id_to_piece:
            b = piece.encode("utf-8")
            assert len(b) < 65535
            fp.write(struct.pack("<H", len(b)))
            fp.write(b)
        merge_arr = np.array(
            [(l, r, m) for (l, r, m, _) in merge_quads],
            dtype=np.uint32,
        )
        fp.write(merge_arr.tobytes())
        for a in added_tokens:
            b = a["content"].encode("utf-8")
            flags = (
                (1 if a.get("special") else 0)
                | (2 if a.get("lstrip") else 0)
                | (4 if a.get("rstrip") else 0)
                | (8 if a.get("normalized") else 0)
            )
            fp.write(struct.pack("<IBH", int(a["id"]), flags, len(b)))
            fp.write(b)

    # Build zero-copy mmap tables for Swift (`tokenizer_mmap.bin`):
    # 1. byte_fallback[256] of u32: token ID for `<0x00>` .. `<0xFF>` (or single-byte char / unk=3)
    # 2. char_table[262144] of (u32 codepoint_plus_1, u32 token_id): open-addressed hash table for all 1-char vocab entries
    # 3. merge_table[2097152] of (u64 pair_key, u32 rank, u32 merged_id): open-addressed hash table for all 580,604 merges
    # 4. added_tokens section: u32 count, then sorted by descending byte length: (u32 id, u8 flags, u16 byte_len, bytes)
    byte_fallback = np.zeros(256, dtype=np.uint32)
    for b in range(256):
        tid = vocab.get(f"<0x{b:02X}>")
        if tid is None:
            tid = vocab.get(chr(b), 3)
        byte_fallback[b] = tid

    CHAR_CAP = 1 << 18  # 262,144 slots (power of 2)
    CHAR_MASK = CHAR_CAP - 1
    char_table = np.zeros((CHAR_CAP, 2), dtype=np.uint32)
    single_char_count = 0
    for piece, tid in vocab.items():
        if len(piece) == 1:
            cp = ord(piece)
            key = cp + 1  # > 0 so 0 means empty slot
            h = ((key * 2654435761) & 0xFFFFFFFF) & CHAR_MASK
            while char_table[h, 0] != 0:
                h = (h + 1) & CHAR_MASK
            char_table[h, 0] = key
            char_table[h, 1] = tid
            single_char_count += 1

    MERGE_CAP = 1 << 21  # 2,097,152 slots (load factor 0.277)
    MERGE_MASK = MERGE_CAP - 1
    merge_dtype = np.dtype([("key", "<u8"), ("rank", "<u4"), ("merged_id", "<u4")])
    merge_table = np.zeros(MERGE_CAP, dtype=merge_dtype)
    for left_id, right_id, merged_id, rank in merge_quads:
        key = (int(left_id) << 32) | int(right_id)
        assert key != 0
        # Fibonacci 64-bit hash
        h = (((key * 11400714819323198485) & 0xFFFFFFFFFFFFFFFF) >> 43) & MERGE_MASK
        while merge_table[h]["key"] != 0:
            h = (h + 1) & MERGE_MASK
        merge_table[h]["key"] = key
        merge_table[h]["rank"] = rank
        merge_table[h]["merged_id"] = merged_id

    raw_added_list = sorted(
        [a for a in added_tokens if not a.get("normalized", False) and a["content"]],
        key=lambda a: len(a["content"].encode("utf-8")),
        reverse=True,
    )

    mmap_tok_path = BUNDLE_DIR / "tokenizer_mmap.bin"
    with open(mmap_tok_path, "wb") as fp:
        # 32-byte aligned header: magic b"S3TM", version=1, char_cap, merge_cap, num_added, pad0, pad1, pad2
        fp.write(struct.pack("<4sIIIIIII", b"S3TM", 1, CHAR_CAP, MERGE_CAP, len(raw_added_list), 0, 0, 0))
        fp.write(byte_fallback.tobytes())
        fp.write(char_table.tobytes())
        fp.write(merge_table.tobytes())
        for a in raw_added_list:
            b = a["content"].encode("utf-8")
            flags = (
                (1 if a.get("special") else 0)
                | (2 if a.get("lstrip") else 0)
                | (4 if a.get("rstrip") else 0)
                | (8 if a.get("normalized") else 0)
            )
            fp.write(struct.pack("<IBH", int(a["id"]), flags, len(b)))
            fp.write(b)

    console.print(
        f"[green]✓ Exported zero-copy `tokenizer_mmap.bin` ({mmap_tok_path.stat().st_size / (1024*1024):.2f} MB, "
        f"{single_char_count} 1-char scalars, {len(merge_quads)} O(1) merges, {len(raw_added_list)} added_tokens)[/green]"
    )
    return {
        "vocab_size": vocab_size,
        "num_merges": len(merge_quads),
        "single_char_count": single_char_count,
        "added_tokens": added_tokens,
    }


def verify_bpe_parity() -> None:
    """Verify that our exact BPE algorithm reproduces `laya_coreml.tokenizer.Tokenizer` bit-for-bit."""
    tok = Tokenizer(MODEL_DIR / "tokenizer")
    t = json.loads((MODEL_DIR / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    vocab: dict[str, int] = t["model"]["vocab"]
    merges: list[list[str]] = t["model"]["merges"]
    added_tokens: list[dict] = t.get("added_tokens", [])
    merge_map: dict[tuple[int, int], tuple[int, int]] = {}
    for rank, pair in enumerate(merges):
        left_s, right_s = pair[0], pair[1]
        merge_map[(vocab[left_s], vocab[right_s])] = (rank, vocab[left_s + right_s])

    # Sort non-normalized added tokens by descending length for greedy left-to-right matching
    raw_added = sorted(
        [(a["content"], int(a["id"]), bool(a.get("lstrip", False))) for a in added_tokens if not a.get("normalized", False) and a["content"]],
        key=lambda x: len(x[0]),
        reverse=True,
    )

    def bpe_word(piece: str) -> list[int]:
        if not piece:
            return []
        syms: list[int] = []
        for ch in piece:
            tid = vocab.get(ch)
            if tid is not None:
                syms.append(tid)
            else:
                for b in ch.encode("utf-8"):
                    syms.append(vocab[f"<0x{b:02X}>"])
        while len(syms) >= 2:
            best_rank = 1_000_000_000
            best_idx = -1
            best_merged = -1
            for i in range(len(syms) - 1):
                info = merge_map.get((syms[i], syms[i + 1]))
                if info is not None and info[0] < best_rank:
                    best_rank = info[0]
                    best_idx = i
                    best_merged = info[1]
            if best_idx < 0:
                break
            syms[best_idx] = best_merged
            del syms[best_idx + 1]
        return syms

    def encode_segment(seg: str) -> list[int]:
        if not seg:
            return []
        norm = seg.replace(" ", "▁")
        if not norm.startswith("▁"):
            norm = "▁" + norm
        pieces: list[str] = []
        cur_start = 0
        for i in range(1, len(norm)):
            if norm[i] == "▁":
                pieces.append(norm[cur_start:i])
                cur_start = i
        pieces.append(norm[cur_start:])
        out: list[int] = []
        for p in pieces:
            out.extend(bpe_word(p))
        return out

    def pure_encode(text: str) -> list[int]:
        if not text:
            return []
        out: list[int] = []
        n = len(text)
        i = 0
        seg_start = 0
        while i < n:
            matched = None
            for pat, tid, lstrip in raw_added:
                if text.startswith(pat, i):
                    matched = (pat, tid, lstrip)
                    break
            if matched is not None:
                pat, tid, lstrip = matched
                seg = text[seg_start:i]
                if lstrip:
                    seg = seg.rstrip(" \t\r\n")
                if seg:
                    out.extend(encode_segment(seg))
                out.append(tid)
                i += len(pat)
                seg_start = i
            else:
                i += 1
        if seg_start < n:
            out.extend(encode_segment(text[seg_start:]))
        return out

    test_strings = [
        "",
        " ",
        "  ",
        "a",
        " a",
        "  a",
        "a b",
        "a  b",
        "a\nb",
        "a\tb",
        "warmup",
        "choice question: Select the single best immediate engineering action given this telemetry.",
        "noul question: Does this code change introduce a state-synchronization or concurrency risk?",
        "noul question: Is this turn worth retaining in long-term episodic memory?",
        " branch: isolate in a git branch or run a broader test suite before proceeding",
        " inspect: read or inspect relevant source files first before mutating state",
        " proceed: proceed directly with the current tool execution or code edit",
        " verify: run targeted verification or unit tests immediately after this step",
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
        '{"phase":"POST_TOOL_EDIT","tool":"edit","target":"taskbot.py","tier":"TIER2_HIGH_BLAST_OR_DYNAMIC","blast_radius":{"touched_symbols":["close"],"cross_file_callers":4}}',
        '{"phase":"PRE_MODEL_ TURN","recent_user_intent":"Fix the race condition in taskbot.py without breaking runner.py"}',
        "func (e *Engine) AssessPreTool(ctx context.Context, toolName string, args map[string]any) Verdict",
    ]
    for s in test_strings:
        ref = tok(s, add_special_tokens=False)["input_ids"]
        got = pure_encode(s)
        if ref != got:
            console.print(f"[red]MISMATCH on {s!r}:[/red]\n  ref={ref}\n  got={got}")
            raise AssertionError(f"Tokenizer parity failed on {s!r}")
    console.print(f"[bold green]✓ 100% Bit-Exact Tokenizer Parity Verified across {len(test_strings)} test cases![/bold green]")


def main() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    compile_mlpackage()
    heads_layout = export_embeddings_and_heads()
    tok_meta = export_tokenizer()
    verify_bpe_parity()

    cal = json.loads(CALIBRATION_JSON.read_text(encoding="utf-8")) if CALIBRATION_JSON.is_file() else {}
    bundle_cfg = {
        "format": "s3-cortex-swift-bundle-v1",
        "max_length": 96,
        "head_max_len": 192,
        "max_options": 32,
        "hidden_size": 768,
        "local_attention_window": 64,
        "special_tokens": {
            "pad_id": 0,
            "sep_id": 1,
            "cls_id": 2,
            "unk_id": 3,
            "mask_id": 4,
            "mask_token": "<mask>",
        },
        "temperature": cal.get("temperature", [1.0, 1.0, 1.0]),
        "temperature_by_options": cal.get("temperature_by_options", {}),
        "binning_map": cal.get("binning_map", {}),
        "heads_layout": heads_layout,
        "vocab_size": tok_meta["vocab_size"],
        "num_merges": tok_meta["num_merges"],
    }
    (BUNDLE_DIR / "bundle_config.json").write_text(json.dumps(bundle_cfg, indent=2), encoding="utf-8")
    console.print("[bold green]✓ Wrote `bundle_config.json`[/bold green]")


if __name__ == "__main__":
    main()
