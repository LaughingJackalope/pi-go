#!/usr/bin/env python3
"""
`export_l512_swift_bundle.py` — Exports the Tier-1B Deep Cortex (`answerdotai/ModernBERT-large`,
420M parameters, Enumerated Shapes L in [16, 32, 64, 96, 128, 192, 256, 384, 512])
into `/Users/mp/decision/models/modernbert_large_l512/swift_bundle/`:
  1. `model.mlmodelc/`         — Pre-compiled CoreML graph (via `xcrun coremlcompiler compile`)
  2. `tokenizer_mmap.bin`      — Zero-copy `mmap()` ByteLevel BPE table (`byte_to_id[256]` + `merge_table[262144]` + `added_tokens`)
  3. `bundle_config.json`      — Special token IDs (`[CLS]=50281`, `[SEP]=50282`, `[PAD]=50283`, `[MASK]=50284`),
                                 enumerated lengths, and calibrated temperatures
"""

from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from laya_coreml.artifacts import package_for_coreml
from laya_coreml.hub import resolve_checkpoint
from laya_coreml.tokenizer import Tokenizer
from rich.console import Console

DECISION_DIR = Path(__file__).resolve().parent
OUT_MODEL_DIR = DECISION_DIR / "models" / "modernbert_large_l512"
BUNDLE_DIR = OUT_MODEL_DIR / "swift_bundle"

console = Console()


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def compile_mlpackage(src_dir: Path) -> Path:
    mlpackage = package_for_coreml(src_dir / "model.mlpackage")
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
    console.print(f"[green]✓ Compiled `ModernBERT-large` `model.mlpackage` -> `model.mlmodelc` in {dt:.1f} ms[/green]")
    return mlmodelc


def export_bytelevel_tokenizer(src_dir: Path) -> dict:
    tok_json = json.loads((src_dir / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
    vocab: dict[str, int] = tok_json["model"]["vocab"]
    merges: list[list[str] | str] = tok_json["model"]["merges"]
    added_tokens: list[dict] = tok_json.get("added_tokens", [])

    for a in added_tokens:
        vocab[a["content"]] = int(a["id"])

    b2u = gpt2_bytes_to_unicode()
    byte_to_id = np.zeros(256, dtype=np.uint32)
    for b in range(256):
        ch = b2u[b]
        byte_to_id[b] = vocab.get(ch, 50280)

    merge_quads: list[tuple[int, int, int, int]] = []
    for rank, pair in enumerate(merges):
        if isinstance(pair, str):
            l_s, r_s = pair.split(" ", 1)
        else:
            l_s, r_s = pair[0], pair[1]
        l_id = vocab[l_s]
        r_id = vocab[r_s]
        m_id = vocab[l_s + r_s]
        merge_quads.append((l_id, r_id, m_id, rank))

    MERGE_CAP = 1 << 18  # 262,144 slots for 50,009 merges (load factor 0.19)
    MERGE_MASK = MERGE_CAP - 1
    merge_dtype = np.dtype([("key", "<u8"), ("rank", "<u4"), ("merged_id", "<u4")])
    merge_table = np.zeros(MERGE_CAP, dtype=merge_dtype)
    for l_id, r_id, m_id, rank in merge_quads:
        key = (int(l_id) << 32) | int(r_id)
        h = (((key * 11400714819323198485) & 0xFFFFFFFFFFFFFFFF) >> 46) & MERGE_MASK
        while merge_table[h]["key"] != 0:
            h = (h + 1) & MERGE_MASK
        merge_table[h]["key"] = key
        merge_table[h]["rank"] = rank
        merge_table[h]["merged_id"] = m_id

    # Inspect added_tokens
    console.print(f"[cyan]ModernBERT-large added_tokens: {len(added_tokens)}[/cyan]")
    # Sort added_tokens by descending byte length
    sorted_added = sorted(
        [a for a in added_tokens if a["content"]],
        key=lambda a: len(a["content"].encode("utf-8")),
        reverse=True,
    )

    mmap_path = BUNDLE_DIR / "tokenizer_mmap.bin"
    with open(mmap_path, "wb") as fp:
        # 32-byte header: magic b"S3BL" (ByteLevel BPE), version=1, merge_cap, num_added, cls_id, sep_id, pad_id, mask_id
        fp.write(
            struct.pack(
                "<4sIIIIIII",
                b"S3BL",
                1,
                MERGE_CAP,
                len(sorted_added),
                50281,  # [CLS]
                50282,  # [SEP]
                50283,  # [PAD]
                50284,  # [MASK]
            )
        )
        fp.write(byte_to_id.tobytes())
        fp.write(merge_table.tobytes())
        for a in sorted_added:
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
        f"[green]✓ Exported `ModernBERT-large` `tokenizer_mmap.bin` ({mmap_path.stat().st_size / (1024*1024):.2f} MB, "
        f"{len(merge_quads)} O(1) merges, {len(sorted_added)} added_tokens)[/green]"
    )
    return {
        "vocab_size": len(vocab),
        "num_merges": len(merge_quads),
        "byte_to_id": byte_to_id,
        "merge_quads": merge_quads,
        "sorted_added": sorted_added,
    }


def verify_bytelevel_parity(src_dir: Path, meta: dict) -> None:
    tok = Tokenizer(src_dir / "tokenizer")
    byte_to_id = meta["byte_to_id"]
    merge_map = {(l, r): (rank, m) for (l, r, m, rank) in meta["merge_quads"]}
    sorted_added = meta["sorted_added"]

    # GPT-2 ByteLevel regex used by HuggingFace `tokenizers` ByteLevel(use_regex=True)
    pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-zÀ-ÖØ-öø-ÿ]+| ?[0-9]+| ?[^\sA-Za-zÀ-ÖØ-öø-ÿ0-9]+|\s+(?!\S)|\s+"""
    )

    def bpe_bytes(raw_bytes: bytes) -> list[int]:
        if not raw_bytes:
            return []
        syms = [int(byte_to_id[b]) for b in raw_bytes]
        while len(syms) >= 2:
            best_rank = 1_000_000_000
            best_idx = -1
            best_m = -1
            for i in range(len(syms) - 1):
                info = merge_map.get((syms[i], syms[i + 1]))
                if info is not None and info[0] < best_rank:
                    best_rank = info[0]
                    best_idx = i
                    best_m = info[1]
            if best_idx < 0:
                break
            syms[best_idx] = best_m
            del syms[best_idx + 1]
        return syms

    def encode_segment(seg: str) -> list[int]:
        if not seg:
            return []
        out: list[int] = []
        for m in pat.finditer(seg):
            piece = m.group(0)
            out.extend(bpe_bytes(piece.encode("utf-8")))
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
            for a in sorted_added:
                ac = a["content"]
                # Note: in ModernBERT tokenizer, multi-space added_tokens (50254..50276) only match
                # when a run of spaces is NOT followed by a word char (because ByteLevel attaches the last space to the next word!)
                if text.startswith(ac, i):
                    if ac[0] == " " and len(ac) >= 2:
                        after_idx = i + len(ac)
                        if after_idx < n and text[after_idx] != " ":
                            # Wait: if there are K >= 2 spaces followed by non-space,
                            # the first K-1 spaces match added_token of length K-1 (if >=2),
                            # and the last space attaches to the word!
                            continue
                    matched = a
                    break
            if matched is not None:
                if i > seg_start:
                    out.extend(encode_segment(text[seg_start:i]))
                out.append(int(matched["id"]))
                i += len(matched["content"])
                seg_start = i
            else:
                i += 1
        if seg_start < n:
            out.extend(encode_segment(text[seg_start:]))
        return out

    test_strings = [
        "",
        "warmup",
        "choice question: Select the single best immediate engineering action given this telemetry.",
        "noul question: Does this code change introduce a state-synchronization or concurrency risk?",
        " branch: isolate in a git branch or run a broader test suite before proceeding",
        " inspect: read or inspect relevant source files first before mutating state",
        " proceed: proceed directly with the current tool execution or code edit",
        " verify: run targeted verification or unit tests immediately after this step",
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
        '{"phase":"POST_TOOL_EDIT","tool":"edit","target":"taskbot.py","tier":"TIER2_HIGH_BLAST_OR_DYNAMIC"}',
        "File: apps/taskbot/files/taskbot.py. Class: Bot. CRITICAL TAIL VERDICT: FATAL UNLOCKED_STATE_WRITE at line 306!",
    ]
    for s in test_strings:
        ref = tok(s, add_special_tokens=False)["input_ids"]
        got = pure_encode(s)
        if ref != got:
            console.print(f"[red]MISMATCH on {s!r}:[/red]\n  ref={ref}\n  got={got}")
            raise AssertionError(f"ByteLevel BPE parity failed on {s!r}")
    console.print(f"[bold green]✓ 100% Bit-Exact `ModernBERT-large` ByteLevel BPE Parity Verified across {len(test_strings)} test cases![/bold green]")


def main() -> None:
    src_dir = resolve_checkpoint("aac6fef/laya-coreml", local_files_only=True)
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    compile_mlpackage(src_dir)
    meta = export_bytelevel_tokenizer(src_dir)
    verify_bytelevel_parity(src_dir, meta)

    rl_cfg = json.loads((src_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    cml_cfg = json.loads((src_dir / "coreml_config.json").read_text(encoding="utf-8"))
    bundle_cfg = {
        "format": "s3-cortex-l512-swift-bundle-v1",
        "model": "convaiinnovations/laya (answerdotai/ModernBERT-large 420M FP16 L512)",
        "max_length": 512,
        "head_max_len": 192,
        "max_options": 32,
        "enumerated_lengths": cml_cfg["shape"]["lengths"],
        "special_tokens": {
            "unk_id": 50280,
            "cls_id": 50281,
            "sep_id": 50282,
            "pad_id": 50283,
            "mask_id": 50284,
            "mask_token": "[MASK]",
        },
        "temperature": rl_cfg.get("temperature", [1.6369, 1.2514, 1.9834]),
        "temperature_by_options": rl_cfg.get("temperature_by_options", {}),
    }
    (BUNDLE_DIR / "bundle_config.json").write_text(json.dumps(bundle_cfg, indent=2), encoding="utf-8")
    console.print("[bold green]✓ Wrote `models/modernbert_large_l512/swift_bundle/bundle_config.json`[/bold green]")


if __name__ == "__main__":
    main()
