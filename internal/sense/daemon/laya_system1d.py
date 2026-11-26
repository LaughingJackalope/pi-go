#!/usr/bin/env python3
"""
`laya_system1d.py` — Always-On Apple M2 Neural Engine Tactical Subsystem Daemon
for `dimetron/pi-go` (`internal/sense`) and `portal` (`model-gateway` `/v1/decisions`).

Default endpoint: http://127.0.0.1:8765
  - GET  /healthz       -> Live daemon telemetry (calls served, median ANE latency, PID, uptime)
  - POST /v1/systemone  -> Zero-token System 1 multi-head decision pass (~20 ms on M2 ANE)
  - POST /v1/decisions  -> Alias compatible with `portal` `apps/model-gateway`

Usage:
  python3 laya_system1d.py --start       # Start background daemon (idempotent) & wait for healthz
  python3 laya_system1d.py --status      # Query running daemon health & latency telemetry
  python3 laya_system1d.py --stop        # Stop background daemon
  python3 laya_system1d.py --foreground  # Run in foreground
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

DECISION_DIR = Path(__file__).resolve().parent
ANE_MODEL_DIR = Path(os.environ.get("LAYA_ANE_MODEL_DIR", "/Users/mp/decision/models/snake"))
if not ANE_MODEL_DIR.is_dir():
    ANE_MODEL_DIR = DECISION_DIR / "models" / "snake"
CUSTOM_ACT_HEAD = DECISION_DIR / "portal_custom_act_head.npz"
V2_FILM_SRR_HEAD = DECISION_DIR / "portal_s3_v2_film_srr_head.npz"
CALIBRATION_JSON = DECISION_DIR / "portal_calibration.json"
PID_FILE = DECISION_DIR / ".laya_system1d.pid"
LOG_FILE = DECISION_DIR / ".laya_system1d.log"

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

console = Console()


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def query_health(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, timeout: float = 1.0) -> dict | None:
    url = f"http://{host}:{port}/healthz"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    return None


def run_foreground(host: str, port: int) -> None:
    import laya_coreml as laya

    t0 = time.perf_counter()
    agent = laya.load(str(ANE_MODEL_DIR), local_files_only=True)
    has_custom_act = False
    if CUSTOM_ACT_HEAD.is_file() and hasattr(agent, "action"):
        w = np.load(CUSTOM_ACT_HEAD)
        for k in ("0.weight", "0.bias", "2.weight", "2.bias"):
            if k in w:
                agent.action[k] = w[k].astype(np.float32)
        has_custom_act = True

    srr_weights = None
    if V2_FILM_SRR_HEAD.is_file():
        srr_weights = dict(np.load(V2_FILM_SRR_HEAD))

    # Install zero-overhead tap on agent.forward keyed by qtype (0=choice, 1=score, 2=noul)
    agent._action_inputs_by_qtype = {}

    def tapped_forward(batch):
        outputs = agent.model.predict(agent.model_inputs(batch))
        logits = next(v for v in outputs.values() if v.shape[1] == 1).reshape(1, 32).astype(np.float32)
        pooled = next(v for v in outputs.values() if v.shape[1] != 1).reshape(1, -1).astype(np.float32)
        logits = np.where(batch["marker_mask"].astype(bool), logits, -1e4)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        k = np.maximum(batch["marker_mask"].sum(axis=-1), 2).astype(np.float32)
        entropy = -(p * np.log(np.maximum(p, 1e-9))).sum(axis=-1) / np.log(k)
        top = np.sort(p, axis=-1)[:, -2:]
        features = np.stack((top[:, 1], top[:, 1] - top[:, 0], entropy, k / 255.0), axis=-1)
        action_input = np.concatenate((pooled, features), axis=-1)
        qt = int(batch["qtype"][0])
        agent._action_inputs_by_qtype[qt] = action_input.copy()
        hidden = action_input @ agent.action["0.weight"].T + agent.action["0.bias"]
        hidden = hidden * (1 + agent._erf(hidden / np.sqrt(2)).astype(np.float32)) / 2
        action = hidden @ agent.action["2.weight"].T + agent.action["2.bias"]
        return logits, action

    agent.forward = tapped_forward

    has_calibration = False
    if CALIBRATION_JSON.is_file() and hasattr(agent, "load_calibration"):
        try:
            agent.load_calibration(str(CALIBRATION_JSON))
            has_calibration = True
        except Exception:
            pass

    agent.predict("warmup", {"q": {"type": "noul", "instructions": "Warmup?"}})
    warmup_ms = (time.perf_counter() - t0) * 1000
    started_at = time.time()

    lock = threading.Lock()
    latencies: list[float] = []
    calls_served = 0

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_json(self, code: int, payload: dict) -> None:
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/healthz":
                with lock:
                    med = float(np.median(latencies)) if latencies else 0.0
                    p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
                    cnt = calls_served
                return self._send_json(
                    200,
                    {
                        "status": "ok",
                        "service": "laya-system1d",
                        "endpoint": f"http://{host}:{port}",
                        "model": "laya-multilingual-coreml-ane (mmBERT-base 322M FP16 ANE)",
                        "custom_act_head": has_custom_act,
                        "s3_v2_film_srr_head": srr_weights is not None,
                        "calibrated": has_calibration,
                        "warmup_ms": round(warmup_ms, 1),
                        "calls_served": cnt,
                        "median_latency_ms": round(med, 2),
                        "p95_latency_ms": round(p95, 2),
                        "uptime_sec": round(time.time() - started_at, 1),
                        "pid": os.getpid(),
                    },
                )
            self._send_json(404, {"error": "not found"})

        def do_POST(self):
            nonlocal calls_served
            if self.path not in ("/v1/systemone", "/v1/decisions"):
                return self._send_json(404, {"error": f"unsupported path {self.path}"})
            t_req = time.perf_counter()
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            state = body.get("state", "")
            questions = body.get("questions", {})
            use_runoff = bool(body.get("runoff", False))
            with lock:
                agent._action_inputs_by_qtype.clear()
                out = agent.predict(state, questions)
                first_by_qtype = dict(agent._action_inputs_by_qtype)
                # Upgrade 3: "Consider, Then Choose" Cyclic Runoff when top-2 choice margin < 0.18
                for qid, qspec in questions.items():
                    if qspec.get("type") == "choice" and qid in out.get("answers", {}):
                        ans = out["answers"][qid]
                        probs = ans.get("probabilities", {})
                        vals = sorted(probs.values(), reverse=True)
                        margin = (vals[0] - vals[1]) if len(vals) >= 2 else 1.0
                        if (use_runoff or margin < 0.18) and len(probs) >= 3:
                            items = list(qspec.get("criteria", {}).items())
                            rotated = dict(items[1:] + items[:1])
                            q_rot = {**qspec, "criteria": rotated}
                            p_rot = agent.predict(state, {qid: q_rot})["answers"][qid]["probabilities"]
                            avg_p = {k: round((float(probs[k]) + float(p_rot[k])) / 2.0, 4) for k in probs}
                            ans["probabilities"] = avg_p
                            ans["choice"] = max(avg_p, key=avg_p.get)
                            ans["cyclic_runoff_applied"] = True
                    # Upgrade 1 & 2: Evaluate Bounded FiLM + Signed Rescue Routing (SRR) head at 0 extra ANE calls
                    qt_id = {"choice": 0, "score": 1, "noul": 2}.get(qspec.get("type", "choice"), 0)
                    if srr_weights is not None and qt_id in first_by_qtype and qid in out.get("answers", {}):
                        vec772 = first_by_qtype[qt_id]
                        pooled = vec772[:, :768]
                        u = vec772[:, 768:]
                        alpha = float(srr_weights["alpha"][0])
                        hc = (pooled - srr_weights["mu_h"]) / srr_weights["sigma_h"]
                        rms = np.sqrt(np.mean(hc * hc, axis=-1, keepdims=True) + 1e-6)
                        hn = (hc / rms).astype(np.float32)
                        z_s = np.tanh(hn @ srr_weights["W_s"] + srr_weights["b_s"])
                        z_u = np.tanh(u @ srr_weights["W_u"] + srr_weights["b_u"])
                        gamma = alpha * np.tanh(z_s @ srr_weights["W_gamma"] + srr_weights["b_gamma"])
                        beta = alpha * np.tanh(z_s @ srr_weights["W_beta"] + srr_weights["b_beta"])
                        f = (1.0 + gamma) * z_u + beta
                        srr_p = 1.0 / (1.0 + np.exp(-np.clip(f @ srr_weights["W_out"] + srr_weights["b_out"], -30.0, 30.0)))
                        r_hat, h_hat, a_hat = float(srr_p[0, 0]), float(srr_p[0, 1]), float(srr_p[0, 2])
                        gain = r_hat - h_hat - 0.12
                        act_p = float(np.clip((1.0 - a_hat) * (1.0 / (1.0 + np.exp(4.0 * gain))), 0.0, 1.0))
                        out["answers"][qid]["action"] = {
                            "act_probability": round(act_p, 4),
                            "rescue_prob": round(r_hat, 4),
                            "harm_prob": round(h_hat, 4),
                            "ambig_prob": round(a_hat, 4),
                            "signed_gain": round(gain, 4),
                        }
                ms = (time.perf_counter() - t_req) * 1000
                calls_served += 1
                latencies.append(ms)
                if len(latencies) > 500:
                    del latencies[:250]
            out["ane_latency_ms"] = round(ms, 2)
            self._send_json(200, out)

    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    srv = ThreadedHTTPServer((host, port), Handler)

    def _shutdown(signum, frame):
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    print(
        f"[laya-system1d] Online at http://{host}:{port}/v1/systemone "
        f"(PID {os.getpid()}, warmup {warmup_ms:.0f} ms, custom_act_head={has_custom_act})",
        flush=True,
    )
    srv.serve_forever()


def print_status(info: dict) -> None:
    t = Table(title="S3 Tactical Subsystem Daemon (`laya-system1d`) · Live M2 ANE Telemetry")
    t.add_column("Property", style="cyan")
    t.add_column("Value", style="bold green")
    t.add_row("Status", f"ONLINE (PID {info['pid']})")
    t.add_row("Endpoint", f"{info['endpoint']}/v1/systemone (and /v1/decisions)")
    t.add_row("ANE Model", info["model"])
    t.add_row("Custom `act_head` Loaded", str(info["custom_act_head"]))
    t.add_row("Pure-NumPy Calibration Loaded", str(info["calibrated"]))
    t.add_row("Calls Served", str(info["calls_served"]))
    t.add_row("Median ANE Latency", f"{info['median_latency_ms']} ms")
    t.add_row("P95 ANE Latency", f"{info['p95_latency_ms']} ms")
    t.add_row("Uptime", f"{info['uptime_sec']} s")
    console.print(t)


def start_daemon(host: str, port: int) -> None:
    existing = query_health(host, port)
    if existing:
        console.print("[bold green]S3 Tactical Subsystem Daemon is already online![/bold green]")
        print_status(existing)
        return

    log_fp = open(LOG_FILE, "a", encoding="utf-8")
    proc = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--foreground", "--host", host, "--port", str(port)],
        stdout=log_fp,
        stderr=log_fp,
        start_new_session=True,
    )
    console.print(f"Booting `laya-system1d` on M2 Neural Engine (PID {proc.pid})...")
    deadline = time.time() + 45.0
    while time.time() < deadline:
        info = query_health(host, port, timeout=0.5)
        if info:
            print_status(info)
            return
        if proc.poll() is not None:
            console.print(f"[bold red]Daemon exited unexpectedly with code {proc.returncode}[/bold red]")
            console.print(LOG_FILE.read_text(encoding="utf-8")[-1000:])
            sys.exit(1)
        time.sleep(0.5)
    console.print("[bold red]Timed out waiting for laya-system1d /healthz[/bold red]")
    sys.exit(1)


def stop_daemon(host: str, port: int) -> None:
    info = query_health(host, port)
    pid = info.get("pid") if info else None
    if not pid and PID_FILE.is_file():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
    if not pid:
        console.print("[yellow]No running laya-system1d daemon found.[/yellow]")
        return
    try:
        os.kill(pid, signal.SIGTERM)
        console.print(f"[green]Stopped laya-system1d (PID {pid}).[/green]")
    except ProcessLookupError:
        console.print(f"[yellow]PID {pid} was not running.[/yellow]")
    PID_FILE.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Laya M2 ANE Tactical Subsystem Daemon")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--start", "--daemon", action="store_true", help="Start background daemon")
    group.add_argument("--stop", action="store_true", help="Stop background daemon")
    group.add_argument("--status", action="store_true", help="Show running daemon status")
    group.add_argument("--foreground", action="store_true", help="Run server in foreground")
    args = parser.parse_args()

    if args.stop:
        stop_daemon(args.host, args.port)
    elif args.status:
        info = query_health(args.host, args.port)
        if not info:
            console.print("[red]laya-system1d is OFFLINE.[/red]")
            sys.exit(1)
        print_status(info)
    elif args.start:
        start_daemon(args.host, args.port)
    else:
        run_foreground(args.host, args.port)


if __name__ == "__main__":
    main()
