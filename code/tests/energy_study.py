#!/usr/bin/env python3
"""Supplementary diagnostic: relative energy drift over a fixed simulated time
T = 0.2 as the timestep shrinks, across opening angles theta.

Leapfrog with exact forces (theta = 0) should show drift falling roughly as
dt^2. If drift at theta > 0 stops falling with dt and instead varies with theta,
that is consistent with a contribution from the approximate tree force; it does
not by itself isolate the cause. Evidence about the integrator/force pairing,
not a correctness proof, and never run inside a benchmark.
"""
import argparse
import datetime
import json
from pathlib import Path
import platform
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--serial", required=True)
parser.add_argument("--out", required=True)
opts = parser.parse_args()

T, N = 0.2, 2000
rows = []
for theta in (0.0, 0.2, 0.3, 0.5, 0.7):
    for dt in (8e-3, 4e-3, 2e-3, 1e-3, 5e-4):
        steps = round(T / dt)
        cmd = [opts.serial, "--n", str(N), "--steps", str(steps), "--dt", repr(dt),
               "--theta", str(theta), "--energy"]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
        rec = json.loads(next(l for l in out.splitlines() if l.startswith("result "))[7:])
        rows.append({"theta": theta, "dt": dt, "steps": steps,
                     "energy_rel_drift": rec["energy_rel_drift"]})
        print(f"energy theta={theta} dt={dt:g} steps={steps} rel_drift={rec['energy_rel_drift']:.3e}")

Path(opts.out).write_text(json.dumps({
    "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "kind": "local supplementary diagnostic; not correctness proof, not cluster evidence",
    "platform": platform.platform(), "n": N, "simulated_time": T, "seed": 42,
    "leaf_size": 8, "rows": rows}, indent=2) + "\n")
