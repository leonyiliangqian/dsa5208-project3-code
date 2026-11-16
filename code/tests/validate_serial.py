#!/usr/bin/env python3
"""Serial gate: edge-case self-test, argument rejection, theta=0 agreement with
direct summation at two leaf sizes, and retained theta>0 diagnostics."""
import argparse
import datetime
import hashlib
import json
import platform
import math
from pathlib import Path
import re
import subprocess

from execution_context import execution_context

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--binary", required=True, help="bh_serial executable to test")
parser.add_argument("--out", required=True, help="JSON evidence file to write")
opts = parser.parse_args()
binary = str(Path(opts.binary).resolve())

self_test = subprocess.run([binary, "--self-test"], capture_output=True, text=True)
print(self_test.stdout, end="")
if self_test.returncode != 0 or "self_test failures=0" not in self_test.stdout:
    raise SystemExit("FAIL: edge-case self-test")

bad_args = [["--n", "12abc"], ["--n", "0"], ["--n", "-5"], ["--theta", "-1"],
            ["--theta", "nan"], ["--dt", "0"], ["--steps", "0"], ["--leaf-size", "0"],
            ["--method", "fmm"], ["--bogus"]]
for args in bad_args:
    run = subprocess.run([binary, *args], capture_output=True, text=True)
    if run.returncode != 2:
        raise SystemExit(f"FAIL: {args} exited {run.returncode}, expected 2")
print(f"argument rejection: {len(bad_args)} invalid invocations exit 2")

rows = []
for leaf_size in (1, 8):
    for theta in (0.0, 0.3, 0.5):
        cmd = [binary, "--validate", "--n", "2000", "--theta", str(theta),
               "--leaf-size", str(leaf_size), "--seed", "42"]
        run = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(run.stdout, end="")
        values = dict(re.findall(r"(max_rel_err|mean_rel_err|max_abs_err)=([^\s]+)", run.stdout))
        if len(values) != 3:
            raise SystemExit("FAIL: missing validation diagnostics")
        errors = {key: float(value) for key, value in values.items()}
        if not all(math.isfinite(value) and value >= 0 for value in errors.values()):
            raise SystemExit("FAIL: nonfinite or negative error")
        if theta == 0 and errors["max_rel_err"] > 1e-10:
            raise SystemExit("FAIL: theta=0 max relative error exceeds 1e-10")
        rows.append({"theta": theta, "leaf_size": leaf_size, "n": 2000, "seed": 42, **errors,
                     "stdout": run.stdout})
sources = [root / "serial/barnes_hut.cpp", root / "common/nbody.hpp"]
result = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          **execution_context(),
          "source_sha256": {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
                            for f in sources},
          "platform": platform.platform(), "binary": str(Path(binary).relative_to(root)),
          "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
          "theta_zero_threshold": 1e-10, "self_test_stdout": self_test.stdout,
          "rejected_arguments": bad_args, "measurements": rows}
output = Path(opts.out)
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, indent=2) + "\n")
print("PASS: serial gate (edge cases, arguments, theta=0 vs direct)")
