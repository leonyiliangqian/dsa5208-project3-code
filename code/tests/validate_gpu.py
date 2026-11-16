#!/usr/bin/env python3
"""Gate for the OpenMP target solver.

Each check answers a question about the kernel or where it ran:
  device      Did the kernel run where the record says it ran?
  self-test   Does each backend match direct summation (theta=0), the host tree
              (theta>0) and a closed form, on inputs that stress the tree?
  validate    Does a larger clustered input give the same forces as the host tree?
  trajectory  Does a multi-step run end where the serial solver ends?
  rejection   Are bad arguments refused?
  mutation    (--expect-no-device only) Do the checks catch deliberately broken
              kernels? A check that cannot fail shows nothing.

With --expect-no-device the target regions run on the host, so this says nothing
about a GPU, and the record says so. Errors that only exist with separate device
memory, such as a missing transfer, can only show up with --expect-device.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from execution_context import execution_context

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--serial", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--source", required=True)
ap.add_argument("--cxx", required=True)
ap.add_argument("--cxxflags", required=True)
ap.add_argument("--out", required=True)
mode = ap.add_mutually_exclusive_group(required=True)
mode.add_argument("--expect-device", action="store_true")
mode.add_argument("--expect-no-device", action="store_true")
a = ap.parse_args()

serial, gpu = str(Path(a.serial).resolve()), str(Path(a.gpu).resolve())
checks = []


def run(cmd, **env):
    return subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, **env})


def check(name, ok, detail=""):
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}")


# ------------------------------------------------------------------- device
probe = run([gpu, "--device-check"])
m = re.search(r"num_devices=(\d+) kernel_on_device=(\d) separate_memory=(\d)", probe.stdout)
info = dict(zip(("num_devices", "kernel_on_device", "separate_memory"), map(int, m.groups()))) if m else {}
check("device: probe reports", probe.returncode == 0 and m, probe.stdout.strip())
strict = run([gpu, "--require-device", "--device-check"])
forced = run([gpu, "--device-check"], OMP_TARGET_OFFLOAD="MANDATORY")
if a.expect_device:
    check("device: kernel on device with separate memory",
          info.get("num_devices", 0) >= 1 and info.get("kernel_on_device") == 1 and info.get("separate_memory") == 1)
    check("device: --require-device accepts", strict.returncode == 0, f"rc={strict.returncode}")
    check("device: OMP_TARGET_OFFLOAD=MANDATORY runs", forced.returncode == 0, f"rc={forced.returncode}")
else:
    check("device: host fallback reported, not a device",
          info.get("kernel_on_device") == 0 and info.get("separate_memory") == 0, str(info))
    check("device: --require-device refuses (exit 7)", strict.returncode == 7, f"rc={strict.returncode}")
    # Recorded, not required: GCC 12 without offload support returns success here,
    # so this variable alone cannot prove offload, and --require-device does not rely on it.
    print(f"INFO OMP_TARGET_OFFLOAD=MANDATORY without a device: rc={forced.returncode}")
    mandatory_without_device_rc = forced.returncode

# ---------------------------------------------------------------- self-test
for backend in ("device", "cpu"):
    r = run([gpu, "--self-test", "--backend", backend], OMP_NUM_THREADS="4")
    print(r.stdout, end="")
    n_cases = len(re.findall(r"^self_test case=", r.stdout, re.M))
    check(f"self-test {backend}", r.returncode == 0 and "self_test failures=0" in r.stdout, f"{n_cases} cases")

# ----------------------------------------------------------------- validate
for backend in ("device", "cpu"):
    for theta, leaf in ((0.0, 8), (0.5, 8), (0.5, 1), (0.9, 16)):
        r = run([gpu, "--validate", "--backend", backend, "--n", "20000", "--theta", str(theta),
                 "--leaf-size", str(leaf)], OMP_NUM_THREADS="4")
        line = next((x for x in r.stdout.splitlines() if x.startswith("validate_gpu")), "")
        check(f"validate {backend} theta={theta} leaf={leaf}", r.returncode == 0 and line.endswith("PASS"), line)

# --------------------------------------------------------------- trajectory
with tempfile.TemporaryDirectory() as tmp:
    args = ["--n", "5000", "--steps", "20", "--theta", "0.5", "--leaf-size", "8", "--seed", "5"]
    want = Path(tmp) / "serial.txt"
    r0 = run([serial, *args, "--dump", str(want)])
    rows0 = [list(map(float, x.split())) for x in want.read_text().splitlines()[1:]] if r0.returncode == 0 else []
    for backend in ("device", "cpu"):
        got = Path(tmp) / f"{backend}.txt"
        r = run([gpu, *args, "--backend", backend, "--dump", str(got)], OMP_NUM_THREADS="4")
        rows = [list(map(float, x.split())) for x in got.read_text().splitlines()[1:]] if r.returncode == 0 else []
        # Fixed in advance: 20 steps of tiny per-step rounding differences.
        limit = 1e-10
        worst = max((abs(u - v) / max(abs(v), 1e-3) for x, y in zip(rows, rows0) for u, v in zip(x, y)), default=float("inf"))
        check(f"trajectory {backend} against serial", len(rows) == 5000 == len(rows0) and worst <= limit,
              f"worst relative difference {worst:.3e}, limit {limit:g}")
        rec = next((x[7:] for x in r.stdout.splitlines() if x.startswith("result ")), "{}")
        res = json.loads(rec)
        expect = bool(a.expect_device and backend == "device")
        check(f"record {backend} states where the kernel ran",
              res.get("backend") == backend and res.get("kernel_on_device") is expect, f"kernel_on_device={res.get('kernel_on_device')}")

# ---------------------------------------------------------------- rejection
for bad in (["--n", "0"], ["--theta", "-1"], ["--backend", "fpga"], ["--bogus"], ["--leaf-size", "0"]):
    r = run([gpu, *bad])
    check(f"rejects {' '.join(bad)}", r.returncode == 2, f"rc={r.returncode}")

# ----------------------------------------------------------------- mutation
if a.expect_no_device:
    source = Path(a.source).resolve()
    text = source.read_text()
    mutants = {
        # Removing the j != i test is not listed: a softened self-pair has zero offset
        # and contributes exactly zero, so that change cannot alter any force.
        "monopole softening dropped": ("const double r2 = d2 + softening_sq;", "const double r2 = d2;"),
        "cell holding the target not forced open": ("if (!holds_i && s * s < theta_sq * d2)", "if (s * s < theta_sq * d2)"),
        "acceptance test reversed": ("s * s < theta_sq * d2", "s * s > theta_sq * d2"),
        "last child never visited": ("for (int c = 0; c < 8; ++c) {\n                if (node.child[c] >= 0) stack", "for (int c = 0; c < 7; ++c) {\n                if (node.child[c] >= 0) stack"),
        "softening dropped from leaf sum": ("ex * ex + ey * ey + ez * ez + softening_sq", "ex * ex + ey * ey + ez * ez"),
        "leaf range one short": ("k < node.first + node.count; ++k", "k + 1 < node.first + node.count; ++k"),
        "tree order used as slot": ("slot_[order_[k]] = static_cast<std::int32_t>(k)", "slot_[k] = static_cast<std::int32_t>(k)"),
    }
    with tempfile.TemporaryDirectory() as tmp:
        for name, (old, new) in mutants.items():
            if old not in text:
                check(f"mutation '{name}' applies", False, "pattern not found")
                continue
            src = source.parent / f".mutant_{os.getpid()}.cpp"
            exe = Path(tmp) / "mutant"
            src.write_text(text.replace(old, new))
            try:
                b = run([*shlex.split(a.cxx), *shlex.split(a.cxxflags), "-w", "-o", str(exe), str(src)])
            finally:
                src.unlink()
            if b.returncode != 0:
                check(f"mutation '{name}' builds", False, b.stderr[-200:])
                continue
            caught = [run([str(exe), "--self-test", "--backend", be], OMP_NUM_THREADS="4").returncode != 0
                      for be in ("device", "cpu")]
            check(f"mutation caught: {name}", any(caught), f"device={caught[0]} cpu={caught[1]}")

failed = [c["name"] for c in checks if not c["ok"]]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps({
    "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "mode": "device" if a.expect_device else "no device: target regions ran on the host; says nothing about a GPU",
    "real_gpu_executed": bool(a.expect_device and not failed),
    "cxx": a.cxx, "cxxflags": a.cxxflags, "device_probe": info,
    "mandatory_without_device_rc": None if a.expect_device else mandatory_without_device_rc,
    **execution_context(),
    "checks": checks, "failed": failed}, indent=1) + "\n")
print(f"{'PASS' if not failed else 'FAIL'}: GPU gate, {len(checks)} checks, {len(failed)} failed")
sys.exit(1 if failed else 0)
