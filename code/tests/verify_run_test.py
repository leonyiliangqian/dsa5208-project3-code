#!/usr/bin/env python3
"""Regression checks for scripts/verify_run.py's phase and stdout checks.

Pilot4 job 1535640 (256 ranks) was flagged by two over-strict checks: the sum of
independently reduced phase maxima exceeded the wall-time maximum, and the stdout wall
time printed with %.4f differed by double rounding from the record's %.6f value. These
cases use those real values, and records with genuine inconsistencies that must still fail.

  python3 tests/verify_run_test.py      (from code/)
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from verify_run import phase_problems, stdout_wall_ok  # noqa: E402

failed = 0


def expect(name, ok):
    global failed
    print(f"{'PASS' if ok else 'FAIL'} {name}")
    failed += not ok


def rec(wall, integrate, exchange, build, traverse, **extra):
    r = {"wall_s": wall, "integrate_s": integrate, "exchange_s": exchange, "build_s": build, "traverse_s": traverse}
    r.update(extra)
    return r


# Real records from pilot4 1535640 and bench1 1528774.
p4_strong = rec(1.26982, 0.000101, 0.415638, 0.62734, 0.596956)
p4_weak = rec(15.05075, 0.001161, 1.163878, 9.766635, 5.032654)
b1_64x1 = rec(2.5493, 0.0004, 0.2817, 0.7764, 1.7446)
b1_serial = rec(58.0539, 0.0217, 0.0, 0.3013, 57.7309)

expect("pilot4 256x1 strong: phase maxima over wall are allowed with 256 ranks", phase_problems(p4_strong, 256) == [])
expect("pilot4 256x1 weak: consistent", phase_problems(p4_weak, 256) == [])
expect("bench1 64x1: consistent", phase_problems(b1_64x1, 64) == [])
expect("bench1 serial: phases add up", phase_problems(b1_serial, 1) == [])

# Genuine inconsistencies must still fail.
expect("one rank: phases that do not add up fail", phase_problems(p4_strong, 1) != [])
expect("serial: missing phase time fails", phase_problems(rec(58.05, 0.02, 0.0, 0.30, 40.0), 1) != [])
expect("many ranks: a phase longer than the wall time fails", phase_problems(rec(1.27, 0.0, 0.1, 0.2, 1.40), 256) != [])
expect("many ranks: phases far under the wall time fail", phase_problems(rec(1.27, 0.0, 0.1, 0.2, 0.5), 256) != [])
expect("negative phase fails", phase_problems(rec(1.27, -0.01, 0.4, 0.6, 0.6), 256) != [])
expect("traverse mean above its maximum fails",
       phase_problems(rec(1.27, 0.0, 0.4, 0.6, 0.6, traverse_mean_s=0.7, traverse_min_s=0.5), 256) != [])
expect("wall minimum above wall maximum fails", phase_problems(rec(1.27, 0.0, 0.4, 0.6, 0.6, wall_min_s=1.3), 256) != [])

# Stdout against the record.
line = "ranks=256 threads=1 schedule=dynamic chunk=64 n=12800000 steps=2 wall_s=15.0507 build_s=9.7666"
expect("pilot4 weak: 15.0507 printed, 15.050750 recorded (double rounding) passes", stdout_wall_ok(15.05075, line))
expect("rounded the other way also passes", stdout_wall_ok(15.05075, line.replace("15.0507", "15.0508")))
expect("a different wall time fails", not stdout_wall_ok(15.0512, line))
expect("missing wall_s fails", not stdout_wall_ok(15.05075, "ranks=256 build_s=9.7666"))
expect("build_s is not taken for wall_s", not stdout_wall_ok(9.7666, "build_s=9.7666 wall_s=15.0507"))
expect("serial stdout passes", stdout_wall_ok(58.053852, "method=bh n=1000000 steps=3 wall_s=58.0539 build_s=0.3013"))

print(f"verify_run_test: {17 - failed} passed, {failed} failed")
sys.exit(1 if failed else 0)
