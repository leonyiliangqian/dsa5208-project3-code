#!/usr/bin/env python3
"""Validators must label where they ran: unit cases, then the real serial validator.

  python3 tests/test_execution_context.py --serial bin/<os>/bh_serial
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from execution_context import KINDS, execution_context

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--serial", required=True)
opts = parser.parse_args()
failures = 0


def check(name, ok, detail=""):
    global failures
    failures += not ok
    print(f"context {name} {'PASS' if ok else 'FAIL'} {detail}")


BATCH = {"PBS_ENVIRONMENT": "PBS_BATCH", "PBS_JOBID": "1.mgmt", "PBS_QUEUE": "q", "SOURCE_GIT_REV": "abc"}
c = execution_context(BATCH)
check("real_batch", c["execution_context"] == "pbs_batch" and c["pbs_jobid"] == "1.mgmt"
      and c["pbs_queue"] == "q" and c["source_git_rev"] == "abc" and c["kind"] == KINDS["pbs_batch"])
c = execution_context({})
check("no_pbs", c["execution_context"] == "local" and c["pbs_jobid"] is None
      and c["source_git_rev"] == "unknown" and c["kind"] == KINDS["local"])
# The stand-ins used by the job self-tests and simulations set a job ID only.
c = execution_context({"PBS_JOBID": "selftest.local"})
check("simulated_jobid", c["execution_context"] == "local" and c["pbs_jobid"] is None)
c = execution_context({"PBS_ENVIRONMENT": "PBS_INTERACTIVE", "PBS_JOBID": "1.mgmt"})
check("interactive", c["execution_context"] == "local")
c = execution_context({"PBS_ENVIRONMENT": "PBS_BATCH"})
check("batch_without_jobid", c["execution_context"] == "local")
check("kinds_differ", "not cluster" in KINDS["local"] and "not cluster" not in KINDS["pbs_batch"]
      and all("not performance evidence" in k for k in KINDS.values()))

# The file a validator writes carries the label, in both contexts.
clean = {k: v for k, v in os.environ.items() if not k.startswith("PBS_") and k != "SOURCE_GIT_REV"}
with tempfile.TemporaryDirectory() as tmp:
    for name, extra, want in (("local", {}, "local"), ("batch", BATCH, "pbs_batch")):
        out = Path(tmp) / f"{name}.json"
        run = subprocess.run([sys.executable, "tests/validate_serial.py", "--binary", opts.serial, "--out", str(out)],
                             cwd=root, env={**clean, **extra}, capture_output=True, text=True)
        d = json.loads(out.read_text()) if out.exists() else {}
        check(f"validate_serial_{name}", run.returncode == 0 and d.get("execution_context") == want
              and d.get("kind") == KINDS[want], f"rc={run.returncode} context={d.get('execution_context')}")
print("PASS: execution context" if not failures else f"FAIL: execution context, {failures} failed")
sys.exit(1 if failures else 0)
