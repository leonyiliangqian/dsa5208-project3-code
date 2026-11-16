#!/usr/bin/env python3
"""Read-only local tool inventory; does not install or access cluster accounts."""
import json
import platform
import shutil
import subprocess

result = {"platform": platform.platform(), "tools": {}}
for tool in ("c++", "make", "python3", "mpicxx", "mpirun", "cmake", "pdflatex", "biber"):
    path = shutil.which(tool)
    item = {"path": path}
    if path:
        run = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=15)
        lines = (run.stdout or run.stderr).splitlines()
        item["version"] = lines[0] if lines else "unknown"
    result["tools"][tool] = item
print(json.dumps(result, indent=2))
