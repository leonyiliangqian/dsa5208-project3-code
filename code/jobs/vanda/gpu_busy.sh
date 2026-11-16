#!/bin/bash
# Evidence from outside the program that the kernel used a GPU: sample
# nvidia-smi while a run with mandatory offload is in progress. The program's
# own report could be wrong; the driver's view of the GPU is independent of it.
#   gpu_busy.sh OUT_DIR BH_GPU_BINARY
# Exit 0 only if the run succeeds and some sample shows nonzero GPU utilisation.
set -uo pipefail
out=${1:?output directory}
bin=${2:?bh_gpu binary}
samples=$out/gpu_samples.csv
: > "$samples"
( while :; do
      nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits >> "$samples" 2>&1
      nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >> "$out/gpu_apps.txt" 2>&1
      sleep 1
  done ) &
sampler=$!
OMP_TARGET_OFFLOAD=MANDATORY "$bin" --require-device --n 1000000 --steps 5 --leaf-size 8
rc=$?
kill "$sampler" 2>/dev/null
wait "$sampler" 2>/dev/null
peak=$(awk -F', *' 'NF>=4 && $3+0 > m {m=$3+0} END {print m+0}' "$samples")
echo "run_rc=$rc samples=$(wc -l < "$samples") peak_gpu_utilisation=$peak"
[ "$rc" -eq 0 ] && [ "$peak" -gt 0 ]
