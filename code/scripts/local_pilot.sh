#!/bin/sh
# Exercise the measurement pipeline end to end inside Docker. Local development
# evidence only: these numbers are not Atlas/Vanda results and must not be
# reported as such. Usage (from code/): docker/run.sh sh scripts/local_pilot.sh
set -eu
RUN_ID=${RUN_ID:-pilot-$(date -u +%Y%m%dT%H%M%SZ)}
OUT=artifacts/local/runs/$RUN_ID
N=${N:-50000}; STEPS=${STEPS:-3}; REPS=${REPS:-3}
ARGS="--n $N --steps $STEPS --warmup 1 --theta 0.5 --leaf-size 8 --seed 42"
make all hybrid >/dev/null

for rep in $(seq 1 "$REPS"); do
    python3 scripts/run_case.py --out "$OUT" --case serial --rep "$rep" --allocated-cores 1 \
        -- bin/linux/bh_serial $ARGS || true
    for rt in 1x1 1x4 2x2 4x1 1x8 2x4 8x1; do
        r=${rt%x*}; t=${rt#*x}
        OMP_NUM_THREADS=$t python3 scripts/run_case.py --out "$OUT" --case "hybrid_$rt" --rep "$rep" \
            --allocated-cores $((r * t)) \
            --launcher "mpirun -np $r --bind-to none" \
            -- bin/linux/bh_hybrid $ARGS --schedule dynamic --chunk 64 || true
    done
done
python3 scripts/analyze.py "$OUT"
