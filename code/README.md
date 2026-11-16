# Barnes–Hut n-body: serial, MPI+OpenMP and GPU

DSA5208 Project 3 source code. C++17 gravitational n-body simulation with the Barnes–Hut tree approximation: a serial reference,
a hybrid MPI+OpenMP solver and an OpenMP target (GPU) solver. The report is handed in separately as a PDF.

This README covers everything needed to build and run the code, and it stands alone. Run records, logs and the report source are
not part of this package. All commands below run from this directory.

## Quick start

With Docker (the image is built from `docker/Dockerfile` on first use: GCC, Open MPI, Python 3):

```sh
unzip dsa5208-project3-source.zip && cd dsa5208-project3/code
docker/run.sh make help                              # list the targets
docker/run.sh make all hybrid gpu                    # serial, MPI+OpenMP, and the GPU solver on the host (no offload)
docker/run.sh make check check-hybrid check-gpu      # the three correctness gates
docker/run.sh bash tests/pipeline_e2e.sh             # tiny end-to-end run: plan, runs, analysis, figures
```

On Linux with GCC, Open MPI and Python 3 installed, run the same commands without `docker/run.sh`. The GPU build above runs its
target regions on the host and checks logic only; a real GPU needs its own compiler and build folder (see OpenMP target solver).

| Path | Contents |
|---|---|
| `common/nbody.hpp` | Particles, Plummer-profile positions, direct summation, octree, leapfrog |
| `common/run_support.hpp` | Shared command-line options and validation, state dump, JSON helpers |
| `serial/barnes_hut.cpp` | Serial driver (`bh_serial`): speedup denominator, `--self-test`, `--validate` |
| `hybrid/bh_hybrid.cpp` | MPI+OpenMP driver (`bh_hybrid`); 1 rank = OpenMP only, 1 thread = MPI only |
| `gpu/bh_gpu.cpp` | OpenMP target driver (`bh_gpu`): `--backend device` or `cpu` |
| `tests/` | Serial, hybrid and GPU gates, energy and timestep study, toolchain probes, pipeline and job-library tests |
| `scripts/` | `run_case.py` (one recorded, time-limited run), `run_manifest.py` (all runs of a plan), `analyze.py`, `verify_run.py` (rechecks a run from its raw records), `plan_budget.py` (checks a plan against its job request), `report_figures.py`, `report_tables.py`, `preflight.py` |
| `manifests/` | Experiment plans as JSON. `local_*` are tiny test plans; a formal job refuses a plan whose status is not `frozen` |
| `jobs/vanda/` | PBS job scripts, gate library and submission wrapper for the Vanda cluster |
| `docker/` | Development image and `run.sh`; `twonode/` simulates two nodes; `tex/` is a LaTeX compile test |

## Dependencies

| Needed for | Tools | Used on Vanda |
|---|---|---|
| Serial | C++17 compiler, GNU make, Python 3.7 or later (standard library only, for the gates and scripts) | GCC 13.2.0 |
| MPI+OpenMP | `mpicxx` with OpenMP and at least `MPI_THREAD_FUNNELED` | `OpenMPI/4.1.6-GCC-13.2.0` |
| GPU | A compiler with OpenMP target offload to the device; any OpenMP compiler still builds it and runs target regions on the host | `nvhpc-hpcx-cuda12/26.1` (`nvc++`), NVIDIA A40 |
| Docker (optional) | Docker; the image in `docker/Dockerfile` holds the Linux tools above except the GPU compiler | not used |

macOS with Apple clang has no OpenMP or MPI here, so it builds only the serial program. Use Docker for the rest.

## Build

Binaries go to `bin/<os>/` (`bin/darwin`, `bin/linux`), so builds for different systems never mix. `make help` lists all targets.

```sh
make all                # serial: bin/<os>/bh_serial
make hybrid             # MPI+OpenMP: bin/<os>/bh_hybrid
make gpu                # OpenMP target: bin/<os>/bh_gpu, on the host without offload support
```

For a real GPU, build with NVHPC into its own folder, so it never mixes with the host build in `bin/<os>/`:
`make BIN=bin/nvhpc CXX=nvc++ GPU_CXX=nvc++ CXXFLAGS="-O3 -std=c++17" GPU_OMPFLAGS="-mp=gpu -Minfo=mp" all gpu`.

In Docker, put `docker/run.sh` in front: `docker/run.sh make all hybrid gpu`.

## Small validation

Each of these takes under a few minutes on a laptop and needs no cluster:

```sh
make check                        # serial gate
docker/run.sh make check-hybrid   # MPI+OpenMP against serial, particle by particle
docker/run.sh make check-gpu      # GPU solver's logic, on the host
```

Without Docker, on Linux with the tools above, run the same `make` targets directly. For a one-minute look:

```sh
./bin/linux/bh_serial --self-test
./bin/linux/bh_serial --validate --n 2000 --theta 0.5
./bin/linux/bh_serial --n 20000 --steps 3 --warmup 1
OMP_NUM_THREADS=2 mpirun -np 2 --bind-to none ./bin/linux/bh_hybrid --n 20000 --steps 3 --warmup 1 --schedule dynamic --chunk 64
OMP_NUM_THREADS=2 ./bin/linux/bh_gpu --n 20000 --steps 3 --warmup 1 --backend cpu
```

### Gates

- `make check`: edge-case self-test (empty, singleton, all-coincident, coincident
  cluster, separations below the depth-64 cell, unequal masses, lattice points on
  cell planes, two-particle closed-form force at theta 0/0.5/10, symmetric cube,
  plus negative controls that must fail); invalid arguments exit 2; theta=0 vs
  direct summation max relative error <= 1e-10 at leaf sizes 1 and 8.
- `make check-hybrid`: 15 configurations (1–5 ranks, 1–4 threads, static/dynamic/
  guided, uneven N, ranks owning nothing, N=1) compared with serial at the same
  settings. Tolerances are fixed in `tests/validate_hybrid.py`: 1-step
  accelerations `1e-12·|ref| + 1e-14·max|ref|`; 10-step positions, velocities and
  accelerations `1e-10·|ref| + 1e-13·max|ref|`. Also: 3 repeated threaded runs must
  be bitwise identical, invalid arguments must exit 2 on all ranks without hanging,
  and a 1e-9 perturbation must be detected.

Each writes JSON with source and binary hashes to `artifacts/local/`, which is created when you run it.

## Running

Both programs accept `--n`, `--theta`, `--dt`, `--steps`, `--warmup`, `--seed`,
`--leaf-size`, `--energy` and `--dump FILE`. `bh_serial` adds `--method bh|direct`,
`--validate` and `--self-test`; `bh_hybrid` adds `--schedule static|dynamic|guided`
and `--chunk C` (0 = OpenMP default). Threads per rank come from `OMP_NUM_THREADS`.

```sh
./bin/linux/bh_serial --n 50000 --steps 5 --warmup 1
OMP_NUM_THREADS=4 mpirun -np 2 --bind-to none ./bin/linux/bh_hybrid \
    --n 50000 --steps 5 --warmup 1 --schedule dynamic --chunk 64
```

Each prints a human-readable line and a `result {...}` JSON line with the
configuration and timings. Process binding matters: Open MPI's default binds a
rank to one core when there are few ranks, so its OpenMP threads share that core
(measured locally: 1 rank x 4 threads took 0.89 s bound vs 0.24 s with
`--bind-to none`). Cluster jobs must give each rank as many cores as threads.

## Numerical model

Softened gravity (G = 1, softening 0.025), Plummer-profile positions with small
random velocities from a fixed seed (not an equilibrium Plummer model),
kick–drift–kick leapfrog. Leaves hold up to `--leaf-size` particles (default 8);
particles still sharing a cell at depth 64 stay in one leaf, so none are dropped.
A cell is used as a monopole only if it passes s/d < theta and does not contain
the target particle; otherwise it is opened, and opened leaves are summed
pairwise without self-interaction. Default leaf size 8: in a local sweep L=8 and
L=16 were fastest with overlapping ranges. It was fixed at 8 before the formal runs, and the cluster leaf study kept it.

## Hybrid design (replicated baseline)

Rank 0 generates the initial state and broadcasts positions, velocities and
masses. Rank r owns a contiguous ID range (balanced; ranks may own nothing).
Each step: half kick and drift of owned particles, `MPI_Allgatherv` of positions
in ID order, identical global tree build on every rank, OpenMP
`schedule(runtime)` force loop over owned particles, half kick. Masses are sent
once; velocities and accelerations are gathered only for `--dump`/`--energy`.
`MPI_THREAD_FUNNELED` is required and checked; MPI is called only by the main
thread outside parallel regions. After the timed loop, all ranks verify that
their global positions and masses agree bitwise and that owned counts sum to N.

This is not a distributed tree: every rank builds the whole tree and stores all
N positions, so memory per rank is O(N) and the build is redundant.

## Timing

Timed region: `--steps` steps after setup (initial forces, optional energy,
`--warmup` steps). Hybrid: barrier, then per-rank `MPI_Wtime`, reduced with
MPI_MAX; phase times (integrate, exchange, build, traverse) are per-rank sums
reduced with MPI_MAX, plus the traverse mean for the imbalance ratio max/mean. Each maximum
may come from a different rank, so with several ranks the phases need not sum to the wall
time, and the exchange time includes any wait for slower ranks at `MPI_Allgatherv`.
Serial times the same region and phases with `steady_clock`. Use serial
Barnes–Hut, never direct summation, as the speedup denominator.

## Measurement pipeline

```sh
python3 scripts/run_manifest.py manifests/<m>.json --out <run dir> --bin <bin dir> \
    [--mpirun ...] [--cores-per-node C] [--nodes K] [--first-cpu CPU] [--deadline EPOCH] [--plan-only]
python3 scripts/analyze.py <run dir> [<run dir> ...] [--out <dir>]     # summary.csv/json
python3 scripts/report_figures.py <summary.json> [...] --allow-local --out <dir>     # pgfplots figures and tables
```

A manifest lists cases; list-valued fields are swept. Repetitions are
interleaved. Every run has its own timeout and runs in its own process group
(TERM, then KILL); a run that cannot finish before the deadline is recorded as
skipped. Each record carries the manifest id and sha256, study tag, command,
allocated cores, CPU model, source and binary hashes, compiler, MPI and flags.
`analyze.py` groups repetitions by every setting that can change a timing and
computes speedup only against a serial baseline with the same problem, CPU
model, source tree and flags. `report_figures.py` writes pgfplots figures and
tables; it refuses inputs outside `artifacts/cluster/` unless `--allow-local`
with an `--out` other than the report's own folder, and then marks every caption
LOCAL PIPELINE TEST.

Tests: `docker/run.sh bash tests/pipeline_e2e.sh` (manifest -> runs -> analysis
-> figures on tiny inputs, timeout, deadline, interruption and refusal paths);
`bash tests/jobs/pbs_entry_test.sh` (job scripts started outside code/, as PBS does);
`docker/run.sh bash tests/gpu_timing_e2e.sh` (G2 pipeline and fallback refusal, with a stub);
`docker/twonode/run.sh 'bash tests/jobs/gates_selftest.sh'` (gate library fault
injection on two simulated nodes);
`bash tests/jobs/allocation_check_test.sh` (allocation check: host-name case,
Open MPI's head-node alias, mixed CPU models, CPU counts);
`docker/run.sh bash tests/jobs/launch_policy_test.sh` (every cluster launch states
its binding; the exchange probe with 64 ranks on one node, and its failure).
`TWONODE_UPPER_HOSTNAME=1 docker/twonode/run.sh ...` gives the simulated nodes
upper-case host names, as on Vanda; with `OMPI_MCA_if_base_do_not_resolve=1`
Open MPI then starts a second daemon on the head node, as it does there.

## Vanda (PBS)

Every job is submitted only through `jobs/vanda/submit.sh`, from a clean clone of the commit
to run. Without `--yes` it only prints the `qsub` line; it records the login node's commit as
`SOURCE_GIT_REV` (compute nodes have no git). Free queue (up to 160 CPUs per job and user) and
charged queue (needed for 256 cores; charged on the request, cores x walltime x rate):

```sh
bash jobs/vanda/submit.sh bench1 --queue auto_free --no-project              # dry run, free
bash jobs/vanda/submit.sh bench4 --queue auto --project <your-project>       # dry run, charged
``` Jobs build on the compute node and
run their steps as gates (`jobs/vanda/gates.sh`): each has a time limit, a
process snapshot before expiry and an ownership-checked cleanup on every node
after it; kinds are critical, required and optional. The jobs depend on this site's queues and modules, so elsewhere treat them as examples.

| Job | Request | Purpose |
|---|---|---|
| `correctness.pbs` | 1 x 8 cores, 30 min | serial, hybrid, placement, sanitizer gates on one node |
| `twonode.pbs` | 2 x 8 cores, scatter, 30 min | two-host correctness, portable build, `pbs_tmrsh` launch |
| `diag.pbs` | 2 x 8 cores, 4 GB, 20 min | locates a launch stall, one factor per step |
| `pilot.pbs`, `pilot16.pbs` | 1 x 64 cores or 1 x 16 cores, 60 min | one-repetition sizing runs |
| `pilot2.pbs`, `pilot4.pbs` | 2 or 4 x 64 cores | launch, correctness and sizing at 128 and 256 cores |
| `bench16.pbs`, `bench1.pbs`, `bench2.pbs`, `bench4.pbs` | 1 x 16 cores; 1, 2 or 4 x 64 cores | formal studies from a frozen plan, with one CPU model |
| `gpu_validate.pbs`, `gpu_timing.pbs` | 1 GPU, 36 cores, 60 GB | device and correctness gates; device against CPU timing |

### Measured configuration (frozen)

The report's CPU results come from three jobs, each with its own frozen plan; plans and solver sources are left as measured.

| Plan in `manifests/` | Id | SHA-256 | Job and commit |
|---|---|---|---|
| `studies_16core.json` | `studies-16core-v1` | `2525574ba733e4a74c44100daa0a8ec2b8a89e7e9e81c7a8025ea44439226dcd` | `bench16.pbs`, `e86bf5d` |
| `studies_single_node.json` | `studies-single-node-v1` | `ea1ceb2c734b08009ce4a582ed8d1c0e2d8d6a877d0e30d0d392274de7c1c64b` | `bench1.pbs`, `a64e0fe` |
| `studies_four_node.json` | `studies-four-node-v1` | `0ddec3b7b31d0d30bfc49c54010c6d25d31954dc0eccece63b7c9a5d615edf6c` | `bench4.pbs`, `a72ce26` |

All three built with `-O3 -std=c++17 -Wall -Wextra -march=native` (Open MPI 4.1.6, GCC 13.2.0) on the compute node. The solver
sources (`common/`, `serial/`, `hybrid/`, `gpu/`) are unchanged since then. The source hash in each run record also covers the
`Makefile`, whose job-check targets and `help` changed afterwards, so a rebuild records a different hash with the same solver code.

## OpenMP target solver

`gpu/bh_gpu.cpp` offloads the force traversal; the tree is built on the host. `--backend device`
uses `target`, and `--backend cpu` runs the same kernel on host threads. `--require-device` exits 7
unless the kernel ran on a device with separate memory. A compiler without offload support runs
target regions on the host, so `make check-gpu` then tests logic only. `jobs/vanda/gpu_validate.pbs`
runs it on a device.

```sh
docker/run.sh make check-gpu                                   # no GPU: host fallback, logic only
make BIN=bin/nvhpc CXX=nvc++ GPU_CXX=nvc++ CXXFLAGS="-O3 -std=c++17" GPU_OMPFLAGS="-mp=gpu -Minfo=mp" \
    GPU_GATE_ARGS=--expect-device check-gpu                    # on a GPU node, separate build folder
```

`--expect-device` makes the gate fail unless the kernel really ran on a device; `--expect-no-device` (the default) makes the host
fallback fail if it unexpectedly offloads.

