# Single entry point for building and checking. MPI+OpenMP targets are added
# as they land; `make help` lists what exists.
CXX      ?= c++
MPICXX   ?= mpicxx
PYTHON   ?= python3
MPIRUN   ?= mpirun
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -march=native
# GCC and Clang spell this -fopenmp; override for other compilers.
OMPFLAGS ?= -fopenmp
# Skip the deprecated MPI C++ bindings headers; the code uses the C API only.
MPIDEFS  := -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
# Embed the compile flags in each binary so result records carry them.
BUILDDEF  = -DNBODY_CXXFLAGS='"$(strip $(CXXFLAGS))"'
# Per-OS output directory so macOS and Docker/Linux binaries never mix.
BIN      ?= bin/$(shell uname -s | tr '[:upper:]' '[:lower:]')
# Where gate evidence goes; cluster jobs point this at their run directory.
EVIDENCE ?= artifacts/local

.DEFAULT_GOAL := all

.PHONY: help all hybrid gpu check-gpu clean check check-context check-hybrid preflight sanitize toolchain placement internode

help:       ## Show this help
	@awk -F ':.*## *' '/^[a-z-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

all: $(BIN)/bh_serial  ## Build the serial reference

$(BIN)/bh_serial: serial/barnes_hut.cpp common/nbody.hpp common/run_support.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) $(BUILDDEF) -o $@ $<

hybrid: $(BIN)/bh_hybrid  ## Build the MPI+OpenMP solver (needs mpicxx)

$(BIN)/bh_hybrid: hybrid/bh_hybrid.cpp common/nbody.hpp common/run_support.hpp | $(BIN)
	$(MPICXX) $(CXXFLAGS) $(OMPFLAGS) $(MPIDEFS) $(BUILDDEF) -o $@ $<

# OpenMP target solver. Without offload support the compiler runs target regions on
# the host, so check-gpu tests logic only. On NVHPC: GPU_CXX=nvc++ GPU_OMPFLAGS=-mp=gpu.
GPU_CXX ?= $(CXX)
GPU_OMPFLAGS ?= $(OMPFLAGS)
GPU_GATE_ARGS ?= --expect-no-device

gpu: $(BIN)/bh_gpu  ## Build the OpenMP target solver

$(BIN)/bh_gpu: gpu/bh_gpu.cpp common/nbody.hpp common/run_support.hpp | $(BIN)
	$(GPU_CXX) $(CXXFLAGS) $(GPU_OMPFLAGS) $(BUILDDEF) -o $@ $<

$(BIN):
	mkdir -p $(BIN)

check: $(BIN)/bh_serial  ## Serial gate: edge cases, argument rejection, theta=0 vs direct
	$(PYTHON) tests/validate_serial.py --binary $(BIN)/bh_serial \
	    --out $(EVIDENCE)/serial_validation_$(notdir $(BIN)).json

check-context: $(BIN)/bh_serial  ## Validators label where they ran (local or PBS batch)
	$(PYTHON) tests/test_execution_context.py --serial $(BIN)/bh_serial

# Debug build with AddressSanitizer and UndefinedBehaviorSanitizer; never timed.
SANFLAGS := -O1 -g -std=c++17 -Wall -Wextra -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=undefined

$(BIN)/bh_serial_san: serial/barnes_hut.cpp common/nbody.hpp common/run_support.hpp | $(BIN)
	$(CXX) $(SANFLAGS) -o $@ $<

sanitize: $(BIN)/bh_serial_san  ## Run self-test, validation and a short simulation under ASan/UBSan
	$(BIN)/bh_serial_san --self-test > /dev/null
	$(BIN)/bh_serial_san --validate --n 1000 --theta 0.5
	$(BIN)/bh_serial_san --n 2000 --steps 3 --theta 0.7 --leaf-size 4
	@echo "PASS: sanitizer run reported no errors"

# Toolchain probes: OpenMP team > 1, two MPI ranks, MPI_THREAD_FUNNELED + OpenMP.
$(BIN)/omp_check: tests/toolchain/omp_check.cpp | $(BIN)
	$(CXX) -O2 $(OMPFLAGS) -o $@ $<

$(BIN)/mpi_check: tests/toolchain/mpi_check.cpp | $(BIN)
	$(MPICXX) -O2 $(MPIDEFS) -o $@ $<

$(BIN)/hybrid_check: tests/toolchain/hybrid_check.cpp | $(BIN)
	$(MPICXX) -O2 $(OMPFLAGS) $(MPIDEFS) -o $@ $<

$(BIN)/placement_check: tests/toolchain/placement_check.cpp | $(BIN)
	$(MPICXX) -O2 $(OMPFLAGS) $(MPIDEFS) -o $@ $<

$(BIN)/internode_check: tests/toolchain/internode_check.cpp | $(BIN)
	$(MPICXX) -O2 $(MPIDEFS) -o $@ $<

# Bound placement: PLACE_R ranks per node on PLACE_NODES nodes, each rank on
# PLACE_T cores, one thread per core; the probe also checks the host count.
PLACE_R ?= 2
PLACE_T ?= 4
PLACE_NODES ?= 1
# Per-rank stage tracing on stderr (1) or off (0).
PLACE_TRACE ?= 1
placement: $(BIN)/placement_check  ## Check placement: PLACE_R ranks/node x PLACE_T threads on PLACE_NODES nodes, bound
	OMP_NUM_THREADS=$(PLACE_T) OMP_PROC_BIND=close OMP_PLACES=cores PLACEMENT_EXPECT_BOUND=1 \
	    PLACEMENT_EXPECT_HOSTS=$(PLACE_NODES) PLACEMENT_TRACE=$(PLACE_TRACE) \
	    $(MPIRUN) -np $(shell expr $(PLACE_R) \* $(PLACE_NODES)) --map-by ppr:$(PLACE_R):node:PE=$(PLACE_T) \
	    --bind-to core --report-bindings -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
	    -x PLACEMENT_EXPECT_BOUND -x PLACEMENT_EXPECT_HOSTS -x PLACEMENT_TRACE $(BIN)/placement_check

# Inter-node MPI: INTERNODE_NP/INTERNODE_HOSTS ranks per host, verified point-to-point
# (the probe pairs ranks on different hosts whatever the mapping) and Allgatherv.
# Placement is explicit and the same as place_mpi: one core per rank, bound. With
# no binding given, Open MPI binds each of more than two ranks to a package or NUMA
# domain; the old --map-by node launch relied on that and failed in job 1526501 on
# a cpuset split 36 + 28 over two packages, while bound launches on the same node
# passed. The cause of that failure is not established. Tests may override INTERNODE_PLACE.
INTERNODE_NP ?= 2
INTERNODE_HOSTS ?= 1
INTERNODE_PER_HOST = $(shell expr $(INTERNODE_NP) / $(INTERNODE_HOSTS))
INTERNODE_PLACE ?= --map-by ppr:$(INTERNODE_PER_HOST):node:PE=1 --bind-to core --report-bindings
internode: $(BIN)/internode_check  ## Verify MPI data exchange across INTERNODE_HOSTS hosts
	@[ $$(( $(INTERNODE_NP) % $(INTERNODE_HOSTS) )) -eq 0 ] || { echo "internode: INTERNODE_NP=$(INTERNODE_NP)" \
	    "is not a multiple of INTERNODE_HOSTS=$(INTERNODE_HOSTS)"; exit 2; }
	INTERNODE_EXPECT_HOSTS=$(INTERNODE_HOSTS) $(MPIRUN) -np $(INTERNODE_NP) $(INTERNODE_PLACE) \
	    -x INTERNODE_EXPECT_HOSTS $(BIN)/internode_check

toolchain: $(BIN)/omp_check $(BIN)/mpi_check $(BIN)/hybrid_check  ## Verify OpenMP, MPI and MPI+OpenMP work
	OMP_NUM_THREADS=4 $(BIN)/omp_check
	$(MPIRUN) -np 2 $(BIN)/mpi_check
	$(MPIRUN) -np 2 -x OMP_NUM_THREADS=2 $(BIN)/hybrid_check

# Local binding policy for tests; cluster jobs set placement explicitly.
MPIRUN_FLAGS ?= --bind-to none

check-hybrid: $(BIN)/bh_serial $(BIN)/bh_hybrid  ## Hybrid gate: serial vs MPI+OpenMP by particle ID
	$(PYTHON) tests/validate_hybrid.py --serial $(BIN)/bh_serial --hybrid $(BIN)/bh_hybrid \
	    --mpirun $(MPIRUN) --mpirun-flags "$(MPIRUN_FLAGS)" \
	    --out $(EVIDENCE)/hybrid_validation_$(notdir $(BIN)).json $(HYBRID_GATE_ARGS)

check-gpu: $(BIN)/bh_serial $(BIN)/bh_gpu  ## GPU solver gate (add GPU_GATE_ARGS=--expect-device on a GPU node)
	$(PYTHON) tests/validate_gpu.py --serial $(BIN)/bh_serial --gpu $(BIN)/bh_gpu \
	    --source gpu/bh_gpu.cpp --cxx "$(GPU_CXX)" --cxxflags "$(CXXFLAGS) $(GPU_OMPFLAGS)" \
	    --out $(EVIDENCE)/gpu_validation_$(notdir $(BIN)).json $(GPU_GATE_ARGS)

preflight:  ## Record the local tool inventory
	$(PYTHON) scripts/preflight.py

clean:      ## Remove build outputs for this OS
	rm -rf $(BIN)
