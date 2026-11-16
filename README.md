# DSA5208 Project 3: MPI+OpenMP Barnes–Hut N-body simulation

Source code for the DSA5208 Project 3 report (Yan Yixin, Zhang Yue, Leon Yiliang Qian).

Everything is in [`code/`](code/). **Start with [`code/README.md`](code/README.md)**: it explains how to build and run the
serial, MPI+OpenMP and GPU programs, the correctness tests and the measurement pipeline.

```sh
git clone https://github.com/leonyiliangqian/dsa5208-project3-code.git
cd dsa5208-project3-code/code
docker/run.sh make all hybrid gpu
docker/run.sh make check check-hybrid check-gpu
```

These are the same files as the source ZIP submitted with the report. Run logs, terminal output and run records are not included.
