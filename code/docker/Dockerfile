# Local Linux development environment for Project 3: GCC, Open MPI, Python 3.
# Development and correctness evidence only. Formal timings come from Atlas or
# Vanda, rebuilt there with the cluster's own compiler and MPI.
FROM debian:bookworm-20241016-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        g++ make libopenmpi-dev openmpi-bin python3 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Open MPI refuses to run as root without extra flags; use an ordinary user.
RUN useradd --create-home --uid 1000 dev
USER dev
WORKDIR /work/code
