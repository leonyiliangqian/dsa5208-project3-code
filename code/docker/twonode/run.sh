#!/bin/sh
# Start two containers (node1, node2) with the PBS-like cpuset Vanda gave
# (0,2,...,14), run a command as dev on node1 with a PBS-style node file at
# /tmp/nodefile (each host listed once per CPU, 8 each), then remove both
# containers.
#   docker/twonode/run.sh 'mpirun --hostfile /tmp/nodefile -np 2 hostname'
# TWONODE_UPPER_HOSTNAME=1 gives each container an upper-case hostname (NODE1)
# while the node file and DNS keep node1, as on Vanda (hostname CN-158, PBS
# cn-158), so Open MPI starts a second daemon on the head node.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
project=$(cd "$here/../../.." && pwd)
image=dsa5208-p3-twonode:bookworm
cpus=${TWONODE_CPUSET:-0,2,4,6,8,10,12,14}
docker image inspect dsa5208-p3-dev:bookworm >/dev/null 2>&1 || docker build -t dsa5208-p3-dev:bookworm "$here/.."
docker build -q -t "$image" "$here" >/dev/null   # cached unless the Dockerfile changed
net=p3-twonode-$$
docker network create "$net" >/dev/null
cleanup() { docker rm -f node1 node2 >/dev/null 2>&1 || true; docker network rm "$net" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
for n in node1 node2; do
    hn=$n; [ "${TWONODE_UPPER_HOSTNAME:-0}" = 1 ] && hn=$(echo "$n" | tr '[:lower:]' '[:upper:]')
    docker run -d --rm --name "$n" --hostname "$hn" --network "$net" --network-alias "$n" --cpuset-cpus="$cpus" \
        -v "$project":/work "$image" >/dev/null
done
for n in node1 node2; do for i in 1 2 3 4 5 6 7 8; do echo $n; done; done | docker exec -i -u dev node1 sh -c 'cat > /tmp/nodefile'
docker exec -u dev -w /work/code node1 sh -c 'for i in 1 2 3 4 5; do ssh -o ConnectTimeout=2 node2 true 2>/dev/null && exit 0; sleep 1; done; exit 1' \
    || { echo "node2 sshd not reachable"; exit 1; }
docker exec -u dev -w /work/code -e PBS_NODEFILE=/tmp/nodefile node1 bash -c "$1"
