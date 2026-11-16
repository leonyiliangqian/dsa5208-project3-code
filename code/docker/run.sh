#!/bin/sh
# Run a command in the Project 3 development container, building the image on
# first use. Only the Project 3 directory is mounted, at /work.
#   docker/run.sh make check-linux
#   docker/run.sh bash
set -eu
here=$(cd "$(dirname "$0")" && pwd)
project=$(cd "$here/../.." && pwd)
image=dsa5208-p3-dev:bookworm

if ! docker image inspect "$image" >/dev/null 2>&1; then
    docker build -t "$image" "$here"
fi
tty=""
[ -t 0 ] && [ -t 1 ] && tty="-it"
exec docker run --rm $tty -v "$project":/work -w /work/code "$image" "$@"
