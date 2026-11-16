#!/bin/bash
# The binaries must carry the portable flags, and the compiler must resolve
# them to the generic x86-64 target with no AVX/FMA code generation.
# Needs OUT, BIN and CXXFLAGS in the environment.
: "${OUT:?}" "${BIN:?}" "${CXXFLAGS:?}"
{
    echo "CXXFLAGS=$CXXFLAGS"
    echo "== mpicxx -show"; mpicxx -show
    echo "== resolved target"; mpicxx $CXXFLAGS -Q --help=target 2>/dev/null | grep -E '^ +-m(arch|tune)='
    echo "== ISA macros defined by these flags"; mpicxx $CXXFLAGS -dM -E -x c++ /dev/null | grep -E '__(SSE4_2|AVX|AVX2|FMA|AVX512F)__' || echo none
    echo "== embedded in binaries"
    for b in "$BIN/bh_serial" "$BIN/bh_hybrid"; do printf '%s: ' "$b"; grep -a -o -m1 -- '-O3 -std=c++17[^[:cntrl:]]*' "$b" | head -1; done
} | tee "$OUT/env/build_flags.txt"
grep -Eq -- '-march=[[:space:]]+x86-64$' "$OUT/env/build_flags.txt" \
    && grep -q '^none$' "$OUT/env/build_flags.txt" \
    && [ "$(grep -c -- '-march=x86-64 -mtune=generic' "$OUT/env/build_flags.txt")" -ge 3 ] \
    && ! grep -q -- '-march=native' "$OUT/env/build_flags.txt"
