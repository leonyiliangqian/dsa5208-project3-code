// Toolchain probe: confirm an OpenMP team actually runs more than one thread.
#include <omp.h>
#include <cstdio>

int main() {
    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    std::printf("omp_check _OPENMP=%d max_threads=%d team=%d %s\n", _OPENMP, omp_get_max_threads(),
                team, team > 1 ? "PASS" : "FAIL");
    return team > 1 ? 0 : 1;
}
