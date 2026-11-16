// Toolchain probe: MPI_THREAD_FUNNELED is provided, each rank runs an OpenMP
// team, and an MPI call made by the main thread after the parallel region works.
#include <mpi.h>
#include <omp.h>
#include <cstdio>

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    int min_team = 0;
    MPI_Allreduce(&team, &min_team, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const bool ok = provided >= MPI_THREAD_FUNNELED && size > 1 && min_team > 1;
    if (rank == 0)
        std::printf("hybrid_check ranks=%d provided=%d funneled=%d min_team=%d %s\n", size,
                    provided, MPI_THREAD_FUNNELED, min_team, ok ? "PASS" : "FAIL");
    MPI_Finalize();
    return ok ? 0 : 1;
}
