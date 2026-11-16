// Toolchain probe: ranks agree on the world size and a collective sums rank ids.
#include <mpi.h>
#include <cstdio>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int sum = 0;
    MPI_Allreduce(&rank, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    const bool ok = size > 1 && sum == size * (size - 1) / 2;
    if (rank == 0) {
        char version[MPI_MAX_LIBRARY_VERSION_STRING];
        int len = 0;
        MPI_Get_library_version(version, &len);
        std::printf("mpi_check size=%d rank_sum=%d %s\nmpi_library %.*s\n", size, sum,
                    ok ? "PASS" : "FAIL", 80, version);
    }
    MPI_Finalize();
    return ok ? 0 : 1;
}
