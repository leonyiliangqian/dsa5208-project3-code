// Inter-node MPI probe. Ranks are expected to be spread over hosts (e.g.
// mpirun --map-by node). Each rank r exchanges an 8 MiB patterned buffer with
// partner (r + size/2) % size and verifies every element, then all ranks run
// an MPI_Allgatherv with the solver's uneven contiguous layout and verify it.
// With INTERNODE_EXPECT_HOSTS=k the probe also requires exactly k distinct
// hosts and, for k > 1, that every exchange pair spans two hosts.
#include <mpi.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char host[64] = {0};
    gethostname(host, sizeof host - 1);
    std::vector<char> hosts(static_cast<std::size_t>(size) * 64);
    MPI_Allgather(host, 64, MPI_CHAR, hosts.data(), 64, MPI_CHAR, MPI_COMM_WORLD);
    auto host_of = [&](int r) { return std::string(&hosts[static_cast<std::size_t>(r) * 64]); };

    int ok = 1;
    // Ring exchange with offset h: send to r+h, receive from r-h. h is the
    // smallest offset for which every rank's partner is on another host, so
    // the test is inter-node whatever the rank mapping; every rank computes the
    // same h from the same gathered host list.
    int h = 1;
    for (int k = 1; k < size; ++k) {
        bool all_remote = true;
        for (int r = 0; r < size && all_remote; ++r) all_remote = host_of((r + k) % size) != host_of(r);
        if (all_remote) { h = k; break; }
    }
    const int to = (rank + h) % size, from = (rank - h + size) % size;
    const std::size_t count = (8u << 20) / sizeof(double);
    std::vector<double> send(count), recv(count, -1.0);
    for (std::size_t i = 0; i < count; ++i) send[i] = rank * 1e7 + static_cast<double>(i);
    double t0 = MPI_Wtime();
    MPI_Sendrecv(send.data(), static_cast<int>(count), MPI_DOUBLE, to, 7, recv.data(),
                 static_cast<int>(count), MPI_DOUBLE, from, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    const double exchange_s = MPI_Wtime() - t0;
    for (std::size_t i = 0; i < count && ok; ++i)
        if (recv[i] != from * 1e7 + static_cast<double>(i)) ok = 0;
    const bool cross_host = host_of(to) != host_of(rank) && host_of(from) != host_of(rank);

    // Allgatherv with the solver's balanced uneven ownership of n particles.
    const int n = 1000003;
    std::vector<int> counts(size), displs(size);
    for (int r = 0, at = 0; r < size; ++r) {
        counts[r] = 3 * (n / size + (r < n % size ? 1 : 0));
        displs[r] = at;
        at += counts[r];
    }
    std::vector<double> global(3 * static_cast<std::size_t>(n), -1.0);
    for (int k = 0; k < counts[rank]; ++k) global[displs[rank] + k] = displs[rank] + k;
    t0 = MPI_Wtime();
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global.data(), counts.data(), displs.data(),
                   MPI_DOUBLE, MPI_COMM_WORLD);
    const double allgather_s = MPI_Wtime() - t0;
    for (std::size_t k = 0; k < global.size() && ok; ++k)
        if (global[k] != static_cast<double>(k)) ok = 0;

    int all_ok = 0, all_cross = 0;
    int cross = cross_host ? 1 : 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&cross, &all_cross, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    double times[2] = {exchange_s, allgather_s}, tmax[2];
    MPI_Reduce(times, tmax, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int pass = all_ok;
    if (rank == 0) {
        std::set<std::string> distinct;
        for (int r = 0; r < size; ++r) distinct.insert(host_of(r));
        const char *expect = std::getenv("INTERNODE_EXPECT_HOSTS");
        const int want = expect ? std::atoi(expect) : 0;
        if (want > 0 && static_cast<int>(distinct.size()) != want) pass = 0;
        if (want > 1 && !all_cross) pass = 0;
        std::string list;
        for (const auto &name : distinct) list += (list.empty() ? "" : ",") + name;
        std::printf("internode_check ranks=%d hosts=%zu [%s] offset=%d pairs_cross_host=%d "
                    "data_ok=%d sendrecv_8MiB_max_s=%.4f allgatherv_%dx3_max_s=%.4f "
                    "expect_hosts=%d %s\n",
                    size, distinct.size(), list.c_str(), h, all_cross, all_ok, tmax[0], n,
                    tmax[1], want, pass ? "PASS" : "FAIL");
    }
    MPI_Bcast(&pass, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return pass ? 0 : 1;
}
