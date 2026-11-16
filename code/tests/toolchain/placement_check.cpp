// Placement probe: every OpenMP thread of every rank reports its host, the CPU
// it is running on, and how many CPUs its affinity mask allows. Rank 0 prints
// the map and checks, per host, that
//   - no two threads were observed on the same CPU (no oversubscription), and
//   - with binding requested (PLACEMENT_EXPECT_BOUND=1), each thread's mask is
//     a single core's worth of CPUs, not the whole node.
// With PLACEMENT_TRACE=1 every rank also writes flushed, rank-labelled
// "trace" lines to stderr at each stage (process start with the OS affinity
// mask, after MPI_Init, OpenMP team start/end, around each collective and
// MPI_Finalize), so a stalled run shows how far each rank got even when it
// is killed. Timestamps are wall-clock seconds, comparable across nodes.
// Linux only (sched_getcpu, sched_getaffinity).
#include <mpi.h>
#include <omp.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

struct Sample {
    int rank, thread, cpu, mask_cpus;
    char host[64];
};

bool tracing = false;
int trace_rank = -1;  // launcher's rank id until MPI_Init, then the MPI rank
char trace_host[64] = {0};

// One complete line per call, flushed, so it survives the process being killed.
void trace(const char *stage, const std::string &detail = "") {
    if (!tracing) return;
    timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    std::fprintf(stderr, "trace t=%ld.%03ld host=%s rank=%d pid=%d stage=%s %s\n",
                 static_cast<long>(ts.tv_sec), ts.tv_nsec / 1000000, trace_host, trace_rank,
                 static_cast<int>(getpid()), stage, detail.c_str());
    std::fflush(stderr);
}

std::string cpu_list(const cpu_set_t &mask) {
    std::string out;
    for (int c = 0; c < CPU_SETSIZE; ++c)
        if (CPU_ISSET(c, &mask)) out += (out.empty() ? "" : ",") + std::to_string(c);
    return out;
}

// Spin briefly so threads are actually running concurrently when sampled.
void busy(double seconds) {
    const double t0 = omp_get_wtime();
    volatile double x = 0.0;
    while (omp_get_wtime() - t0 < seconds) x += 1.0;
}

}  // namespace

int main(int argc, char **argv) {
    // Result lines go to a pipe under mpirun; flush each one.
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    const char *trace_env = std::getenv("PLACEMENT_TRACE");
    tracing = trace_env && std::string(trace_env) == "1";
    char host[64] = {0};
    gethostname(host, sizeof host - 1);
    std::snprintf(trace_host, sizeof trace_host, "%s", host);
    if (const char *r = std::getenv("OMPI_COMM_WORLD_RANK")) trace_rank = std::atoi(r);
    {
        cpu_set_t start_mask;
        CPU_ZERO(&start_mask);
        sched_getaffinity(0, sizeof start_mask, &start_mask);
        trace("start", "affinity=" + cpu_list(start_mask) + " omp_num_threads=" +
                           (std::getenv("OMP_NUM_THREADS") ? std::getenv("OMP_NUM_THREADS") : "unset"));
    }

    int provided = 0;
    trace("mpi_init_begin");
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    trace_rank = rank;
    trace("mpi_init_done", "size=" + std::to_string(size) + " provided=" + std::to_string(provided));

    // Every rank must contribute the same number of samples to the gather;
    // agree on that collectively so a mismatch fails on all ranks alike.
    const int threads = omp_get_max_threads();
    int tmin = 0, tmax = 0;
    trace("allreduce_threads_begin", "threads=" + std::to_string(threads));
    MPI_Allreduce(&threads, &tmin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&threads, &tmax, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    trace("allreduce_threads_done", "min=" + std::to_string(tmin) + " max=" + std::to_string(tmax));
    if (tmin != tmax) {
        if (rank == 0)
            std::printf("placement_check threads differ across ranks (min=%d max=%d) FAIL\n",
                        tmin, tmax);
        MPI_Finalize();
        return 1;
    }

    std::vector<Sample> mine(threads);
    trace("omp_begin");
#pragma omp parallel num_threads(threads)
    {
        if (omp_get_thread_num() == 0)
            trace("omp_team_started", "team=" + std::to_string(omp_get_num_threads()));
        busy(0.2);
        cpu_set_t mask;
        CPU_ZERO(&mask);
        sched_getaffinity(0, sizeof mask, &mask);
        Sample &s = mine[omp_get_thread_num()];
        s.rank = rank;
        s.thread = omp_get_thread_num();
        s.cpu = sched_getcpu();
        s.mask_cpus = CPU_COUNT(&mask);
        std::snprintf(s.host, sizeof s.host, "%s", host);
        busy(0.2);
    }
    {
        std::string cpus;
        for (const Sample &x : mine) cpus += (cpus.empty() ? "" : ",") + std::to_string(x.cpu);
        trace("omp_done", "thread_cpus=" + cpus);
    }

    const int bytes = static_cast<int>(threads * sizeof(Sample));
    std::vector<Sample> all(rank == 0 ? static_cast<std::size_t>(size) * threads : 0);
    trace("gather_begin");
    MPI_Gather(mine.data(), bytes, MPI_BYTE, all.data(), bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    trace("gather_done");

    int ok = 1;
    if (rank == 0) {
        const char *expect = std::getenv("PLACEMENT_EXPECT_BOUND");
        const bool expect_bound = expect && std::string(expect) == "1";
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        std::map<std::string, std::set<int>> cpus_per_host;
        std::map<std::string, int> threads_per_host;
        int max_mask = 0;
        for (const Sample &s : all) {
            std::printf("placement host=%s rank=%d thread=%d cpu=%d mask_cpus=%d\n", s.host, s.rank,
                        s.thread, s.cpu, s.mask_cpus);
            cpus_per_host[s.host].insert(s.cpu);
            ++threads_per_host[s.host];
            if (s.mask_cpus > max_mask) max_mask = s.mask_cpus;
        }
        for (const auto &h : threads_per_host) {
            const int distinct = static_cast<int>(cpus_per_host[h.first].size());
            if (distinct != h.second) ok = 0;
            std::printf("placement_host host=%s threads=%d distinct_cpus=%d\n", h.first.c_str(),
                        h.second, distinct);
        }
        // A bound thread may own one core's hardware threads (SMT), never more
        // than a few CPUs; an unbound one sees the whole cpuset.
        if (expect_bound && max_mask > 2) ok = 0;
        // Multi-node jobs: ranks must actually be spread over the expected hosts.
        const char *hosts_env = std::getenv("PLACEMENT_EXPECT_HOSTS");
        const int want_hosts = hosts_env ? std::atoi(hosts_env) : 0;
        if (want_hosts > 0 && static_cast<int>(threads_per_host.size()) != want_hosts) ok = 0;
        std::printf("placement_hosts distinct=%zu expected=%d\n", threads_per_host.size(), want_hosts);
        std::printf("placement_check ranks=%d threads=%d online_cpus=%ld max_mask_cpus=%d "
                    "expect_bound=%d %s\n",
                    size, threads, online, max_mask, expect_bound ? 1 : 0, ok ? "PASS" : "FAIL");
    }
    trace("bcast_begin");
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    trace("finalize_begin", std::string("verdict=") + (ok ? "PASS" : "FAIL"));
    MPI_Finalize();
    trace("finalize_done");
    return ok ? 0 : 1;
}
