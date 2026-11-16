// MPI+OpenMP Barnes-Hut: replicated global tree, distributed target particles.
//
// Every rank holds the positions and masses of all N particles, in particle-ID
// order. Rank r owns a contiguous ID range and integrates only those particles.
// One kick-drift-kick step is:
//
//   half kick   owned velocities, previous accelerations
//   drift       owned positions
//   exchange    MPI_Allgatherv of owned positions into the global array
//   build       identical global octree on every rank (serial, per rank)
//   traverse    OpenMP loop over owned targets, schedule(runtime)
//   half kick   owned velocities, new accelerations
//
// Replicated: positions (exchanged every step) and masses (broadcast once).
// Distributed: velocities and accelerations (gathered only for --dump/--energy).
//
// This is the communication baseline, not a distributed tree: every rank pays
// the full O(N log N) tree build and O(N) memory, which limits scaling. MPI is
// initialised with MPI_THREAD_FUNNELED and called only from the main thread,
// outside OpenMP parallel regions.

#include "../common/nbody.hpp"
#include "../common/run_support.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace nbody;

// Vec3 travels as three MPI_DOUBLEs; make that layout assumption explicit.
static_assert(std::is_standard_layout<Vec3>::value && sizeof(Vec3) == 3 * sizeof(double),
              "Vec3 must be exactly three contiguous doubles");

struct HybridOptions {
    std::string schedule = "static";  // static | dynamic | guided
    int chunk = 0;                    // 0: the OpenMP default for the schedule
};

// Balanced contiguous ownership: the first n % size ranks get one extra
// particle. Ranks beyond n own nothing.
struct Ownership {
    std::vector<int> count, offset;    // in particles
    std::vector<int> count3, offset3;  // in doubles, for Vec3 arrays
    Ownership(std::size_t n, int size) : count(size), offset(size), count3(size), offset3(size) {
        const std::size_t base = n / size, extra = n % size;
        std::size_t at = 0;
        for (int r = 0; r < size; ++r) {
            const std::size_t c = base + (static_cast<std::size_t>(r) < extra ? 1 : 0);
            count[r] = static_cast<int>(c);
            offset[r] = static_cast<int>(at);
            count3[r] = 3 * count[r];  // n <= INT_MAX / 3 is checked at parse time
            offset3[r] = 3 * offset[r];
            at += c;
        }
    }
};

// Each rank contributes its owned slice in place; zero-count ranks are fine.
void allgather_vec3(std::vector<Vec3> &global, const Ownership &own) {
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, global.data(), own.count3.data(),
                   own.offset3.data(), MPI_DOUBLE, MPI_COMM_WORLD);
}

omp_sched_t schedule_kind(const std::string &name) {
    if (name == "static") return omp_sched_static;
    if (name == "dynamic") return omp_sched_dynamic;
    return omp_sched_guided;
}

const char *schedule_name(omp_sched_t kind) {
    // Mask off the monotonic modifier bit OpenMP 4.5+ may report.
    switch (static_cast<int>(kind) & 0xff) {
        case omp_sched_static: return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided: return "guided";
        case omp_sched_auto: return "auto";
        default: return "other";
    }
}

// First line of the MPI library's self-description.
std::string mpi_library() {
    char version[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
    int len = 0;
    MPI_Get_library_version(version, &len);
    std::string s(version, static_cast<std::size_t>(len));
    return s.substr(0, s.find_first_of("\n,"));
}

std::string env_or(const char *name, const char *fallback) {
    const char *v = std::getenv(name);
    return v ? v : fallback;
}

void usage(const char *prog) {
    std::fprintf(stderr,
                 "usage: mpirun -np R %s %s\n"
                 "          [--schedule static|dynamic|guided] [--chunk C]\n"
                 "OpenMP threads per rank come from OMP_NUM_THREADS.\n",
                 prog, kCommonUsage);
}

// Parses on every rank. argv is identical everywhere, so every rank reaches
// the same verdict and all of them exit together without a hanging collective.
int parse(int argc, char **argv, RunOptions &opt, HybridOptions &hopt, int rank) {
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> const char * {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
                return argv[++i];
            };
            if (parse_common_option(arg, next, opt)) continue;
            if (arg == "--schedule") hopt.schedule = next();
            else if (arg == "--chunk") hopt.chunk = parse_int_in(arg, next(), 0, 1 << 20);
            else if (arg == "-h" || arg == "--help") { if (rank == 0) usage(argv[0]); return 0; }
            else throw std::invalid_argument("unknown option: " + arg);
        }
        check_common_options(opt);
        if (hopt.schedule != "static" && hopt.schedule != "dynamic" && hopt.schedule != "guided")
            throw std::invalid_argument("--schedule must be static, dynamic or guided");
        // Vec3 Allgatherv counts are 3 * n doubles in an int.
        if (opt.n > static_cast<std::size_t>(INT_MAX / 3))
            throw std::invalid_argument("--n too large for int MPI counts");
    } catch (const std::invalid_argument &e) {
        if (rank == 0) {
            std::fprintf(stderr, "%s\n", e.what());
            usage(argv[0]);
        }
        return 2;
    }
    return -1;  // continue
}

struct PhaseTimes {
    double integrate = 0.0, exchange = 0.0, build = 0.0, traverse = 0.0;
};

int run(int argc, char **argv, int provided) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    RunOptions opt;
    HybridOptions hopt;
    if (const int code = parse(argc, argv, opt, hopt, rank); code >= 0) return code;

    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            std::fprintf(stderr, "MPI provided thread level %d < MPI_THREAD_FUNNELED\n", provided);
        return 3;
    }
    omp_set_schedule(schedule_kind(hopt.schedule), hopt.chunk);
    // The team that actually runs can be smaller than requested (OMP_DYNAMIC,
    // thread limits), so record it alongside omp_get_max_threads().
    int team = 0;
#pragma omp parallel
    {
#pragma omp single
        team = omp_get_num_threads();
    }
    int team_min = 0;
    MPI_Allreduce(&team, &team_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    const std::size_t n = opt.n;
    const Ownership own(n, size);
    const std::size_t begin = own.offset[rank], end = begin + own.count[rank];

    // Rank 0 generates the deterministic initial state; everyone receives
    // positions, velocities and masses in ID order.
    const double t_setup = MPI_Wtime();
    Particles p;
    if (rank == 0) p = make_plummer(n, opt.seed);
    else p.resize(n);
    MPI_Bcast(p.pos.data(), static_cast<int>(3 * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(p.vel.data(), static_cast<int>(3 * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(p.mass.data(), static_cast<int>(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Octree tree;
    PhaseTimes t;
    auto evaluate = [&]() {
        double t0 = MPI_Wtime();
        tree.build(p, opt.leaf_size);
        t.build += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        const auto lo = static_cast<std::int32_t>(begin), hi = static_cast<std::int32_t>(end);
        const double theta = opt.theta;
#pragma omp parallel for schedule(runtime)
        for (std::int32_t i = lo; i < hi; ++i) p.acc[i] = tree.acceleration(i, p, theta);
        t.traverse += MPI_Wtime() - t0;
    };
    auto step = [&]() {
        double t0 = MPI_Wtime();
        kick(p, 0.5 * opt.dt, begin, end);
        drift(p, opt.dt, begin, end);
        t.integrate += MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        allgather_vec3(p.pos, own);
        t.exchange += MPI_Wtime() - t0;
        evaluate();
        t0 = MPI_Wtime();
        kick(p, 0.5 * opt.dt, begin, end);
        t.integrate += MPI_Wtime() - t0;
    };

    // Gather the distributed fields so rank 0 holds the full state.
    auto gather_state = [&]() {
        allgather_vec3(p.vel, own);
        allgather_vec3(p.acc, own);
    };

    evaluate();
    double energy_start = 0.0;
    if (opt.energy) {
        gather_state();
        if (rank == 0) energy_start = total_energy(p);
    }
    for (int s = 0; s < opt.warmup; ++s) step();
    const double setup_local = MPI_Wtime() - t_setup;

    // Timed region: synchronised start, per-rank elapsed time, MPI_MAX.
    t = PhaseTimes{};
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    for (int s = 0; s < opt.steps; ++s) step();
    const double wall_local = MPI_Wtime() - t0;

    // Phase statistics across ranks: max, and mean for the imbalance ratio.
    double local[6] = {wall_local, t.integrate, t.exchange, t.build, t.traverse, setup_local};
    // Peak memory per rank, before the validation-only gathers below. Every
    // rank holds all N positions, so the sum grows with the number of ranks.
    const double rss = peak_rss_mib();
    double rss_max = 0.0, rss_sum = 0.0;
    MPI_Reduce(&rss, &rss_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&rss, &rss_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double vmax[6], vsum[6], vmin[6];
    MPI_Reduce(local, vmax, 6, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, vsum, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(local, vmin, 6, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Replication check: a weighted checksum of the global positions and the
    // mass sum must be identical on every rank, and owned counts must sum to N.
    // This catches a missing or misplaced exchange; it is not a full bitwise
    // comparison (the dump-based comparison with serial does that).
    double sums[2] = {0.0, 0.0};
    for (std::size_t i = 0; i < n; ++i) {
        sums[0] += p.pos[i].x + 2.0 * p.pos[i].y + 3.0 * p.pos[i].z;
        sums[1] += p.mass[i];
    }
    double smin[2], smax[2];
    MPI_Allreduce(sums, smin, 2, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(sums, smax, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    long long owned = own.count[rank], owned_total = 0;
    MPI_Allreduce(&owned, &owned_total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    const bool replicated = smin[0] == smax[0] && smin[1] == smax[1] &&
                            owned_total == static_cast<long long>(n);

    // Distinct hosts, for the provenance record.
    char host[MPI_MAX_PROCESSOR_NAME] = {0};
    int host_len = 0;
    MPI_Get_processor_name(host, &host_len);
    std::vector<char> hosts(rank == 0 ? static_cast<std::size_t>(size) * MPI_MAX_PROCESSOR_NAME : 0);
    MPI_Gather(host, MPI_MAX_PROCESSOR_NAME, MPI_CHAR, hosts.data(), MPI_MAX_PROCESSOR_NAME,
               MPI_CHAR, 0, MPI_COMM_WORLD);

    double energy_end = 0.0;
    if (opt.energy || !opt.dump.empty()) gather_state();
    if (opt.energy && rank == 0) energy_end = total_energy(p);

    int exit_code = replicated ? 0 : 4;
    if (rank == 0) {
        std::vector<std::string> distinct;
        for (int r = 0; r < size; ++r) {
            const std::string h(&hosts[static_cast<std::size_t>(r) * MPI_MAX_PROCESSOR_NAME]);
            if (std::find(distinct.begin(), distinct.end(), h) == distinct.end()) distinct.push_back(h);
        }
        std::string host_list = "[";
        for (std::size_t k = 0; k < distinct.size(); ++k)
            host_list += (k ? "," : "") + json_string(distinct[k]);
        host_list += "]";

        omp_sched_t kind;
        int chunk = 0;
        omp_get_schedule(&kind, &chunk);
        const int threads = omp_get_max_threads();
        const double mean_traverse = vsum[4] / size;

        std::printf("ranks=%d threads=%d schedule=%s chunk=%d n=%zu steps=%d wall_s=%.4f "
                    "build_s=%.4f traverse_s=%.4f exchange_s=%.4f integrate_s=%.4f "
                    "traverse_imbalance=%.3f replicated=%s\n",
                    size, threads, schedule_name(kind), chunk, n, opt.steps, vmax[0], vmax[3],
                    vmax[4], vmax[2], vmax[1], mean_traverse > 0 ? vmax[4] / mean_traverse : 1.0,
                    replicated ? "yes" : "NO");
        char energy[160] = "";
        if (opt.energy) {
            const double drift_rel = std::abs(energy_start) > 0.0
                                         ? std::abs((energy_end - energy_start) / energy_start)
                                         : 0.0;
            std::printf("energy_start=%.9e energy_end=%.9e rel_drift=%.3e\n", energy_start,
                        energy_end, drift_rel);
            std::snprintf(energy, sizeof energy,
                          ",\"energy_start\":%.17g,\"energy_end\":%.17g,\"energy_rel_drift\":%.6e",
                          energy_start, energy_end, drift_rel);
        }
        std::printf(
            "result {\"program\":\"bh_hybrid\",\"method\":\"bh\",\"n\":%zu,\"theta\":%.17g,"
            "\"dt\":%.17g,\"steps\":%d,\"warmup\":%d,\"leaf_size\":%d,\"seed\":%llu,"
            "\"softening\":%.17g,\"ranks\":%d,\"threads\":%d,\"threads_effective\":%d,"
            "\"schedule_requested\":%s,\"chunk_requested\":%d,"
            "\"schedule_effective\":\"%s\",\"chunk_effective\":%d,"
            "\"mpi_thread_provided\":%d,\"omp_proc_bind\":%s,\"omp_places\":%s,"
            "\"hosts\":%s,\"nodes\":%zu,\"setup_s\":%.6f,"
            "\"wall_s\":%.6f,\"integrate_s\":%.6f,\"exchange_s\":%.6f,\"build_s\":%.6f,"
            "\"traverse_s\":%.6f,\"traverse_mean_s\":%.6f,\"traverse_min_s\":%.6f,"
            "\"wall_min_s\":%.6f,\"replicated\":%s,\"mpi_library\":%s,"
            "\"peak_rss_mib_rank_max\":%.1f,\"peak_rss_mib_rank_sum\":%.1f%s%s}\n",
            n, opt.theta, opt.dt, opt.steps, opt.warmup, opt.leaf_size,
            static_cast<unsigned long long>(opt.seed), kSoftening, size, threads, team_min,
            json_string(hopt.schedule).c_str(), hopt.chunk, schedule_name(kind), chunk, provided,
            json_string(env_or("OMP_PROC_BIND", "")).c_str(),
            json_string(env_or("OMP_PLACES", "")).c_str(), host_list.c_str(), tree.node_count(),
            vmax[5], vmax[0], vmax[1], vmax[2], vmax[3], vmax[4], mean_traverse, vmin[4], vmin[0],
            replicated ? "true" : "false", json_string(mpi_library()).c_str(), rss_max, rss_sum,
            energy, build_json().c_str());
        if (!replicated) std::fprintf(stderr, "replication check FAILED\n");
        if (!opt.dump.empty()) {
            try {
                write_dump(opt.dump, p);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "%s\n", e.what());
                exit_code = 5;
            }
        }
    }
    // Every rank returns rank 0's verdict.
    MPI_Bcast(&exit_code, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exit_code;
}

}  // namespace

int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int code = 0;
    try {
        code = run(argc, argv, provided);
    } catch (const std::exception &e) {
        // A failure on one rank (e.g. allocation) cannot be agreed collectively
        // without risking a hang; abort the whole job instead.
        std::fprintf(stderr, "fatal: %s\n", e.what());
        MPI_Abort(MPI_COMM_WORLD, 6);
    }
    MPI_Finalize();
    return code;
}
