#ifndef _GNU_SOURCE
#define _GNU_SOURCE  // sched_getcpu
#endif
// Barnes-Hut force traversal offloaded with OpenMP target.
//
// The tree is built on the host with the same Octree as the serial and hybrid
// solvers, flattened into plain arrays, and traversed by one of two backends
// compiled from one kernel:
//   device  target teams distribute parallel for, on the default device
//   cpu     parallel for schedule(runtime), on host threads
// so a GPU run and its CPU comparison share a compiler, kernel and tree. The
// serial and hybrid solvers are not changed and remain the CPU baselines.
//
// A compiler without offload support runs target regions on the host. That is
// useful for testing the logic and says nothing about a GPU, so every record
// states where the kernel ran, and --require-device exits 7 unless it ran on a
// device. Run with OMP_TARGET_OFFLOAD=MANDATORY on a GPU node as well.

#ifndef __VERSION__
#define __VERSION__ "unknown"
#endif

#include "../common/nbody.hpp"
#include "../common/run_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif
#include <unistd.h>

namespace {

using namespace nbody;

static_assert(sizeof(Vec3) == 3 * sizeof(double), "Vec3 is mapped as three doubles");

// Fields the traversal reads. Host-only fields of Node are left out so they are
// not transferred.
struct DevNode {
    double com[3];
    double mass;
    double half;
    std::int32_t child[8];
    std::int32_t first;
    std::int32_t count;
    std::int32_t leaf;
};

constexpr int kStackSize = 8 * (Octree::kMaxDepth + 1);

#pragma omp declare target
// Octree::acceleration on plain arrays: same acceptance test, visiting order
// and arithmetic, so the two can be compared particle by particle.
inline void traverse_one(std::int32_t i, const DevNode *nodes, const std::int32_t *order,
                         std::int32_t slot_i, const double *pos, const double *mass,
                         double theta_sq, double softening_sq, double *out) {
    const double xi = pos[3 * i], yi = pos[3 * i + 1], zi = pos[3 * i + 2];
    double ax = 0.0, ay = 0.0, az = 0.0;
    std::int32_t stack[kStackSize];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const DevNode &node = nodes[stack[--top]];
        const double dx = node.com[0] - xi, dy = node.com[1] - yi, dz = node.com[2] - zi;
        const double d2 = dx * dx + dy * dy + dz * dz;
        const double s = 2.0 * node.half;
        const bool holds_i = slot_i >= node.first && slot_i < node.first + node.count;
        if (!holds_i && s * s < theta_sq * d2) {
            const double r2 = d2 + softening_sq;
            const double f = node.mass / (r2 * sqrt(r2));
            ax += dx * f;
            ay += dy * f;
            az += dz * f;
        } else if (node.leaf) {
            for (std::int32_t k = node.first; k < node.first + node.count; ++k) {
                const std::int32_t j = order[k];
                if (j != i) {
                    const double ex = pos[3 * j] - xi, ey = pos[3 * j + 1] - yi, ez = pos[3 * j + 2] - zi;
                    const double r2 = ex * ex + ey * ey + ez * ez + softening_sq;
                    const double f = mass[j] / (r2 * sqrt(r2));
                    ax += ex * f;
                    ay += ey * f;
                    az += ez * f;
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                if (node.child[c] >= 0) stack[top++] = node.child[c];
            }
        }
    }
    out[3 * i] = ax;
    out[3 * i + 1] = ay;
    out[3 * i + 2] = az;
}
#pragma omp end declare target

double seconds_since(const std::chrono::steady_clock::time_point &t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

enum class Backend { device, cpu };

// Where target regions really run, measured rather than assumed.
struct DeviceInfo {
    int num_devices = 0;
    bool on_device = false;
    bool separate_memory = false;  // false means transfer times mean nothing
};

DeviceInfo probe_device() {
    DeviceInfo info;
    info.num_devices = omp_get_num_devices();
    int initial = 1;
#pragma omp target map(from : initial)
    initial = omp_is_initial_device();
    info.on_device = (initial == 0);
    // A write made inside the region without a map back must not reach the host.
    int value = 1;
#pragma omp target enter data map(to : value)
#pragma omp target map(alloc : value)
    value = 2;
    info.separate_memory = (value == 1);
#pragma omp target exit data map(delete : value)
    return info;
}

// The host team as it really runs. Every thread records its CPU after the single's
// barrier, so a team stacked on fewer CPUs shows up here; -1 means not measured.
struct TeamInfo {
    int team = 0, distinct_cpus = -1;
};

TeamInfo probe_team() {
    TeamInfo info;
    std::vector<int> cpus(omp_get_max_threads(), -1);
#pragma omp parallel
    {
#pragma omp single
        info.team = omp_get_num_threads();
#ifdef __linux__
        cpus[omp_get_thread_num()] = sched_getcpu();
#pragma omp barrier
#endif
    }
#ifdef __linux__
    cpus.resize(info.team);
    std::sort(cpus.begin(), cpus.end());
    info.distinct_cpus = static_cast<int>(std::unique(cpus.begin(), cpus.end()) - cpus.begin());
#endif
    return info;
}

const char *proc_bind_name() {
    switch (omp_get_proc_bind()) {
        case omp_proc_bind_false: return "false";
        case omp_proc_bind_close: return "close";
        case omp_proc_bind_spread: return "spread";
        default: return "other";
    }
}

const char *schedule_name(omp_sched_t kind) {
    switch (static_cast<int>(kind) & 0x7fffffff) {
        case omp_sched_static: return "static";
        case omp_sched_dynamic: return "dynamic";
        case omp_sched_guided: return "guided";
        case omp_sched_auto: return "auto";
        default: return "unknown";
    }
}

struct Timers {
    double flatten = 0.0, map_nodes = 0.0, to_device = 0.0, kernel = 0.0, from_device = 0.0;
};

// Device copies of the arrays, kept mapped across steps so transfers can be
// timed apart from the kernel. Particles must keep their size and storage while
// an evaluator exists.
class ForceEvaluator {
  public:
    ForceEvaluator(Particles &p, Backend backend) : p_(p), backend_(backend), n_(p.size()) {
        order_.resize(n_);
        slot_.resize(n_);
        if (backend_ == Backend::device && n_ > 0) {
            [[maybe_unused]] double *pos = pos_ptr(), *acc = acc_ptr(), *mass = p_.mass.data();
            [[maybe_unused]] std::int32_t *order = order_.data(), *slot = slot_.data();
            [[maybe_unused]] const std::size_t n = n_, n3 = 3 * n_;
#pragma omp target enter data map(alloc : pos[0 : n3], acc[0 : n3], order[0 : n], slot[0 : n]) \
    map(to : mass[0 : n])
        }
    }

    ~ForceEvaluator() {
        if (backend_ == Backend::device && n_ > 0) {
            release_nodes();
            [[maybe_unused]] double *pos = pos_ptr(), *acc = acc_ptr(), *mass = p_.mass.data();
            [[maybe_unused]] std::int32_t *order = order_.data(), *slot = slot_.data();
            [[maybe_unused]] const std::size_t n = n_, n3 = 3 * n_;
#pragma omp target exit data map(delete : pos[0 : n3], acc[0 : n3], order[0 : n], slot[0 : n], \
                                 mass[0 : n])
        }
    }

    ForceEvaluator(const ForceEvaluator &) = delete;
    ForceEvaluator &operator=(const ForceEvaluator &) = delete;

    void evaluate(const Octree &tree, double theta, Timers &t) {
        if (n_ == 0) return;
        const auto count_i = static_cast<std::int32_t>(n_);
        const double theta_sq = theta * theta, softening_sq = kSofteningSq;

        auto t0 = std::chrono::steady_clock::now();
        const std::size_t count = flatten(tree);
        t.flatten += seconds_since(t0);
        double *pos = pos_ptr(), *acc = acc_ptr(), *mass = p_.mass.data();
        std::int32_t *order = order_.data(), *slot = slot_.data();

        if (backend_ == Backend::cpu) {
            const DevNode *nodes = nodes_.data();
            t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(runtime)
            for (std::int32_t i = 0; i < count_i; ++i)
                traverse_one(i, nodes, order, slot[i], pos, mass, theta_sq, softening_sq, acc);
            t.kernel += seconds_since(t0);
            return;
        }

        // Remapping when the tree outgrows its device copy is inside the timed region.
        t0 = std::chrono::steady_clock::now();
        reserve_nodes(count);
        t.map_nodes += seconds_since(t0);
        DevNode *nodes = dev_nodes_;
        const std::size_t n = n_, n3 = 3 * n_, cap = capacity_;
        t0 = std::chrono::steady_clock::now();
#pragma omp target update to(nodes[0 : count], order[0 : n], slot[0 : n], pos[0 : n3])
        t.to_device += seconds_since(t0);

        t0 = std::chrono::steady_clock::now();
#pragma omp target teams distribute parallel for \
    map(alloc : nodes[0 : cap], order[0 : n], slot[0 : n], pos[0 : n3], mass[0 : n], acc[0 : n3])
        for (std::int32_t i = 0; i < count_i; ++i)
            traverse_one(i, nodes, order, slot[i], pos, mass, theta_sq, softening_sq, acc);
        t.kernel += seconds_since(t0);

        t0 = std::chrono::steady_clock::now();
#pragma omp target update from(acc[0 : n3])
        t.from_device += seconds_since(t0);
    }

  private:
    double *pos_ptr() { return reinterpret_cast<double *>(p_.pos.data()); }
    double *acc_ptr() { return reinterpret_cast<double *>(p_.acc.data()); }

    std::size_t flatten(const Octree &tree) {
        const std::vector<Node> &src = tree.nodes();
        nodes_.resize(src.size());
        for (std::size_t k = 0; k < src.size(); ++k) {
            const Node &a = src[k];
            DevNode &b = nodes_[k];
            b.com[0] = a.com.x;
            b.com[1] = a.com.y;
            b.com[2] = a.com.z;
            b.mass = a.mass;
            b.half = a.half;
            std::copy(std::begin(a.child), std::end(a.child), b.child);
            b.first = a.first;
            b.count = a.count;
            b.leaf = a.leaf ? 1 : 0;
        }
        const std::vector<std::int32_t> &ord = tree.order();
        std::copy(ord.begin(), ord.end(), order_.begin());
        for (std::size_t k = 0; k < n_; ++k) slot_[order_[k]] = static_cast<std::int32_t>(k);
        return src.size();
    }

    // The node count changes between steps. The mapping grows when needed, and
    // always matches the host storage that dev_nodes_ points at.
    void reserve_nodes(std::size_t count) {
        if (count <= capacity_ && dev_nodes_ == nodes_.data()) return;
        release_nodes();
        if (nodes_.capacity() < count + count / 2) nodes_.reserve(count + count / 2);
        capacity_ = nodes_.capacity();
        dev_nodes_ = nodes_.data();
        [[maybe_unused]] DevNode *nodes = dev_nodes_;
        [[maybe_unused]] const std::size_t cap = capacity_;
#pragma omp target enter data map(alloc : nodes[0 : cap])
    }

    void release_nodes() {
        if (!dev_nodes_) return;
        [[maybe_unused]] DevNode *nodes = dev_nodes_;
        [[maybe_unused]] const std::size_t cap = capacity_;
#pragma omp target exit data map(delete : nodes[0 : cap])
        dev_nodes_ = nullptr;
        capacity_ = 0;
    }

    Particles &p_;
    Backend backend_;
    std::size_t n_;
    std::vector<DevNode> nodes_;
    std::vector<std::int32_t> order_, slot_;
    DevNode *dev_nodes_ = nullptr;
    std::size_t capacity_ = 0;
};

// ---------------------------------------------------------------------------
// Reference checks
// ---------------------------------------------------------------------------

// Both limits are fixed before any device run. kAgree: the same operations in the
// same order on the same tree, so only rounding and fused multiply-add can differ.
// kDirect: the serial gate's limit, as the summation order differs from direct.
constexpr double kAgree = 1e-12, kDirect = 1e-10, kFloor = 1e-13;

double worst_error(const std::vector<Vec3> &got, const std::vector<Vec3> &want, double rel) {
    double scale = 0.0;
    for (const Vec3 &w : want) scale = std::max(scale, std::sqrt(norm2(w)));
    double worst = 0.0;  // in units of the allowed error; above 1 fails
    for (std::size_t i = 0; i < want.size(); ++i) {
        const double err = std::sqrt(norm2(got[i] - want[i]));
        const double allowed = rel * std::sqrt(norm2(want[i])) + kFloor * scale;
        if (allowed > 0.0) worst = std::max(worst, err / allowed);
        else if (err > 0.0) worst = HUGE_VAL;
        if (!std::isfinite(norm2(got[i]))) worst = HUGE_VAL;
    }
    return worst;
}

Particles with_extra(std::size_t n, std::uint64_t seed, std::size_t extra, const Vec3 &where,
                     double offset) {
    Particles p = make_plummer(n, seed);
    const double m = n > 0 ? p.mass[0] : 1.0;
    for (std::size_t k = 0; k < extra; ++k) {
        p.pos.push_back({where.x + offset * static_cast<double>(k), where.y, where.z});
        p.vel.push_back({});
        p.acc.push_back({});
        p.mass.push_back(m);
    }
    return p;
}

int run_self_test(Backend backend) {
    struct Case {
        const char *name;
        Particles p;
    };
    std::vector<Case> cases;
    cases.push_back({"singleton", with_extra(0, 1, 1, {0.3, -0.2, 0.1}, 0.0)});
    cases.push_back({"all_coincident", with_extra(0, 1, 100, {1.0, 1.0, 1.0}, 0.0)});
    cases.push_back({"coincident_cluster", with_extra(500, 7, 50, {0.1, 0.2, 0.3}, 0.0)});
    cases.push_back({"below_max_depth", with_extra(200, 9, 5, {0.0, 0.0, 0.0}, 1e-25)});
    cases.push_back({"plummer", make_plummer(2000, 3)});
    {
        Particles p = make_plummer(1000, 11);
        std::mt19937_64 rng(11);
        std::uniform_real_distribution<double> u(0.1, 10.0);
        for (double &m : p.mass) m = u(rng) / 1000.0;
        cases.push_back({"unequal_masses", std::move(p)});
    }
    {
        // Many particles exactly on cell dividing planes.
        Particles p;
        for (int x = -2; x <= 2; ++x)
            for (int y = -2; y <= 2; ++y)
                for (int z = -2; z <= 2; ++z) {
                    p.pos.push_back({double(x), double(y), double(z)});
                    p.vel.push_back({});
                    p.acc.push_back({});
                    p.mass.push_back(1.0 / 125.0);
                }
        cases.push_back({"lattice_on_planes", std::move(p)});
    }

    int failures = 0;
    auto report = [&](const std::string &name, double worst) {
        const bool ok = worst <= 1.0;
        std::printf("self_test case=%s worst=%.3g %s\n", name.c_str(), worst, ok ? "PASS" : "FAIL");
        failures += ok ? 0 : 1;
    };

    for (Case &c : cases) {
        for (const int leaf_size : {1, 8}) {
            Particles &p = c.p;
            const std::string id = std::string(c.name) + " leaf=" + std::to_string(leaf_size);
            Octree tree;
            tree.build(p, leaf_size);
            ForceEvaluator eval(p, backend);
            Timers t;
            const auto n = static_cast<std::int32_t>(p.size());

            // theta = 0 opens every cell: must reproduce direct summation.
            std::vector<Vec3> exact(p.size());
            for (std::size_t i = 0; i < p.size(); ++i) exact[i] = direct_accel(i, p);
            eval.evaluate(tree, 0.0, t);
            report(id + " theta=0 vs direct", worst_error(p.acc, exact, kDirect));

            // Above 1/sqrt(3) a cell holding the target can pass the test, so the
            // large values exercise the rule that such a cell is always opened.
            for (const double theta : {0.5, 0.9, 2.0}) {
                std::vector<Vec3> want(p.size());
                for (std::int32_t i = 0; i < n; ++i) want[i] = tree.acceleration(i, p, theta);
                eval.evaluate(tree, theta, t);
                report(id + " theta=" + std::to_string(theta).substr(0, 3) + " vs host tree",
                       worst_error(p.acc, want, kAgree));
            }
        }
    }

    // Closed-form softened force on two particles. theta = 10 lets the root pass
    // the acceptance test from both, so it must be opened because it holds them.
    {
        Particles p;
        p.pos = {{0.1, 0.2, 0.3}, {0.6, 0.2, 0.3}};
        p.vel.assign(2, Vec3{});
        p.acc.assign(2, Vec3{});
        p.mass = {0.3, 0.7};
        const double r = 0.5, denom = std::pow(r * r + kSofteningSq, 1.5);
        const std::vector<Vec3> want = {{p.mass[1] * r / denom, 0.0, 0.0},
                                        {-p.mass[0] * r / denom, 0.0, 0.0}};
        Octree tree;
        tree.build(p, 1);
        ForceEvaluator eval(p, backend);
        Timers t;
        eval.evaluate(tree, 10.0, t);
        report("two_particles closed_form", worst_error(p.acc, want, kAgree));
    }

    // Reused device copies: move the particles and change the tree size between
    // evaluations, so stale data or a mapping that failed to grow gives wrong forces.
    {
        Particles p = with_extra(0, 1, 3000, {0.0, 0.0, 0.0}, 0.0);
        for (std::size_t k = 0; k < 2900; ++k) p.pos[k] = {1.0 + 1e-3 * double(k % 7), 1.0, 1.0};
        ForceEvaluator eval(p, backend);
        Timers t;
        Octree tree;
        tree.build(p, 1);
        const std::size_t small_nodes = tree.node_count();
        eval.evaluate(tree, 0.5, t);

        const Particles spread = make_plummer(p.size(), 21);
        p.pos = spread.pos;  // same size, so the storage and mapping stay valid
        tree.build(p, 1);
        const std::size_t large_nodes = tree.node_count();
        std::vector<Vec3> want(p.size());
        for (std::int32_t i = 0; i < static_cast<std::int32_t>(p.size()); ++i)
            want[i] = tree.acceleration(i, p, 0.5);
        eval.evaluate(tree, 0.5, t);
        report("reuse grow nodes " + std::to_string(small_nodes) + "->" + std::to_string(large_nodes),
               large_nodes > small_nodes ? worst_error(p.acc, want, kAgree) : HUGE_VAL);
    }

    std::printf("self_test failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}

void usage(const char *prog) {
    std::fprintf(stderr,
                 "usage: %s %s\n"
                 "          [--backend device|cpu] [--require-device] [--device-check]\n"
                 "          [--self-test] [--validate]\n",
                 prog, kCommonUsage);
}

}  // namespace

int main(int argc, char **argv) {
    RunOptions opt;
    Backend backend = Backend::device;
    bool self_test = false, validate = false, require_device = false, device_check = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> const char * {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
                return argv[++i];
            };
            if (parse_common_option(arg, next, opt)) continue;
            if (arg == "--backend") {
                const std::string b = next();
                if (b == "device") backend = Backend::device;
                else if (b == "cpu") backend = Backend::cpu;
                else throw std::invalid_argument("--backend must be device or cpu");
            } else if (arg == "--self-test") self_test = true;
            else if (arg == "--validate") validate = true;
            else if (arg == "--require-device") require_device = true;
            else if (arg == "--device-check") device_check = true;
            else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
            else throw std::invalid_argument("unknown option: " + arg);
        }
        check_common_options(opt);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "%s\n", e.what());
        usage(argv[0]);
        return 2;
    }

    const DeviceInfo device = probe_device();
    std::printf("device_check num_devices=%d kernel_on_device=%d separate_memory=%d\n",
                device.num_devices, device.on_device ? 1 : 0, device.separate_memory ? 1 : 0);
    if (require_device && !(device.on_device && device.separate_memory)) {
        std::fprintf(stderr, "FATAL: target regions did not run on a device with separate memory\n");
        return 7;
    }
    if (device_check) return 0;
    if (self_test) return run_self_test(backend);

    const TeamInfo team = probe_team();
    const auto t_setup = std::chrono::steady_clock::now();
    Particles p = make_plummer(opt.n, opt.seed);
    Octree tree;
    ForceEvaluator eval(p, backend);
    Timers t;

    if (validate) {
        // One state, three evaluators: this backend, the host tree and direct summation.
        tree.build(p, opt.leaf_size);
        std::vector<Vec3> host(p.size()), exact(p.size());
        for (std::int32_t i = 0; i < static_cast<std::int32_t>(p.size()); ++i)
            host[i] = tree.acceleration(i, p, opt.theta);
        for (std::size_t i = 0; i < p.size(); ++i) exact[i] = direct_accel(i, p);
        eval.evaluate(tree, opt.theta, t);
        const double agree = worst_error(p.acc, host, kAgree);
        const ErrorStats vs_direct = compare(p.acc, exact), host_direct = compare(host, exact);
        std::size_t identical = 0;
        for (std::size_t i = 0; i < p.size(); ++i) {
            if (p.acc[i].x == host[i].x && p.acc[i].y == host[i].y && p.acc[i].z == host[i].z) ++identical;
        }
        const bool ok = agree <= 1.0;
        std::printf("validate_gpu n=%zu theta=%.3f leaf_size=%d agree_worst=%.3g bitwise_identical=%zu "
                    "max_rel_err_vs_direct=%.3e host_tree_max_rel_err_vs_direct=%.3e %s\n",
                    opt.n, opt.theta, opt.leaf_size, agree, identical, vs_direct.max_rel,
                    host_direct.max_rel, ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    const std::size_t n = p.size();
    double integrate_s = 0.0, build_s = 0.0;
    auto evaluate = [&]() {
        const auto t0 = std::chrono::steady_clock::now();
        tree.build(p, opt.leaf_size);
        build_s += seconds_since(t0);
        eval.evaluate(tree, opt.theta, t);
    };
    auto step = [&]() {
        auto t0 = std::chrono::steady_clock::now();
        kick(p, 0.5 * opt.dt, 0, n);
        drift(p, opt.dt, 0, n);
        integrate_s += seconds_since(t0);
        evaluate();
        t0 = std::chrono::steady_clock::now();
        kick(p, 0.5 * opt.dt, 0, n);
        integrate_s += seconds_since(t0);
    };

    evaluate();
    const double energy_start = opt.energy ? total_energy(p) : 0.0;
    for (int s = 0; s < opt.warmup; ++s) step();
    const double setup_s = seconds_since(t_setup);
    integrate_s = build_s = 0.0;
    t = Timers{};

    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < opt.steps; ++s) step();
    const double wall = seconds_since(t0);

    char energy[160] = "";
    if (opt.energy) {
        const double energy_end = total_energy(p);
        const double drift_rel = std::abs(energy_start) > 0.0
                                     ? std::abs((energy_end - energy_start) / energy_start)
                                     : 0.0;
        std::snprintf(energy, sizeof energy,
                      ",\"energy_start\":%.17g,\"energy_end\":%.17g,\"energy_rel_drift\":%.6e",
                      energy_start, energy_end, drift_rel);
    }
    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    const double rss = peak_rss_mib();
    // traverse_s is the kernel alone. Flattening, node mapping and both transfers are
    // separate fields because a GPU run pays them on every step. Both backends flatten;
    // the tree build and flattening use one host thread, so threads speed up only the
    // cpu kernel. wall_s excludes setup_s and the warmup steps.
    omp_sched_t sched_kind;
    int sched_chunk = 0;
    omp_get_schedule(&sched_kind, &sched_chunk);
    std::printf("result {\"program\":\"bh_gpu\",\"backend\":\"%s\",\"kernel_on_device\":%s,"
                "\"separate_memory\":%s,\"num_devices\":%d,\"n\":%zu,\"theta\":%.17g,\"dt\":%.17g,"
                "\"steps\":%d,\"warmup\":%d,\"leaf_size\":%d,\"seed\":%llu,\"softening\":%.17g,"
                "\"ranks\":1,\"threads\":%d,\"threads_effective\":%d,\"distinct_cpus\":%d,\"proc_bind\":\"%s\","
                "\"schedule_effective\":\"%s\",\"chunk_effective\":%d,\"hosts\":[%s],\"nodes\":%zu,"
                "\"setup_s\":%.6f,\"wall_s\":%.6f,\"integrate_s\":%.6f,\"exchange_s\":0,\"build_s\":%.6f,"
                "\"traverse_s\":%.6f,\"flatten_s\":%.6f,\"map_nodes_s\":%.6f,\"to_device_s\":%.6f,\"from_device_s\":%.6f,"
                "\"peak_rss_mib_rank_max\":%.1f,\"peak_rss_mib_rank_sum\":%.1f%s%s}\n",
                backend == Backend::device ? "device" : "cpu",
                backend == Backend::device && device.on_device ? "true" : "false",
                device.separate_memory ? "true" : "false", device.num_devices, opt.n, opt.theta, opt.dt,
                opt.steps, opt.warmup, opt.leaf_size, static_cast<unsigned long long>(opt.seed),
                kSoftening, omp_get_max_threads(), team.team, team.distinct_cpus, proc_bind_name(),
                schedule_name(sched_kind), sched_chunk, json_string(host).c_str(),
                tree.node_count(), setup_s, wall, integrate_s, build_s, t.kernel, t.flatten, t.map_nodes, t.to_device,
                t.from_device, rss, rss, energy, build_json().c_str());
    if (!opt.dump.empty()) write_dump(opt.dump, p);
    return 0;
}
