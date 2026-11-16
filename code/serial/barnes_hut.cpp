// Serial Barnes-Hut n-body reference implementation.
//
// Two force evaluators over the same particle set:
//   direct  O(N^2) pairwise summation, the correctness reference
//   bh      O(N log N) octree traversal with the s/d < theta acceptance test
//
// The parallel versions are validated against the serial ones, and the serial
// Barnes-Hut timing is the base of every speedup number in the report.

#include "../common/nbody.hpp"
#include "../common/run_support.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

using namespace nbody;

void forces_barnes_hut(Particles &p, double theta, int leaf_size, Octree &tree) {
    tree.build(p, leaf_size);
    const auto n = static_cast<std::int32_t>(p.size());
    for (std::int32_t i = 0; i < n; ++i) p.acc[i] = tree.acceleration(i, p, theta);
}

double seconds_since(const std::chrono::steady_clock::time_point &t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------------------
// Edge-case self-test
// ---------------------------------------------------------------------------

// Structural invariants: order() is a permutation, every particle sits in
// exactly one leaf, and the root carries the total mass.
bool tree_invariants_hold(const Octree &tree, const Particles &p, std::string &why) {
    const std::size_t n = p.size();
    if (n == 0) {
        if (tree.node_count() != 0) { why = "empty input built nodes"; return false; }
        return true;
    }
    std::vector<int> seen(n, 0);
    for (const std::int32_t j : tree.order()) {
        if (j < 0 || static_cast<std::size_t>(j) >= n || seen[j]++) { why = "order is not a permutation"; return false; }
    }
    std::size_t in_leaves = 0;
    for (const Node &node : tree.nodes()) {
        if (node.leaf) in_leaves += static_cast<std::size_t>(node.count);
    }
    if (in_leaves != n) { why = "leaves hold " + std::to_string(in_leaves) + " of " + std::to_string(n); return false; }
    double total = 0.0;
    for (const double m : p.mass) total += m;
    if (std::abs(tree.nodes()[0].mass - total) > 1e-12 * total) { why = "root mass differs from total"; return false; }
    return true;
}

// theta = 0 never accepts a cell, so the tree must reproduce direct summation
// up to summation order. Tolerance is relative, with an absolute floor scaled
// by the largest force so that zero-force particles do not divide by zero.
bool matches_direct(const Particles &p, const std::vector<Vec3> &tree_acc, std::string &why) {
    double scale = 0.0;
    std::vector<Vec3> exact(p.size());
    for (std::size_t i = 0; i < p.size(); ++i) {
        exact[i] = direct_accel(i, p);
        scale = std::max(scale, std::sqrt(norm2(exact[i])));
    }
    for (std::size_t i = 0; i < p.size(); ++i) {
        const double err = std::sqrt(norm2(tree_acc[i] - exact[i]));
        if (!(err <= 1e-10 * std::sqrt(norm2(exact[i])) + 1e-13 * scale)) {
            char buf[96];
            std::snprintf(buf, sizeof buf, "particle %zu error %.3e", i, err);
            why = buf;
            return false;
        }
    }
    return true;
}

Particles plummer_with(std::size_t n, std::uint64_t seed, std::size_t extra, const Vec3 &where,
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

int run_self_test() {
    struct Case {
        const char *name;
        Particles p;
    };
    std::vector<Case> cases;
    cases.push_back({"empty", Particles{}});
    cases.push_back({"singleton", plummer_with(0, 1, 1, {0.3, -0.2, 0.1}, 0.0)});
    cases.push_back({"all_coincident", plummer_with(0, 1, 100, {1.0, 1.0, 1.0}, 0.0)});
    cases.push_back({"coincident_cluster", plummer_with(500, 7, 50, {0.1, 0.2, 0.3}, 0.0)});
    // Distinct points 1e-25 apart near the origin (representable there, unlike
    // near 0.05) sit far below root_half / 2^kMaxDepth: forces the max-depth leaf.
    cases.push_back({"below_max_depth", plummer_with(200, 9, 5, {0.0, 0.0, 0.0}, 1e-25)});
    {
        Particles p = make_plummer(1000, 11);
        std::mt19937_64 rng(11);
        std::uniform_real_distribution<double> u(0.1, 10.0);
        for (double &m : p.mass) m = u(rng) / 1000.0;
        cases.push_back({"unequal_masses", std::move(p)});
    }
    {
        // Integer lattice: many particles lie exactly on cell dividing planes.
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
    for (const Case &c : cases) {
        for (const int leaf_size : {1, 8}) {
            Octree tree;
            tree.build(c.p, leaf_size);
            const auto n = static_cast<std::int32_t>(c.p.size());
            std::vector<Vec3> exact_tree(c.p.size()), approx(c.p.size());
            for (std::int32_t i = 0; i < n; ++i) {
                exact_tree[i] = tree.acceleration(i, c.p, 0.0);
                approx[i] = tree.acceleration(i, c.p, 0.5);
            }
            std::string why;
            bool ok = tree_invariants_hold(tree, c.p, why) && matches_direct(c.p, exact_tree, why);
            for (const Vec3 &a : approx) {
                if (ok && !std::isfinite(norm2(a))) { ok = false; why = "nonfinite theta=0.5 force"; }
            }
            std::printf("self_test case=%s n=%zu leaf_size=%d nodes=%zu %s%s%s\n", c.name,
                        c.p.size(), leaf_size, tree.node_count(), ok ? "PASS" : "FAIL",
                        ok ? "" : " reason=", ok ? "" : why.c_str());
            failures += ok ? 0 : 1;
        }
    }
    auto report = [&](const char *name, bool ok, const std::string &why) {
        std::printf("self_test case=%s %s%s%s\n", name, ok ? "PASS" : "FAIL", ok ? "" : " reason=",
                    ok ? "" : why.c_str());
        failures += ok ? 0 : 1;
    };

    // Two particles, closed-form softened force. theta = 10 makes the root cell
    // pass the acceptance test from either particle; it must still be opened
    // because it contains the target.
    {
        Particles p;
        p.pos = {{0.1, 0.2, 0.3}, {0.6, 0.2, 0.3}};
        p.vel.assign(2, Vec3{});
        p.acc.assign(2, Vec3{});
        p.mass = {0.3, 0.7};
        const double r = 0.5, denom = std::pow(r * r + kSofteningSq, 1.5);
        const Vec3 expect[2] = {{p.mass[1] * r / denom, 0.0, 0.0}, {-p.mass[0] * r / denom, 0.0, 0.0}};
        bool ok = true;
        std::string why;
        for (const int leaf_size : {1, 8}) {
            for (const double theta : {0.0, 0.5, 10.0}) {
                Octree tree;
                tree.build(p, leaf_size);
                for (std::int32_t i = 0; i < 2; ++i) {
                    const Vec3 a = tree.acceleration(i, p, theta);
                    const double err = std::sqrt(norm2(a - expect[i]));
                    if (ok && !(err <= 1e-14 * std::sqrt(norm2(expect[i])))) {
                        ok = false;
                        char buf[96];
                        std::snprintf(buf, sizeof buf, "leaf_size=%d theta=%g i=%d err=%.3e",
                                      leaf_size, theta, i, err);
                        why = buf;
                    }
                }
            }
        }
        report("two_particle_analytic", ok, why);
    }

    // Cube corners plus centre: the centre force cancels and every corner feels
    // the same magnitude.
    {
        Particles p;
        for (int c = 0; c < 8; ++c)
            p.pos.push_back({(c & 1) ? 1.0 : -1.0, (c & 2) ? 1.0 : -1.0, (c & 4) ? 1.0 : -1.0});
        p.pos.push_back({0.0, 0.0, 0.0});
        p.vel.assign(9, Vec3{});
        p.acc.assign(9, Vec3{});
        p.mass.assign(9, 1.0 / 9.0);
        Octree tree;
        tree.build(p, 1);
        std::vector<double> mag(9);
        for (std::int32_t i = 0; i < 9; ++i) mag[i] = std::sqrt(norm2(tree.acceleration(i, p, 0.0)));
        bool ok = mag[8] <= 1e-15 * mag[0];
        for (int c = 1; c < 8 && ok; ++c) ok = std::abs(mag[c] - mag[0]) <= 1e-13 * mag[0];
        report("symmetric_cube", ok, "centre force or corner magnitudes differ");
    }

    // Negative controls: the checks above must be able to fail.
    {
        Particles p = make_plummer(300, 3);
        Octree tree;
        tree.build(p, 8);
        std::vector<Vec3> acc(p.size());
        for (std::int32_t i = 0; i < 300; ++i) acc[i] = tree.acceleration(i, p, 0.0);
        std::string why;
        acc[17].x *= 1.0 + 1e-8;
        const bool force_caught = !matches_direct(p, acc, why);
        Particles heavier = p;
        heavier.mass[5] *= 2.0;
        const bool mass_caught = !tree_invariants_hold(tree, heavier, why);
        report("negative_control_force_perturbation", force_caught, "1e-8 perturbation not detected");
        report("negative_control_mass_mismatch", mass_caught, "mass mismatch not detected");
    }

    std::printf("self_test failures=%d\n", failures);
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

void usage(const char *prog) {
    std::fprintf(stderr, "usage: %s %s\n          [--method bh|direct] [--validate] [--self-test]\n",
                 prog, kCommonUsage);
}

}  // namespace

int main(int argc, char **argv) {
    RunOptions opt;
    std::string method = "bh";  // bh | direct
    bool validate = false, self_test = false;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto next = [&]() -> const char * {
                if (i + 1 >= argc) throw std::invalid_argument(arg + " needs a value");
                return argv[++i];
            };
            if (parse_common_option(arg, next, opt)) continue;
            if (arg == "--method") method = next();
            else if (arg == "--validate") validate = true;
            else if (arg == "--self-test") self_test = true;
            else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
            else throw std::invalid_argument("unknown option: " + arg);
        }
        if (self_test) return run_self_test();
        check_common_options(opt);
        if (method != "bh" && method != "direct") throw std::invalid_argument("--method must be bh or direct");
    } catch (const std::invalid_argument &e) {
        std::fprintf(stderr, "%s\n", e.what());
        usage(argv[0]);
        return 2;
    }

    // Setup is timed from initial-condition generation, as in the hybrid
    // driver, where it also covers the broadcast.
    const auto t_setup = std::chrono::steady_clock::now();
    Particles p = make_plummer(opt.n, opt.seed);
    Octree tree;

    // Validation mode: one force evaluation each way, on identical state.
    if (validate) {
        forces_direct(p);
        const std::vector<Vec3> exact = p.acc;
        forces_barnes_hut(p, opt.theta, opt.leaf_size, tree);
        const ErrorStats stats = compare(p.acc, exact);
        std::printf("validate n=%zu theta=%.3f leaf_size=%d nodes=%zu max_rel_err=%.3e "
                    "mean_rel_err=%.3e max_abs_err=%.3e\n",
                    opt.n, opt.theta, opt.leaf_size, tree.node_count(), stats.max_rel,
                    stats.mean_rel, stats.max_abs);
        return 0;
    }

    // Same phase split as the hybrid driver: integrate (kicks + drift), tree
    // build, force traversal. There is no exchange phase in serial.
    const bool use_bh = (method == "bh");
    const std::size_t n = p.size();
    double integrate_s = 0.0, build_s = 0.0, traverse_s = 0.0;
    auto evaluate = [&]() {
        auto t = std::chrono::steady_clock::now();
        if (use_bh) {
            tree.build(p, opt.leaf_size);
            build_s += seconds_since(t);
            t = std::chrono::steady_clock::now();
            for (std::int32_t i = 0; i < static_cast<std::int32_t>(n); ++i)
                p.acc[i] = tree.acceleration(i, p, opt.theta);
        } else {
            forces_direct(p);
        }
        traverse_s += seconds_since(t);
    };
    auto step = [&]() {
        auto t = std::chrono::steady_clock::now();
        kick(p, 0.5 * opt.dt, 0, n);
        drift(p, opt.dt, 0, n);
        integrate_s += seconds_since(t);
        evaluate();
        t = std::chrono::steady_clock::now();
        kick(p, 0.5 * opt.dt, 0, n);
        integrate_s += seconds_since(t);
    };

    // Rest of setup: initial forces, optional energy and warm-up steps.
    evaluate();
    const double energy_start = opt.energy ? total_energy(p) : 0.0;
    for (int s = 0; s < opt.warmup; ++s) step();
    const double setup_s = seconds_since(t_setup);
    integrate_s = build_s = traverse_s = 0.0;

    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < opt.steps; ++s) step();
    const double wall = seconds_since(t0);

    std::printf("method=%s n=%zu theta=%.3f leaf_size=%d steps=%d wall_s=%.4f build_s=%.4f "
                "traverse_s=%.4f integrate_s=%.4f s_per_step=%.4f\n",
                method.c_str(), opt.n, opt.theta, opt.leaf_size, opt.steps, wall, build_s,
                traverse_s, integrate_s, wall / opt.steps);
    char energy[160] = "";
    if (opt.energy) {
        const double energy_end = total_energy(p);
        const double drift_rel = std::abs(energy_start) > 0.0
                                     ? std::abs((energy_end - energy_start) / energy_start)
                                     : 0.0;
        std::printf("energy_start=%.9e energy_end=%.9e rel_drift=%.3e\n", energy_start, energy_end,
                    drift_rel);
        std::snprintf(energy, sizeof energy,
                      ",\"energy_start\":%.17g,\"energy_end\":%.17g,\"energy_rel_drift\":%.6e",
                      energy_start, energy_end, drift_rel);
    }
    char host[256] = "unknown";
    gethostname(host, sizeof host - 1);
    // Measured after the timed loop and before any dump is written.
    const double rss = peak_rss_mib();
    std::printf("result {\"program\":\"bh_serial\",\"method\":\"%s\",\"n\":%zu,\"theta\":%.17g,"
                "\"dt\":%.17g,\"steps\":%d,\"warmup\":%d,\"leaf_size\":%d,\"seed\":%llu,"
                "\"softening\":%.17g,\"ranks\":1,\"threads\":1,\"threads_effective\":1,"
                "\"hosts\":[%s],\"nodes\":%zu,\"setup_s\":%.6f,"
                "\"wall_s\":%.6f,\"integrate_s\":%.6f,\"exchange_s\":0,\"build_s\":%.6f,"
                "\"traverse_s\":%.6f,\"peak_rss_mib_rank_max\":%.1f,\"peak_rss_mib_rank_sum\":%.1f%s%s}\n",
                method.c_str(), opt.n, opt.theta, opt.dt, opt.steps, opt.warmup, opt.leaf_size,
                static_cast<unsigned long long>(opt.seed), kSoftening, json_string(host).c_str(),
                tree.node_count(), setup_s, wall, integrate_s, build_s, traverse_s, rss, rss,
                energy, build_json().c_str());
    if (!opt.dump.empty()) write_dump(opt.dump, p);
    return 0;
}
