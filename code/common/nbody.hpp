// Shared n-body core: particle state, initial conditions, direct summation,
// the Barnes-Hut octree and the leapfrog integrator.
//
// The serial reference and the MPI+OpenMP solver both include this header, so
// they build the same tree and apply the same acceptance test. Hybrid-vs-serial
// differences are then down to decomposition and summation order, not to two
// diverging copies of the tree code.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace nbody {

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
};

inline Vec3 operator+(const Vec3 &a, const Vec3 &b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline Vec3 operator-(const Vec3 &a, const Vec3 &b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline Vec3 operator*(const Vec3 &a, double s) { return {a.x * s, a.y * s, a.z * s}; }
inline double norm2(const Vec3 &a) { return a.x * a.x + a.y * a.y + a.z * a.z; }

struct Particles {
    std::vector<Vec3> pos, vel, acc;
    std::vector<double> mass;
    std::size_t size() const { return mass.size(); }
    void resize(std::size_t n) {
        pos.resize(n);
        vel.resize(n);
        acc.resize(n);
        mass.resize(n);
    }
};

// Gravitational constant is folded into the mass scale; G = 1 in model units.
constexpr double kSoftening = 0.025;   // Plummer softening, in units of the cloud radius
constexpr double kSofteningSq = kSoftening * kSoftening;

// Softened pairwise acceleration contribution of mass m at offset d.
inline Vec3 pair_accel(const Vec3 &d, double m) {
    const double r2 = norm2(d) + kSofteningSq;
    return d * (m / (r2 * std::sqrt(r2)));
}

// ---------------------------------------------------------------------------
// Initial conditions
// ---------------------------------------------------------------------------

// Plummer sphere: a standard clustered test distribution. Clustering matters --
// a uniform cube would hide the load imbalance the parallel version has to deal
// with, since tree depth varies with local density.
inline Particles make_plummer(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);

    Particles p;
    p.resize(n);
    std::fill(p.mass.begin(), p.mass.end(), n > 0 ? 1.0 / static_cast<double>(n) : 0.0);

    for (std::size_t i = 0; i < n; ++i) {
        // Inverse-transform sampling of the Plummer radial profile.
        double m = uniform(rng);
        while (m <= 0.0 || m >= 1.0) m = uniform(rng);
        const double radius = 1.0 / std::sqrt(std::pow(m, -2.0 / 3.0) - 1.0);

        const double cos_theta = 2.0 * uniform(rng) - 1.0;
        const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
        const double phi = 2.0 * M_PI * uniform(rng);

        p.pos[i] = {radius * sin_theta * std::cos(phi), radius * sin_theta * std::sin(phi),
                    radius * cos_theta};

        // Cold start: small random velocities. The dynamics do not have to be
        // physically faithful for a scaling study, but they must be identical
        // across implementations, hence the fixed seed.
        const double speed = 0.1 * uniform(rng);
        const double v_cos = 2.0 * uniform(rng) - 1.0;
        const double v_sin = std::sqrt(std::max(0.0, 1.0 - v_cos * v_cos));
        const double v_phi = 2.0 * M_PI * uniform(rng);
        p.vel[i] = {speed * v_sin * std::cos(v_phi), speed * v_sin * std::sin(v_phi),
                    speed * v_cos};
    }
    return p;
}

// ---------------------------------------------------------------------------
// Direct summation
// ---------------------------------------------------------------------------

inline Vec3 direct_accel(std::size_t i, const Particles &p) {
    Vec3 a{};
    for (std::size_t j = 0; j < p.size(); ++j) {
        if (j != i) a = a + pair_accel(p.pos[j] - p.pos[i], p.mass[j]);
    }
    return a;
}

inline void forces_direct(Particles &p) {
    for (std::size_t i = 0; i < p.size(); ++i) p.acc[i] = direct_accel(i, p);
}

// ---------------------------------------------------------------------------
// Octree
// ---------------------------------------------------------------------------

// Nodes live in one flat vector. Children are indices, not pointers: this keeps
// the tree contiguous, which matters later because the MPI version has to
// serialise subtrees to build locally essential trees on each rank.
//
// Leaves are buckets: a contiguous range [first, first + count) of order().
// Every particle lands in exactly one leaf, including coincident particles that
// reach kMaxDepth, so no particle is ever dropped from the tree.
struct Node {
    Vec3 centre;                 // geometric centre of the cell
    double half = 0.0;           // half the cell edge length
    Vec3 com;                    // centre of mass
    double mass = 0.0;
    std::int32_t child[8];       // -1 when absent
    std::int32_t first = 0;      // particle range in order()
    std::int32_t count = 0;
    bool leaf = true;

    Node() { std::fill(std::begin(child), std::end(child), -1); }
};

class Octree {
  public:
    // Coincident (or near-coincident) particles would subdivide forever; past
    // this depth they share a leaf and the softening handles them.
    static constexpr int kMaxDepth = 64;

    void build(const Particles &p, int leaf_size) {
        nodes_.clear();
        const std::size_t n = p.size();
        if (n > static_cast<std::size_t>(INT32_MAX))
            throw std::length_error("particle count exceeds int32 index range");
        order_.resize(n);
        std::iota(order_.begin(), order_.end(), 0);
        scratch_.resize(n);
        if (n == 0) return;

        // Root cell: a cube enclosing every particle, grown slightly so that
        // boundary particles are strictly inside.
        Vec3 lo = p.pos[0], hi = p.pos[0];
        for (std::size_t i = 1; i < n; ++i) {
            lo.x = std::min(lo.x, p.pos[i].x);
            lo.y = std::min(lo.y, p.pos[i].y);
            lo.z = std::min(lo.z, p.pos[i].z);
            hi.x = std::max(hi.x, p.pos[i].x);
            hi.y = std::max(hi.y, p.pos[i].y);
            hi.z = std::max(hi.z, p.pos[i].z);
        }
        double half = std::max({0.5 * (hi.x - lo.x), 0.5 * (hi.y - lo.y), 0.5 * (hi.z - lo.z)});
        half *= 1.0 + 1e-9;
        if (half <= 0.0) half = 1.0;  // degenerate case: all particles coincide

        nodes_.reserve(2 * n / static_cast<std::size_t>(std::max(1, leaf_size)) + 64);
        nodes_.emplace_back();
        nodes_[0].centre = {0.5 * (lo.x + hi.x), 0.5 * (lo.y + hi.y), 0.5 * (lo.z + hi.z)};
        nodes_[0].half = half;
        build_node(0, 0, static_cast<std::int32_t>(n), 0, p, std::max(1, leaf_size));
        slot_.resize(n);
        for (std::size_t k = 0; k < n; ++k) slot_[order_[k]] = static_cast<std::int32_t>(k);
    }

    // Accumulate the acceleration on particle i by walking the tree, opening a
    // cell only when it fails the multipole acceptance criterion s/d < theta.
    // A cell containing i is always opened, never used as a monopole: for
    // theta above about 1/sqrt(3) such a cell can pass the test, and its
    // monopole would then include i's own mass. Leaves that are opened are
    // summed particle by particle, skipping i itself.
    //
    // Thread safety: reads only; the traversal stack is a local array, so
    // concurrent calls for different i need no synchronization.
    Vec3 acceleration(std::int32_t i, const Particles &p, double theta) const {
        if (nodes_.empty()) return {};
        const double theta_sq = theta * theta;
        const Vec3 xi = p.pos[i];
        Vec3 a{};

        // Each pop pushes at most 8 children and depth is capped at kMaxDepth,
        // so at most 7 * kMaxDepth + 8 entries are ever pending.
        std::int32_t stack[8 * (kMaxDepth + 1)];
        int top = 0;
        stack[top++] = 0;

        while (top > 0) {
            const Node &node = nodes_[stack[--top]];
            const Vec3 d = node.com - xi;
            const double s = 2.0 * node.half;
            // s/d < theta, squared to avoid the square root on the test itself.
            // Every node covers the contiguous range [first, first + count) of
            // order(), so containment is a range test on i's slot.
            const bool holds_i = slot_[i] >= node.first && slot_[i] < node.first + node.count;
            if (!holds_i && s * s < theta_sq * norm2(d)) {
                a = a + pair_accel(d, node.mass);
            } else if (node.leaf) {
                for (std::int32_t k = node.first; k < node.first + node.count; ++k) {
                    const std::int32_t j = order_[k];
                    if (j != i) a = a + pair_accel(p.pos[j] - xi, p.mass[j]);
                }
            } else {
                for (int c = 0; c < 8; ++c) {
                    if (node.child[c] >= 0) stack[top++] = node.child[c];
                }
            }
        }
        return a;
    }

    std::size_t node_count() const { return nodes_.size(); }
    const std::vector<Node> &nodes() const { return nodes_; }
    const std::vector<std::int32_t> &order() const { return order_; }

  private:
    static int octant_of(const Vec3 &centre, const Vec3 &pos) {
        return (pos.x > centre.x ? 1 : 0) | (pos.y > centre.y ? 2 : 0) | (pos.z > centre.z ? 4 : 0);
    }

    std::int32_t make_child(std::int32_t parent, int octant) {
        const double quarter = 0.5 * nodes_[parent].half;
        Vec3 c = nodes_[parent].centre;
        c.x += (octant & 1) ? quarter : -quarter;
        c.y += (octant & 2) ? quarter : -quarter;
        c.z += (octant & 4) ? quarter : -quarter;

        if (nodes_.size() >= static_cast<std::size_t>(INT32_MAX))
            throw std::length_error("octree node count exceeds int32 index range");
        nodes_.emplace_back();
        const auto idx = static_cast<std::int32_t>(nodes_.size() - 1);
        nodes_[idx].centre = c;
        nodes_[idx].half = quarter;
        nodes_[parent].child[octant] = idx;
        return idx;
    }

    // Top-down build over order_[begin, end). Indices, not references, are held
    // across make_child because emplace_back may reallocate nodes_.
    void build_node(std::int32_t idx, std::int32_t begin, std::int32_t end, int depth,
                    const Particles &p, int leaf_size) {
        nodes_[idx].first = begin;
        nodes_[idx].count = end - begin;

        if (end - begin <= leaf_size || depth >= kMaxDepth) {
            double mass = 0.0;
            Vec3 weighted{};
            for (std::int32_t k = begin; k < end; ++k) {
                mass += p.mass[order_[k]];
                weighted = weighted + p.pos[order_[k]] * p.mass[order_[k]];
            }
            nodes_[idx].mass = mass;
            nodes_[idx].com = mass > 0.0 ? weighted * (1.0 / mass) : nodes_[idx].centre;
            return;
        }

        // Counting sort of the range by octant, via scratch_.
        const Vec3 centre = nodes_[idx].centre;
        std::int32_t offset[9] = {0};
        for (std::int32_t k = begin; k < end; ++k) ++offset[octant_of(centre, p.pos[order_[k]]) + 1];
        for (int c = 0; c < 8; ++c) offset[c + 1] += offset[c];
        std::int32_t cursor[8];
        for (int c = 0; c < 8; ++c) cursor[c] = begin + offset[c];
        for (std::int32_t k = begin; k < end; ++k) {
            const std::int32_t j = order_[k];
            scratch_[cursor[octant_of(centre, p.pos[j])]++] = j;
        }
        std::copy(scratch_.begin() + begin, scratch_.begin() + end, order_.begin() + begin);

        nodes_[idx].leaf = false;
        double mass = 0.0;
        Vec3 weighted{};
        for (int c = 0; c < 8; ++c) {
            const std::int32_t lo = begin + offset[c], hi = begin + offset[c + 1];
            if (lo == hi) continue;
            const std::int32_t ci = make_child(idx, c);
            build_node(ci, lo, hi, depth + 1, p, leaf_size);
            mass += nodes_[ci].mass;
            weighted = weighted + nodes_[ci].com * nodes_[ci].mass;
        }
        nodes_[idx].mass = mass;
        nodes_[idx].com = mass > 0.0 ? weighted * (1.0 / mass) : centre;
    }

    std::vector<Node> nodes_;
    std::vector<std::int32_t> order_, scratch_, slot_;  // slot_ inverts order_
};

// ---------------------------------------------------------------------------
// Integration and diagnostics
// ---------------------------------------------------------------------------

// Kick-drift-kick leapfrog. Symplectic for an exact force; with an approximate,
// changing tree force the energy drift is a diagnostic to measure, not a bound.
// Both act on particles [begin, end); the hybrid solver passes its owned range.
inline void kick(Particles &p, double half_dt, std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) p.vel[i] = p.vel[i] + p.acc[i] * half_dt;
}

inline void drift(Particles &p, double dt, std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) p.pos[i] = p.pos[i] + p.vel[i] * dt;
}

inline double total_energy(const Particles &p) {
    double kinetic = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) kinetic += 0.5 * p.mass[i] * norm2(p.vel[i]);
    double potential = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        for (std::size_t j = i + 1; j < p.size(); ++j) {
            const double r = std::sqrt(norm2(p.pos[j] - p.pos[i]) + kSofteningSq);
            potential -= p.mass[i] * p.mass[j] / r;
        }
    }
    return kinetic + potential;
}

struct ErrorStats {
    double max_rel = 0.0;
    double mean_rel = 0.0;
    double max_abs = 0.0;
};

// Relative error per particle; where the exact force is zero, the absolute
// error is used instead and also reported separately as max_abs.
inline ErrorStats compare(const std::vector<Vec3> &approx, const std::vector<Vec3> &exact) {
    ErrorStats stats;
    double sum = 0.0;
    for (std::size_t i = 0; i < exact.size(); ++i) {
        const double num = std::sqrt(norm2(approx[i] - exact[i]));
        const double den = std::sqrt(norm2(exact[i]));
        const double rel = den > 0.0 ? num / den : num;
        stats.max_rel = std::max(stats.max_rel, rel);
        stats.max_abs = std::max(stats.max_abs, num);
        sum += rel;
    }
    stats.mean_rel = exact.empty() ? 0.0 : sum / static_cast<double>(exact.size());
    return stats;
}

}  // namespace nbody
