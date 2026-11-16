// Command-line options, state dumps and result records shared by the serial
// and hybrid drivers, so both accept the same physics parameters with the same
// validation and write comparable output.

#pragma once

#include "nbody.hpp"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <sys/resource.h>

// Compiler flags, passed in by the Makefile so every result records how its
// binary was built.
#ifndef NBODY_CXXFLAGS
#define NBODY_CXXFLAGS "unknown"
#endif

namespace nbody {

// Build provenance fields for the result JSON (leading comma included).
inline std::string build_json() {
    return ",\"compiler\":" + std::string("\"") + __VERSION__ + "\",\"cxxflags\":\"" + NBODY_CXXFLAGS + "\"";
}

// Peak resident set size of this process so far, in MiB. ru_maxrss is in KiB
// on Linux and in bytes on macOS.
inline double peak_rss_mib() {
    struct rusage u {};
    getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
    return static_cast<double>(u.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(u.ru_maxrss) / 1024.0;
#endif
}

// Parameters that define the simulated problem and the timed region. Anything
// that changes the mathematics lives here so both drivers agree on it.
struct RunOptions {
    std::size_t n = 10000;
    double theta = 0.5;
    double dt = 1e-3;
    int steps = 10;
    int warmup = 0;          // untimed steps run before the timed region
    int leaf_size = 8;
    std::uint64_t seed = 42;
    bool energy = false;     // O(N^2) energy before/after; small inputs only
    std::string dump;        // final state file, empty for none
};

// Whole-string numeric parsing: "12abc", "", "-5" and out-of-range values throw.
inline unsigned long long parse_u64(const std::string &name, const char *text) {
    char *end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE || text[0] == '-')
        throw std::invalid_argument("invalid value for " + name + ": '" + text + "'");
    return v;
}

inline double parse_double(const std::string &name, const char *text) {
    char *end = nullptr;
    errno = 0;
    const double v = std::strtod(text, &end);
    if (end == text || *end != '\0' || errno == ERANGE || !std::isfinite(v))
        throw std::invalid_argument("invalid value for " + name + ": '" + text + "'");
    return v;
}

inline int parse_int_in(const std::string &name, const char *text, long long lo, long long hi) {
    const unsigned long long v = parse_u64(name, text);
    if (static_cast<long long>(v) < lo || v > static_cast<unsigned long long>(hi))
        throw std::invalid_argument(name + " must be in [" + std::to_string(lo) + ", " +
                                    std::to_string(hi) + "]");
    return static_cast<int>(v);
}

// Consumes argv[i] (and its value) if it is a common option. `next` returns
// the following argument or throws when it is missing.
template <class Next>
bool parse_common_option(const std::string &arg, Next next, RunOptions &opt) {
    if (arg == "--n") opt.n = parse_u64(arg, next());
    else if (arg == "--theta") opt.theta = parse_double(arg, next());
    else if (arg == "--dt") opt.dt = parse_double(arg, next());
    else if (arg == "--steps") opt.steps = parse_int_in(arg, next(), 1, 1000000000);
    else if (arg == "--warmup") opt.warmup = parse_int_in(arg, next(), 0, 1000000000);
    else if (arg == "--leaf-size") opt.leaf_size = parse_int_in(arg, next(), 1, 4096);
    else if (arg == "--seed") opt.seed = parse_u64(arg, next());
    else if (arg == "--energy") opt.energy = true;
    else if (arg == "--dump") opt.dump = next();
    else return false;
    return true;
}

inline void check_common_options(const RunOptions &opt) {
    // Particle indices are int32 in the tree.
    if (opt.n < 1 || opt.n > 2000000000ULL) throw std::invalid_argument("--n must be in [1, 2e9]");
    if (opt.theta < 0.0) throw std::invalid_argument("--theta must be >= 0");
    if (opt.dt <= 0.0) throw std::invalid_argument("--dt must be > 0");
}

constexpr const char *kCommonUsage =
    "[--n N] [--theta T] [--dt DT] [--steps S] [--warmup W] [--seed K]\n"
    "          [--leaf-size L] [--energy] [--dump FILE]";

// Full-precision text dump in particle-ID order, for serial/hybrid comparison.
inline void write_dump(const std::string &path, const Particles &p) {
    std::FILE *f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot open dump file " + path);
    std::fprintf(f, "# id x y z vx vy vz ax ay az mass\n");
    for (std::size_t i = 0; i < p.size(); ++i) {
        const Vec3 &x = p.pos[i], &v = p.vel[i], &a = p.acc[i];
        std::fprintf(f, "%zu %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", i, x.x,
                     x.y, x.z, v.x, v.y, v.z, a.x, a.y, a.z, p.mass[i]);
    }
    if (std::fclose(f) != 0) throw std::runtime_error("error writing dump file " + path);
}

// Minimal JSON string escaping for paths and names in result records.
inline std::string json_string(const std::string &s) {
    std::string out = "\"";
    for (const char c : s) {
        if (c == '"' || c == '\\') out += '\\';
        if (static_cast<unsigned char>(c) < 0x20) continue;
        out += c;
    }
    return out + "\"";
}

}  // namespace nbody
