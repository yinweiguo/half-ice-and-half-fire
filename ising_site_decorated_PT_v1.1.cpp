// Monte Carlo simulation of a site-decorated Ising square lattice
//      Input file: para.in
//      Output file: M_surface.csv
// v1.0 2026-02-04
// v1.1 2026-02-05 Added mabs_mean, mabs_err, useful for very tiny heff

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

// -------------------------
// RNG: xorshift64*
// -------------------------
struct XorShift64Star {
    uint64_t x;
    explicit XorShift64Star(uint64_t seed = 88172645463325252ull) : x(seed ? seed : 1ull) {}
    inline uint64_t next_u64() {
        uint64_t z = x;
        z ^= z >> 12;
        z ^= z << 25;
        z ^= z >> 27;
        x = z;
        return z * 2685821657736338717ull;
    }
    inline double next_u01() {
        return (next_u64() >> 11) * (1.0 / 9007199254740992.0);
    }
};

static inline std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// -------------------------
// Geometry
// -------------------------
struct LatticeGeom {
    int L = 0, N = 0;
    std::vector<int> up, dn, lt, rt;
    std::vector<int> black, white;

    explicit LatticeGeom(int L_) : L(L_), N(L_*L_), up(N), dn(N), lt(N), rt(N) {
        black.reserve(N/2 + 1);
        white.reserve(N/2 + 1);

        for (int y = 0; y < L; ++y) {
            int yp = (y + 1) % L;
            int ym = (y - 1 + L) % L;
            for (int x = 0; x < L; ++x) {
                int xp = (x + 1) % L;
                int xm = (x - 1 + L) % L;
                int i = y * L + x;
                up[i] = ym * L + x;
                dn[i] = yp * L + x;
                lt[i] = y * L + xm;
                rt[i] = y * L + xp;

                if (((x + y) & 1) == 0) black.push_back(i);
                else white.push_back(i);
            }
        }
    }
};

struct Replica {
    std::vector<int8_t> s; // ±1
    int Eint = 0;          // interaction energy (integer): -sum_{<ij>} s_i s_j, counting each bond once
    int M = 0;             // magnetization sum
    XorShift64Star rng;

    explicit Replica(int N=0, uint64_t seed=1) : s(N, 1), rng(seed) {}
};

static inline int compute_Eint(const LatticeGeom& g, const std::vector<int8_t>& s) {
    int E = 0;
    for (int i = 0; i < g.N; ++i) {
        int si = (int)s[i];
        E += -si * ((int)s[g.rt[i]] + (int)s[g.dn[i]]);
    }
    return E;
}

static inline int compute_M(const std::vector<int8_t>& s) {
    int M = 0;
    for (auto v : s) M += (int)v;
    return M;
}

static inline double Etot(double J, int Eint, double Hfield, int M) {
    // Total energy for Hamiltonian:  -J sum_{<ij>} s_i s_j - Hfield * sum_i s_i
    // Here Eint already equals -sum_{<ij>} s_i s_j, so interaction piece is J*Eint.
    return J * (double)Eint - Hfield * (double)M;
}

// -------------------------
// Site-decorated mapping: Eqs (5) and (7)
// -------------------------
// +/- = 2 cosh(β h μb ± β Jab)
// h_eff = h + [1/(2β μa)] (ln(+) - ln(-))
static inline double heff(double h, double T, double Jab, double mu_a, double mu_b) {
    const double beta = 1.0 / T;
    const double ap = beta * (h * mu_b + Jab);
    const double am = beta * (h * mu_b - Jab);

    // ln(2 cosh(x)) in a stable way:
    auto ln2cosh = [](double x) {
        double ax = std::fabs(x);
        // ln(2cosh x) = ax + ln(1+exp(-2ax))
        return ax + std::log1p(std::exp(-2.0 * ax));
    };

    double ln_plus  = ln2cosh(ap);
    double ln_minus = ln2cosh(am);

    return h + (ln_plus - ln_minus) / (2.0 * beta * mu_a);
}

// Solve T0(h) from h_eff(h,T0)=0 (Eq. 14), equivalent to Eq. (15).
static inline bool solve_T0_bisect(double h, double Jab, double mu_a, double mu_b,
                                  double& T0_out) {
    // We need a bracket [Tlo, Thi] with heff(Tlo) < 0 and heff(Thi) > 0.
    // For Jab<0 and 0<h<hc=|Jab|/mu_a, heff(T->0) ~ h + Jab/mu_a < 0, and heff(T->∞)->h>0.
    const double Tlo0 = 1e-6;
    const double Thi0 = 1e3;

    double f_lo = heff(h, Tlo0, Jab, mu_a, mu_b);
    double f_hi = heff(h, Thi0, Jab, mu_a, mu_b);

    if (!(f_lo < 0.0 && f_hi > 0.0)) {
        // No finite-T0 under the given parameters (or h too large/small).
        return false;
    }

    double lo = Tlo0, hi = Thi0;
    for (int it = 0; it < 200; ++it) {
        double mid = 0.5 * (lo + hi);
        double fmid = heff(h, mid, Jab, mu_a, mu_b);
        if (fmid > 0.0) hi = mid;
        else lo = mid;
        if (std::fabs(hi - lo) / mid < 1e-4) break;
    }
    T0_out = 0.5 * (lo + hi);
    return true;
}

// -------------------------
// Heat-bath sweep (checkerboard)
// -------------------------
static inline void heatbath_half_sweep(
    const LatticeGeom& g,
    Replica& rep,
    const std::vector<int>& sites,
    double beta,
    double J,
    double Hfield,
    const double p_up[5] // nn=-4,-2,0,2,4
) {
    for (int idx = 0; idx < (int)sites.size(); ++idx) {
        int i = sites[idx];
        int8_t sold = rep.s[i];

        int nn = (int)rep.s[g.up[i]] + (int)rep.s[g.dn[i]] + (int)rep.s[g.lt[i]] + (int)rep.s[g.rt[i]];
        int k = (nn + 4) / 2;

        double u = rep.rng.next_u01();
        int8_t snew = (u < p_up[k]) ? int8_t(+1) : int8_t(-1);

        if (snew != sold) {
            // Interaction energy increment: dEint = 2*s_old*nn
            int dEint = 2 * (int)sold * nn;
            rep.Eint += dEint;
            rep.M += (int)snew - (int)sold;
            rep.s[i] = snew;
        }
    }
}

static inline void heatbath_sweep(const LatticeGeom& g, Replica& rep, double T, double J, double Hfield) {
    const double beta = 1.0 / T;

    // Precompute P(s_i=+1 | nn) = 1/(1+exp(-2β(J*nn + Hfield)))
    double p_up[5];
    for (int kk = 0; kk < 5; ++kk) {
        int nn = -4 + 2*kk;
        double x = J * (double)nn + Hfield;
        double a = -2.0 * beta * x;
        if (a > 50) p_up[kk] = 0.0;
        else if (a < -50) p_up[kk] = 1.0;
        else p_up[kk] = 1.0 / (1.0 + std::exp(a));
    }

    heatbath_half_sweep(g, rep, g.black, beta, J, Hfield, p_up);
    heatbath_half_sweep(g, rep, g.white, beta, J, Hfield, p_up);
}

// -------------------------
// Replica exchange (swap neighboring T's)
// NOTE: Here each replica has its own T AND its own Hfield(T) = mu_a * heff(h,T).
// Detailed balance holds with the generalized swap rule.
// -------------------------
static inline void attempt_swaps(std::vector<Replica>& reps,
                                const std::vector<double>& T_list,
                                const std::vector<double>& H_list,
                                double J) {
    const int R = (int)reps.size();
    for (int r = 0; r < R - 1; ++r) {
        double b1 = 1.0 / T_list[r];
        double b2 = 1.0 / T_list[r+1];

        double E1 = Etot(J, reps[r].Eint,   H_list[r],   reps[r].M);
        double E2 = Etot(J, reps[r+1].Eint, H_list[r+1], reps[r+1].M);

        // Swap acceptance for parameter-dependent Hamiltonians:
        // A = min(1, exp( -β1 H1(s2) -β2 H2(s1) +β1 H1(s1) +β2 H2(s2) ))
        // With our Etot(J,Eint,H,M), this becomes:
        double E1_on_2 = Etot(J, reps[r+1].Eint, H_list[r],   reps[r+1].M);
        double E2_on_1 = Etot(J, reps[r].Eint,   H_list[r+1], reps[r].M);

        double d = -b1 * E1_on_2 - b2 * E2_on_1 + b1 * E1 + b2 * E2;

        double u = reps[r].rng.next_u01();
        if (d >= 0.0 || u < std::exp(d)) {
            std::swap(reps[r].s, reps[r+1].s);
            std::swap(reps[r].Eint, reps[r+1].Eint);
            std::swap(reps[r].M, reps[r+1].M);
        }
    }
}

static inline void mean_and_stderr(const std::vector<double>& x, double& mean, double& err) {
    int n = (int)x.size();
    if (n == 0) { mean = 0; err = 0; return; }
    double m = 0.0;
    for (double v : x) m += v;
    m /= (double)n;

    double var = 0.0;
    for (double v : x) {
        double d = v - m;
        var += d*d;
    }
    var /= (double)std::max(1, n - 1);
    mean = m;
    err = std::sqrt(var / (double)n);
}

// Build a T list in [Tmin,Tmax] plus dense windows around Tc and T0.
static inline std::vector<double> build_T_list(double Tmin, double Tmax, double Tc, double T0, bool hasT0) {
    std::vector<double> T;

    auto add_lin = [&](double a, double b, double step) {
        if (b <= a) return;
        int n = (int)std::floor((b - a) / step) + 1;
        for (int i = 0; i < n; ++i) {
            double v = a + step * i;
            if (v >= Tmin - 1e-15 && v <= Tmax + 1e-15) T.push_back(v);
        }
    };

    // baseline coarse grid
    add_lin(Tmin, Tmax, 0.1);

    // dense around Tc
    add_lin(Tc - 0.08, Tc + 0.08, 0.01);
    add_lin(Tc - 0.02, Tc + 0.02, 0.002);

    // dense around T0(h)
    if (hasT0) {
        add_lin(T0 - 0.08, T0 + 0.08, 0.01);
        add_lin(T0 - 0.02, T0 + 0.02, 0.002);
    }

    // clamp & unique-sort
    for (double &v : T) v = std::min(std::max(v, Tmin), Tmax);
    std::sort(T.begin(), T.end());
    T.erase(std::unique(T.begin(), T.end(), [](double a, double b){ return std::fabs(a-b) < 1e-12; }), T.end());

    return T;
}

// -------------------------
// Input parsing
// -------------------------
struct Input {
    int L = 120;
    double J = 1.0;
    double Jab = -2.0;
    double mu_a = 1.0;
    double mu_b = 4.0/3.0;

    int swap_every = 5;
    int therm_sweeps = 12000;
    int meas_sweeps  = 40000;
    int bin_size = 50;
    uint64_t seed = 1234567;

    double T_min = 1.5;
    double T_max = 2.5;

    std::vector<double> h_list;
};

static inline bool parse_para_in(const std::string& fname, Input& in) {
    std::ifstream f(fname);
    if (!f) return false;

    std::string line;
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;

        // accept either "key = value" or "h = value"
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = trim(line.substr(0, eq));
        std::string val = trim(line.substr(eq + 1));

        auto to_double = [&](const std::string& s) {
            return std::stod(s);
        };
        auto to_int = [&](const std::string& s) {
            return std::stoi(s);
        };

        if (key == "L") in.L = to_int(val);
        else if (key == "J") in.J = to_double(val);
        else if (key == "Jab") in.Jab = to_double(val);
        else if (key == "mu_a") in.mu_a = to_double(val);
        else if (key == "mu_b") in.mu_b = to_double(val);
        else if (key == "swap_every") in.swap_every = to_int(val);
        else if (key == "therm_sweeps") in.therm_sweeps = to_int(val);
        else if (key == "meas_sweeps") in.meas_sweeps = to_int(val);
        else if (key == "bin_size") in.bin_size = to_int(val);
        else if (key == "seed") in.seed = (uint64_t)std::stoull(val);
        else if (key == "T_min") in.T_min = to_double(val);
        else if (key == "T_max") in.T_max = to_double(val);
        else if (key == "h") in.h_list.push_back(to_double(val));
    }

    std::sort(in.h_list.begin(), in.h_list.end());
    in.h_list.erase(std::unique(in.h_list.begin(), in.h_list.end(), [](double a, double b){ return std::fabs(a-b) < 1e-15; }), in.h_list.end());
    return !in.h_list.empty();
}

int main() {
    Input in;
    if (!parse_para_in("para.in", in)) {
        std::cerr << "Failed to read para.in or no h values provided.\n";
        return 1;
    }

    const int L = in.L;
    const int N = L * L;

    // Square-lattice critical temperature for J=1 (Eq. 26 in the manuscript)
    const double Tc = 2.0 * in.J / std::log(1.0 + std::sqrt(2.0));

    std::cout << "L=" << L << " N=" << N
              << " J=" << in.J
              << " Jab=" << in.Jab
              << " mu_a=" << in.mu_a
              << " mu_b=" << in.mu_b
              << " Tc=" << Tc
              << " h_points=" << in.h_list.size()
              << "\n"
              << " swap_every=" << in.swap_every
              << " therm_sweeps=" << in.therm_sweeps
              << " meas_sweeps=" << in.meas_sweeps
              << " bin_size=" << in.bin_size
              << " seed=" << in.seed 
              << "\n";

    LatticeGeom geom(L);

    std::ofstream out("M_surface.csv");
    out << "h,T,T0,heff,Hfield,m_mean,m_err,mabs_mean,mabs_err,meas_bins\n";

    for (int hi = 0; hi < (int)in.h_list.size(); ++hi) {
        double h = in.h_list[hi];

        double T0 = std::numeric_limits<double>::quiet_NaN();
        bool hasT0 = solve_T0_bisect(h, in.Jab, in.mu_a, in.mu_b, T0);

        if (!hasT0) {
            std::cout << "h=" << h << " : no finite T0 found (check parameter regime).\n";
        } else {
            std::cout << "h=" << h << " : T0=" << T0 << "\n";
        }

        // Build temperature list for this h
        std::vector<double> T_list = build_T_list(in.T_min, in.T_max, Tc, T0, hasT0);
        const int R = (int)T_list.size();

        // Precompute effective fields per replica: Hfield(T)=mu_a * h_eff(h,T)
        std::vector<double> H_list(R);
        for (int r = 0; r < R; ++r) {
            double he = heff(h, T_list[r], in.Jab, in.mu_a, in.mu_b);
            H_list[r] = in.mu_a * he; // field coupling in the backbone Hamiltonian
        }

        // Init replicas
        std::vector<Replica> reps;
        reps.reserve(R);
        uint64_t base_seed = in.seed + 100000ull * (uint64_t)hi;

        for (int r = 0; r < R; ++r) {
            reps.emplace_back(N, base_seed + 7777ull * (uint64_t)r);
            for (int i = 0; i < N; ++i) {
                reps[r].s[i] = (reps[r].rng.next_u01() < 0.5) ? int8_t(+1) : int8_t(-1);
            }
            reps[r].M = compute_M(reps[r].s);
            reps[r].Eint = compute_Eint(geom, reps[r].s);
        }

        // Thermalize
        for (int t = 0; t < in.therm_sweeps; ++t) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int r = 0; r < R; ++r) {
                heatbath_sweep(geom, reps[r], T_list[r], in.J, H_list[r]);
            }
            if ((t + 1) % in.swap_every == 0) {
                attempt_swaps(reps, T_list, H_list, in.J);
            }
        }

        // Measurement bins
        std::vector<std::vector<double>> m_bins(R), mabs_bins(R);
        for (int r = 0; r < R; ++r) {
            int cap = in.meas_sweeps / in.bin_size + 2;
            m_bins[r].reserve(cap);
            mabs_bins[r].reserve(cap);
        }

        std::vector<double> m_acc(R, 0.0), mabs_acc(R, 0.0);
        int acc_count = 0;

        for (int t = 0; t < in.meas_sweeps; ++t) {
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int r = 0; r < R; ++r) {
                heatbath_sweep(geom, reps[r], T_list[r], in.J, H_list[r]);
            }
            if ((t + 1) % in.swap_every == 0) {
               attempt_swaps(reps, T_list, H_list, in.J);
            }

            for (int r = 0; r < R; ++r) {
                double m = (double)reps[r].M / (double)N;
                m_acc[r] += m;
                mabs_acc[r] += std::fabs(m);
            }
            acc_count++;

            if (acc_count == in.bin_size) {
                for (int r = 0; r < R; ++r) {
                    m_bins[r].push_back(m_acc[r] / (double)in.bin_size);
                    mabs_bins[r].push_back(mabs_acc[r] / (double)in.bin_size);
                    m_acc[r] = 0.0;
                    mabs_acc[r] = 0.0;
                }
                acc_count = 0;
            }
        }

        // Output
        for (int r = 0; r < R; ++r) {
            double m_mean, m_err, mabs_mean, mabs_err;
            mean_and_stderr(m_bins[r], m_mean, m_err);
            mean_and_stderr(mabs_bins[r], mabs_mean, mabs_err);
            double he = heff(h, T_list[r], in.Jab, in.mu_a, in.mu_b);
            out << h << "," << T_list[r] << "," << (hasT0 ? T0 : std::numeric_limits<double>::quiet_NaN())
                << "," << he << "," << (in.mu_a * he)
                << "," << m_mean << "," << m_err
                << "," << mabs_mean << "," << mabs_err
                << "," << (int)m_bins[r].size() << "\n";
        }

        std::cout << "done h=" << h << " with R=" << R << " temperatures\n";
    }

    std::cout << "Wrote M_surface.csv\n";
    return 0;
}

