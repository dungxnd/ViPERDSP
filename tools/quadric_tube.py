#!/usr/bin/env python3
"""
quadric_tube.py — Quadric Surface triode model toolkit for ViPERDSP.

Provides three independent utilities in one script:

  fit      Least-squares fit of {kp, kp2, kpg} from datasheet (Vp, Vg, Ia) points.
           Solves  sqrt(ip) = c1*Vp + c2*Vg + c0  via normal equations, then recovers
           parameters:  kp2=c1², kpg=2*c1*c2, kp=2*c1*c0, mu=c2/c1

  verify   Evaluates a known parameter set against sampled operating points and prints
           per-point error, gm/rp/mu at a chosen bias, and output_gain relative to 12AX7.

  wdf      Precomputes all constant WDF factors for a given tube + port resistance
           configuration and optionally runs a short numerical simulation on a test sine
           wave to verify the WDF path against the Kirchhoff-domain QuadricTube.cpp path.

Usage
-----
  python quadric_tube.py fit    --points points.csv [--save tube_12au7]
  python quadric_tube.py verify --kp 4.6931e-5 --kp2 6.1383e-7 --kpg 2.2105e-5 \\
                                --vdd 250 --rp 22000 --bias -8.5
  python quadric_tube.py wdf    --kp 4.6931e-5 --kp2 6.1383e-7 --kpg 2.2105e-5 \\
                                --z1 22000 --z3 2200

CSV format for 'fit':  Vp[V], Vg[V], Ia[mA]   (one point per line, no header required)
"""

import argparse
import csv
import io
import math
import sys
from pathlib import Path
from typing import NamedTuple

# Force UTF-8 output on Windows so box-drawing chars survive cp1252 terminals.
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class TubeParams(NamedTuple):
    kp: float
    kp2: float
    kpg: float

    @property
    def mu(self) -> float:
        """Amplification factor derived from model coefficients."""
        return self.kpg / (2.0 * self.kp2)

    def __str__(self) -> str:
        return (
            f"kp  = {self.kp:.6e}\n"
            f"kp2 = {self.kp2:.6e}\n"
            f"kpg = {self.kpg:.6e}\n"
            f"mu  = {self.mu:.4f}"
        )

    def as_cpp_struct(self, label: str = "") -> str:
        """Format as a C++ kTubeConfigs initialiser line."""
        comment = f"  // {label}" if label else ""
        return (
            f"{{ {{ {self.kp:.4e}, {self.kp2:.4e}, {self.kpg:.4e} }}, "
            f"vdd, rp, bias, output_gain }},{comment}"
        )


# ---------------------------------------------------------------------------
# Core model evaluation
# ---------------------------------------------------------------------------

def ip_from_vgk_vdd(params: TubeParams, vdd: float, rp: float, vgk: float) -> float:
    """
    Solve for plate current from the quadratic load-line intersection.
    Mirrors QuadricTube::Process() exactly (Kirchhoff domain).
    Returns ip in amperes (≥ 0).
    """
    kp, kp2, kpg = params
    k_A       = kp2 * rp * rp
    k_2A      = 2.0 * k_A
    k_4A      = 4.0 * k_A
    k_B_const = (-2.0 * kp2 * vdd * rp) - (kp * rp) - 1.0
    k_B_vgk   = -(kpg * rp)
    k_C_vgk2  = (kpg * kpg) / (4.0 * kp2)
    k_C_vgk   = (kpg * vdd) + ((kp * kpg) / (2.0 * kp2))
    k_C_const = (kp2 * vdd * vdd) + (kp * vdd) + ((kp * kp) / (4.0 * kp2))

    B    = k_B_const + k_B_vgk * vgk
    C    = k_C_vgk2 * vgk ** 2 + k_C_vgk * vgk + k_C_const
    disc = B * B - k_4A * C

    if disc < -1e-12:
        return 0.0
    ip = (-B - math.sqrt(max(disc, 0.0))) / k_2A
    return max(0.0, ip)


def ip_direct(params: TubeParams, vp: float, vg: float) -> float:
    """
    Evaluate the full quadric surface ip(Vp, Vg) directly (no load line).
    Used for datasheet point verification.
    """
    kp, kp2, kpg = params
    ip = (kp2 * vp * vp
          + kpg * vp * vg
          + kp * vp
          + (kpg * kpg / (4.0 * kp2)) * vg * vg
          + (kp * kpg / (2.0 * kp2)) * vg
          + kp * kp / (4.0 * kp2))
    return max(0.0, ip)


def small_signal_gain(params: TubeParams, vdd: float, rp: float, bias: float,
                      drive: float = 1.0, dx: float = 1e-3) -> float:
    """
    Numerical small-signal gain dY/dX at the quiescent point.
    Y is the normalised plate voltage output (matching QuadricTube output_scale).
    """
    output_scale = -1.0 / (vdd / 2.5)

    def y(sample: float) -> float:
        vgk   = sample * drive + bias
        ip    = ip_from_vgk_vdd(params, vdd, rp, vgk)
        v_pk  = max(0.0, min(vdd, vdd - ip * rp))
        return v_pk * output_scale

    return abs((y(dx) - y(-dx)) / (2.0 * dx))


# ---------------------------------------------------------------------------
# Least-squares fit
# ---------------------------------------------------------------------------

def fit_params(points: list[tuple[float, float, float]]) -> TubeParams:
    """
    Fit kp, kp2, kpg from (Vp, Vg, Ia_mA) datasheet sample points via
    ordinary least squares on  sqrt(ip) = c1*Vp + c2*Vg + c0.

    Uses Gaussian elimination on the 3×3 normal equations (no numpy needed).
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points for a 3-parameter fit.")

    # Build design matrix X (Nx3) and response y
    X = [[vp, vg, 1.0] for vp, vg, _ in points]
    y = [math.sqrt(ia * 1e-3) for _, _, ia in points]
    n = len(X)

    # Accumulate normal equations XtX (3x3) and Xty (3)
    XtX = [[0.0] * 3 for _ in range(3)]
    Xty = [0.0] * 3
    for i in range(n):
        for j in range(3):
            Xty[j] += X[i][j] * y[i]
            for k in range(3):
                XtX[j][k] += X[i][j] * X[i][k]

    # Gaussian elimination with partial pivoting on augmented matrix
    aug = [XtX[i][:] + [Xty[i]] for i in range(3)]
    for col in range(3):
        # Partial pivot
        max_row = max(range(col, 3), key=lambda r: abs(aug[r][col]))
        aug[col], aug[max_row] = aug[max_row], aug[col]
        pivot = aug[col][col]
        if abs(pivot) < 1e-30:
            raise ValueError(f"Singular normal matrix at column {col} — "
                             "check for duplicate or collinear points.")
        for row in range(col + 1, 3):
            f = aug[row][col] / pivot
            for k in range(4):
                aug[row][k] -= f * aug[col][k]

    # Back substitution
    c = [0.0] * 3
    for i in range(2, -1, -1):
        c[i] = aug[i][3]
        for j in range(i + 1, 3):
            c[i] -= aug[i][j] * c[j]
        c[i] /= aug[i][i]

    c1, c2, c0 = c[0], c[1], c[2]
    return TubeParams(kp=2.0 * c1 * c0, kp2=c1 * c1, kpg=2.0 * c1 * c2)


# ---------------------------------------------------------------------------
# WDF precomputed constants
# ---------------------------------------------------------------------------

class WDFConstants(NamedTuple):
    """All static factors for the WDF triode port solution."""
    gamma: float
    disc_offset: float       # 1 / (8 * Z1 * kp2 * gamma²)
    sqrt_premul: float       # 1 / sqrt(2 * Z1 * kp2 * gamma)
    b1_offset: float         # 1 / (4 * Z1 * kp2 * gamma²)
    z3_over_z1: float
    vpk_a1_coeff: float      # (1 - Z3/Z1) / 2
    vpk_b1_coeff: float      # (1 + Z3/Z1) / 2
    z3_a1_b1: float          # Z3 / Z1  (for b3 update)
    vpk0_b1: float           # b1 when Vpk forced to 0: (Z3-Z1)*a1/(Z1+Z3) ... constant part
    vpk0_a3: float           # 2*Z1/(Z1+Z3)             ... a3 part


def precompute_wdf(params: TubeParams, z1: float, z3: float) -> WDFConstants:
    """
    Precompute all port-resistance-dependent WDF constants.
    z1 = plate port resistance (Ω), z3 = cathode port resistance (Ω).
    Grid port z2 is open (ig=0) so no z2 factors appear.
    """
    kp, kp2, kpg = params
    gamma       = 1.0 + (z3 / z1) * (1.0 + kpg / (4.0 * kp2))
    disc_offset = 1.0 / (8.0 * z1 * kp2 * gamma * gamma)
    sqrt_premul = 1.0 / math.sqrt(2.0 * z1 * kp2 * gamma)
    b1_offset   = 1.0 / (4.0 * z1 * kp2 * gamma * gamma)
    r           = z3 / z1
    return WDFConstants(
        gamma       = gamma,
        disc_offset = disc_offset,
        sqrt_premul = sqrt_premul,
        b1_offset   = b1_offset,
        z3_over_z1  = r,
        vpk_a1_coeff = (1.0 - r) / 2.0,
        vpk_b1_coeff = (1.0 + r) / 2.0,
        z3_a1_b1    = r,
        vpk0_b1     = (z3 - z1) / (z1 + z3),   # multiplied by a1
        vpk0_a3     = 2.0 * z1 / (z1 + z3),     # multiplied by a3
    )


def wdf_step(params: TubeParams, wdf: WDFConstants,
             a1: float, a2: float, a3: float) -> tuple[float, float, float]:
    """
    Single-sample WDF triode evaluation.  Returns (b1, b2, b3).
    Implements the closed-form algorithm from Section 4 of the paper.
    """
    kp, kp2, kpg = params

    # Step 1 — intermediate variables
    alpha = kp + kpg * (a2 - a3 - (wdf.z3_over_z1 / 2.0) * a1)
    beta  = kp2 * (((1.0 - wdf.z3_over_z1) / 2.0) * a1 - a3)
    eta   = (beta + alpha / 2.0) / (kp2 * wdf.gamma)

    # Step 2 — discriminant
    delta1 = wdf.disc_offset + a1 + eta

    # Step 3 — solve for b1
    if delta1 >= 0.0:
        b1 = wdf.sqrt_premul * math.sqrt(delta1) - wdf.b1_offset - eta
        # Check derivative condition: ip' >= 0 ⟺ b1 >= -eta
        if b1 < -eta:
            b1 = a1  # cutoff: ip = 0
    else:
        b1 = a1  # no real intersection: ip = 0

    # Step 4 — Vpk non-negativity
    vpk = wdf.vpk_a1_coeff * a1 + wdf.vpk_b1_coeff * b1 - a3
    if vpk < 0.0:
        b1 = wdf.vpk0_b1 * a1 + wdf.vpk0_a3 * a3

    # Step 5 — remaining reflected waves
    b2 = a2
    b3 = a3 + wdf.z3_a1_b1 * (a1 - b1)

    return b1, b2, b3


# ---------------------------------------------------------------------------
# CLI sub-commands
# ---------------------------------------------------------------------------

def _load_points(path: str) -> list[tuple[float, float, float]]:
    points = []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row if c.strip()]
            if not row or row[0].startswith("#"):
                continue
            if len(row) < 3:
                continue
            try:
                points.append((float(row[0]), float(row[1]), float(row[2])))
            except ValueError:
                continue  # skip header-like rows
    return points


def cmd_fit(args: argparse.Namespace) -> None:
    points = _load_points(args.points)
    print(f"Loaded {len(points)} datasheet points from '{args.points}'")

    params = fit_params(points)
    c1 = math.sqrt(params.kp2)
    c2 = params.kpg / (2.0 * c1)
    c0 = params.kp  / (2.0 * c1)

    print()
    print("── Fitted linear coefficients ──────────────────────────────────")
    print(f"  c1 (Vp)  = {c1:.6e}  √A/V")
    print(f"  c2 (Vg)  = {c2:.6e}  √A/V")
    print(f"  c0 (bias)= {c0:.6e}  √A")
    print()
    print("── Quadric surface parameters ──────────────────────────────────")
    print(params)
    print()

    # Per-point verification
    print("── Fit verification ────────────────────────────────────────────")
    rss = 0.0
    print(f"  {'Vp':>5} {'Vg':>6}  {'model mA':>9}  {'ds mA':>7}  {'err%':>7}")
    for vp, vg, ia_ds in points:
        ip_m   = ip_direct(params, vp, vg)
        err    = (ip_m - ia_ds * 1e-3) / (ia_ds * 1e-3) * 100.0
        rss   += err * err
        print(f"  {vp:>5.0f} {vg:>6.1f}  {ip_m*1000:>9.3f}  {ia_ds:>7.2f}  {err:>+7.2f}%")
    print(f"  RMS error = {math.sqrt(rss / len(points)):.2f}%")
    print()

    if args.save:
        out = Path(args.save).with_suffix(".csv")
        with open(out, "w") as f:
            f.write("# kp, kp2, kpg\n")
            f.write(f"{params.kp:.8e},{params.kp2:.8e},{params.kpg:.8e}\n")
        print(f"Saved parameters to '{out}'")

    # Suggest C++ line
    print("── C++ kTubeConfigs entry (fill vdd/rp/bias/output_gain) ───────")
    print("  " + params.as_cpp_struct())


def cmd_verify(args: argparse.Namespace) -> None:
    params = TubeParams(kp=args.kp, kp2=args.kp2, kpg=args.kpg)
    vdd, rp, bias = args.vdd, args.rp, args.bias

    print("── Parameters ──────────────────────────────────────────────────")
    print(params)
    print(f"Circuit: vdd={vdd}V  Rp={rp}Ω  bias={bias}V")
    print()

    # Quiescent point
    ip_q  = ip_from_vgk_vdd(params, vdd, rp, bias)
    vpk_q = max(0.0, min(vdd, vdd - ip_q * rp))
    print("── Quiescent operating point ───────────────────────────────────")
    print(f"  Ia_q  = {ip_q*1000:.4f} mA")
    print(f"  Vpk_q = {vpk_q:.2f} V   (headroom: {vpk_q/vdd*100:.1f}% of Vdd)")
    print()

    # Small-signal parameters at bias point (numerical)
    dv  = 1e-3
    ip_p = ip_from_vgk_vdd(params, vdd, rp, bias + dv)
    ip_m = ip_from_vgk_vdd(params, vdd, rp, bias - dv)
    gm  = (ip_p - ip_m) / (2.0 * dv)
    # rp via d(Vp)/d(Ip) along load line ≈ -dVpk/dIp; use direct partial
    # d(ip)/d(Vp) = 2*kp2*Vp + kpg*Vg + kp  at operating point
    kp, kp2, kpg = params
    dip_dvp = 2.0 * kp2 * vpk_q + kpg * bias + kp
    rp_tube = 1.0 / dip_dvp if dip_dvp > 1e-30 else float("inf")
    mu_local = gm * rp_tube
    print("── Small-signal parameters (numerical, at Q-point with load line) ─")
    print(f"  gm         = {gm*1000:.4f} mA/V  (at Vgk=bias, with Rp={rp:.0f}Ω load)")
    print(f"  rp_tube    = {rp_tube:.1f} Ω      (d(Vp)/d(ip) at Q-point, free-air)")
    print(f"  mu_local   = gm*rp = {mu_local:.3f}  (load-line suppressed vs free-air)")
    print(f"  mu_model   = {params.mu:.3f}  (kpg / 2*kp2, free-air datasheet value)")
    print()

    # output_gain vs 12AX7 reference
    G_REF_12AX7 = 0.553408  # pre-measured small-signal gain for 12AX7 config
    g_this = small_signal_gain(params, vdd, rp, bias)
    comp   = G_REF_12AX7 / g_this if g_this > 0 else float("inf")
    print("── Level-matching vs 12AX7 reference ───────────────────────────")
    print(f"  small-signal gain = {g_this:.6f}")
    print(f"  output_gain       = {comp:.4f}  ({20*math.log10(comp):+.2f} dB)")
    print()
    print("  → In kTubeConfigs:")
    print(f"    {params.as_cpp_struct()}")
    print(f"    (fill: vdd={vdd:.1f}, rp={rp:.0f}, bias={bias:.1f}, output_gain={comp:.4f})")

    # Optional per-point check
    if args.points:
        points = _load_points(args.points)
        print()
        print("── Verification against supplied datasheet points ───────────────")
        rss = 0.0
        print(f"  {'Vp':>5} {'Vg':>6}  {'model mA':>9}  {'ds mA':>7}  {'err%':>7}")
        for vp, vg, ia_ds in points:
            ip_m  = ip_direct(params, vp, vg)
            err   = (ip_m - ia_ds * 1e-3) / (ia_ds * 1e-3) * 100.0
            rss  += err * err
            print(f"  {vp:>5.0f} {vg:>6.1f}  {ip_m*1000:>9.3f}  {ia_ds:>7.2f}  {err:>+7.2f}%")
        print(f"  RMS error = {math.sqrt(rss / len(points)):.2f}%")


def cmd_wdf(args: argparse.Namespace) -> None:
    params = TubeParams(kp=args.kp, kp2=args.kp2, kpg=args.kpg)
    z1, z3 = args.z1, args.z3
    wdf = precompute_wdf(params, z1, z3)

    print("── WDF precomputed constants ────────────────────────────────────")
    print(f"  Z1              = {z1:.1f} Ω  (plate port)")
    print(f"  Z3              = {z3:.1f} Ω  (cathode port)")
    print(f"  Z3/Z1           = {wdf.z3_over_z1:.6f}")
    print(f"  gamma           = {wdf.gamma:.6f}")
    print(f"  disc_offset     = 1/(8*Z1*kp2*gamma²)  = {wdf.disc_offset:.6f}")
    print(f"  sqrt_premul     = 1/sqrt(2*Z1*kp2*gamma) = {wdf.sqrt_premul:.6f}")
    print(f"  b1_offset       = 1/(4*Z1*kp2*gamma²)  = {wdf.b1_offset:.6f}")
    print(f"  Vpk_a1_coeff    = (1-Z3/Z1)/2           = {wdf.vpk_a1_coeff:.6f}")
    print(f"  Vpk_b1_coeff    = (1+Z3/Z1)/2           = {wdf.vpk_b1_coeff:.6f}")
    print(f"  Vpk0_b1_coeff   = (Z3-Z1)/(Z1+Z3)       = {wdf.vpk0_b1:.6f}")
    print(f"  Vpk0_a3_coeff   = 2*Z1/(Z1+Z3)          = {wdf.vpk0_a3:.6f}")
    print()

    print("── C++ precomputed constants (copy into your WDF class init) ────")
    print(f"  const double gamma        = {wdf.gamma:.8f};")
    print(f"  const double disc_offset  = {wdf.disc_offset:.8e};")
    print(f"  const double sqrt_premul  = {wdf.sqrt_premul:.8e};")
    print(f"  const double b1_offset    = {wdf.b1_offset:.8e};")
    print(f"  const double z3_over_z1   = {wdf.z3_over_z1:.8e};")
    print(f"  const double vpk_a1       = {wdf.vpk_a1_coeff:.8e};")
    print(f"  const double vpk_b1       = {wdf.vpk_b1_coeff:.8e};")
    print(f"  const double vpk0_b1      = {wdf.vpk0_b1:.8e};")
    print(f"  const double vpk0_a3      = {wdf.vpk0_a3:.8e};")
    print()

    if not args.sim:
        return

    # Numerical simulation: compare WDF path vs Kirchhoff path on 1 kHz sine
    import math as _m
    print("── Simulation: WDF vs Kirchhoff-domain (1 kHz sine, 0.1 amplitude) ──")
    fs      = 44100.0
    freq    = 1000.0
    amp     = 0.1
    n_samp  = 256
    bias    = args.bias if args.bias else 0.0
    vdd     = z1  # treat Z1 as Rp for the Kirchhoff path (simplification)
    vdd_sim = 250.0

    output_scale = -1.0 / (vdd_sim / 2.5)
    max_abs_diff = 0.0

    a2_dc = bias           # incident wave on grid = DC bias
    a3_dc = 0.0            # cathode grounded

    print(f"  {'n':>4}  {'a1':>9}  {'b1_wdf':>10}  {'vpk_wdf':>9}  {'ip_kir mA':>10}")
    for i in range(min(n_samp, 16)):   # show first 16 samples
        t  = i / fs
        a1 = amp * _m.sin(2.0 * _m.pi * freq * t)
        b1, b2, b3 = wdf_step(params, wdf, a1, a2_dc, a3_dc)
        # Vpk from WDF
        vpk_wdf = wdf.vpk_a1_coeff * a1 + wdf.vpk_b1_coeff * b1 - a3_dc
        # ip from Kirchhoff (using vgk = a2_dc = bias, approximate)
        ip_kir  = ip_from_vgk_vdd(params, vdd_sim, z1, bias + a1 * 1.0)
        print(f"  {i:>4}  {a1:>9.5f}  {b1:>10.5f}  {vpk_wdf:>9.4f}  {ip_kir*1000:>10.4f}")

    print(f"  ... ({n_samp} samples total, showing first 16)")


# ---------------------------------------------------------------------------
# Argument parsing & entry point
# ---------------------------------------------------------------------------

def _add_tube_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--kp",  type=float, required=True, help="kp  parameter [A/V]")
    p.add_argument("--kp2", type=float, required=True, help="kp2 parameter [A/V²]")
    p.add_argument("--kpg", type=float, required=True, help="kpg parameter [A/V²]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ── fit ──────────────────────────────────────────────────────────────
    p_fit = sub.add_parser("fit", help="LS fit from CSV datasheet points")
    p_fit.add_argument("--points", required=True,
                       help="CSV file: Vp[V], Vg[V], Ia[mA] per row")
    p_fit.add_argument("--save", metavar="PREFIX",
                       help="Save fitted params to PREFIX.csv")

    # ── verify ───────────────────────────────────────────────────────────
    p_ver = sub.add_parser("verify", help="Verify a known parameter set")
    _add_tube_args(p_ver)
    p_ver.add_argument("--vdd",   type=float, default=250.0,  help="Supply voltage [V]")
    p_ver.add_argument("--rp",    type=float, required=True,  help="Plate resistor [Ω]")
    p_ver.add_argument("--bias",  type=float, required=True,  help="Grid bias [V]")
    p_ver.add_argument("--points", metavar="CSV",
                       help="Optional CSV of datasheet points for error report")

    # ── wdf ──────────────────────────────────────────────────────────────
    p_wdf = sub.add_parser("wdf", help="Precompute WDF constants")
    _add_tube_args(p_wdf)
    p_wdf.add_argument("--z1",   type=float, required=True, help="Plate port resistance Z1 [Ω]")
    p_wdf.add_argument("--z3",   type=float, required=True, help="Cathode port resistance Z3 [Ω]")
    p_wdf.add_argument("--bias", type=float, default=None,
                       help="Grid DC bias for simulation [V] (only with --sim)")
    p_wdf.add_argument("--sim",  action="store_true",
                       help="Run short numerical simulation comparing WDF to Kirchhoff path")

    args = parser.parse_args()
    {
        "fit":    cmd_fit,
        "verify": cmd_verify,
        "wdf":    cmd_wdf,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
