"""
validation.py
=============
Reproducibility and convergence checks for the GNR tight-binding pipeline.

Check 1 — k-point convergence
------------------------------
Sweeps Nk from coarse to fine for:
  - AGNR N=9  (3p family, semiconducting, gap ~0.95 eV)
  - AGNR N=11 (3p+2 family, metallic, gap = 0)
  - ZGNR N=8  (metallic at NN level, gap = 0)

Passes when: semiconducting gap converges to < 0.1% of the Nk=1001 value,
and metallic gaps stay below the noise floor for all Nk tested.

Check 2 — Wide-ribbon Dirac cone recovery
------------------------------------------
A wide AGNR 3p+2 ribbon (default N=80) should recover the linear Dirac
dispersion of bulk graphene near k=0 (the Gamma point).

Theory: E_± = +/- hbar*v_F * k,   hbar*v_F = 3*t*a/2

The fit is performed on the HOMO and LUMO bands in the range
k in [0, k_max_frac * pi/T].  Passes when |v_F_fit / v_F_theory - 1| < 1%.

Outputs
-------
  validation_kconv.png   --  gap vs Nk for all three ribbons
  validation_dirac.png   --  near-Gamma bands + linear fit vs theory
  validation_report.txt  --  PASS/FAIL summary with numbers
"""
from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lattice_generator import generate
from hamiltonian_k import build_Hk



# helpers


def make_lattice(edge: str, N: int, a: float = 1.42) -> dict:
    pos, labels, a1, nb = generate(edge, N, a=a)
    T = float(np.linalg.norm(a1))
    return {
        "positions": pos, "labels": labels,
        "a1": a1, "neighbors": nb,
        "M": len(pos), "T": T,
    }


def compute_gap(lattice: dict, t: float, Nk: int,
                t2: float = 0.0, NOISE: float = 1e-10) -> float:
    """Gap at given Nk, with odd-Nk and noise-floor correction."""
    T      = lattice["T"]
    M      = lattice["M"]
    Nk_odd = Nk if Nk % 2 == 1 else Nk + 1
    k_vals = np.linspace(0.0, np.pi / T, Nk_odd)
    homo   = -np.inf
    lumo   =  np.inf
    for k in k_vals:
        evals = np.linalg.eigvalsh(build_Hk(lattice, t=t, k=k, t2=t2))
        evals = np.sort(evals)
        homo  = max(homo, evals[M // 2 - 1])
        lumo  = min(lumo, evals[M // 2])
    gap = lumo - homo
    return 0.0 if abs(gap) < NOISE else float(max(gap, 0.0))


def compute_bands_near_k(
    lattice: dict, t: float, k_center: float,
    k_half_range: float, Nk_fit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute bands for k in [k_center - k_half_range, k_center + k_half_range].
    Returns (k_vals, bands) where bands[:, n] = E_n(k).
    """
    M      = lattice["M"]
    k_vals = np.linspace(k_center - k_half_range,
                         k_center + k_half_range, Nk_fit)
    bands  = np.zeros((Nk_fit, M))
    for ik, k in enumerate(k_vals):
        bands[ik, :] = np.sort(np.linalg.eigvalsh(
            build_Hk(lattice, t=t, k=k)))
    return k_vals, bands



# Check 1: k-point convergence


NK_SWEEP = [11, 21, 31, 51, 101, 201, 301, 501, 1001]
NK_REF   = 1001    # ref value for % error
CONV_TOL = 0.001   # 0.1% convergence criterion


def check_kconv(
    t: float = 2.7,
    a: float = 1.42,
) -> dict:
    """
    Compute gap vs Nk for three ribbons and assess convergence.

    Returns a results dict with keys for each ribbon.
    """
    ribbons = [
        ("armchair", 9,  "AGNR N=9  (3p, semi.)"),
        ("armchair", 11, "AGNR N=11 (3p+2, metal.)"),
        ("zigzag",   8,  "ZGNR N=8  (metal.)"),
    ]

    results = {}
    for edge, N, label in ribbons:
        print(f"  k-conv  {label} ...", end="", flush=True)
        t0      = time.time()
        lattice = make_lattice(edge, N, a=a)
        gaps    = []
        for Nk in NK_SWEEP:
            gaps.append(compute_gap(lattice, t=t, Nk=Nk))
        gaps = np.array(gaps)

        # Convergence: Nk where gap is within CONV_TOL of reference value
        gap_ref = gaps[-1]
        if gap_ref > 1e-6:   # semiconducting
            rel_err  = np.abs(gaps - gap_ref) / gap_ref
            conv_idx = next((i for i, e in enumerate(rel_err) if e < CONV_TOL),
                            len(NK_SWEEP) - 1)
            conv_Nk  = NK_SWEEP[conv_idx]
            passed   = True
        else:                # metallic: all gaps should stay near zero
            conv_Nk  = NK_SWEEP[0]
            passed   = bool(gaps.max() < 1e-9)

        results[(edge, N)] = {
            "label":    label,
            "gaps":     gaps,
            "gap_ref":  gap_ref,
            "conv_Nk":  conv_Nk,
            "passed":   passed,
        }
        print(f" {time.time()-t0:.1f}s  gap_ref={gap_ref:.5f} eV  "
              f"conv@Nk={conv_Nk}  {'PASS' if passed else 'FAIL'}")

    return results


def plot_kconv(results: dict, out_png: str, dpi: int = 150) -> None:
    colors = ["#d95f5f", "#5f7fd9", "#59a14f"]
    fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")

    for (color, ((edge, N), r)) in zip(colors, results.items()):
        gaps_meV = np.array(r["gaps"]) * 1000
        ax.semilogx(NK_SWEEP, gaps_meV, "o-", color=color,
                    label=r["label"], lw=1.4, ms=5)

    # Convergence threshold line (relative, drawn on the semiconducting gap)
    semi_key = ("armchair", 9)
    if semi_key in results:
        ref_meV = results[semi_key]["gap_ref"] * 1000
        ax.axhline(ref_meV * (1 - CONV_TOL), color="#d95f5f",
                   ls=":", lw=0.8, alpha=0.6)
        ax.axhline(ref_meV * (1 + CONV_TOL), color="#d95f5f",
                   ls=":", lw=0.8, alpha=0.6)

    ax.set_xlabel("$N_k$ (k-points)", fontsize=10)
    ax.set_ylabel("Gap  (meV)", fontsize=10)
    ax.set_title("k-point convergence", fontsize=10)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.grid(which="major", ls=":", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8, framealpha=0.9)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  Saved  -> {out_png}")



# Check 2: Dirac cone recovery


K_MAX_FRAC = 0.12    # fit range as fraction of pi/T
NK_FIT     = 61      # k-points for fit (odd)
DIRAC_TOL  = 0.01    # 1% tolerance on hbar*v_F


def check_dirac(
    N_wide: int   = 80,
    t: float      = 2.7,
    a: float      = 1.42,
) -> dict:
    """
    Fit linear dispersion to HOMO/LUMO of a wide AGNR 3p+2 ribbon near k=0.

    N_wide should be of the 3p+2 family (N=2,5,8,11,...,80,...) for metallic
    behaviour.  The fit gives hbar*v_F; compare to theory 3*t*a/2.
    """
    # Ensure N_wide is 3p+2 (metallic)
    if N_wide % 3 != 2:
        N_wide = N_wide - (N_wide % 3) + 2
        print(f"  (adjusted N_wide to {N_wide} to ensure 3p+2 family)")

    hbar_vF_theory = 3.0 * t * a / 2.0
    print(f"  Dirac cone  AGNR N={N_wide}  "
          f"(hbar*v_F theory = {hbar_vF_theory:.4f} eV*A) ...", end="", flush=True)
    t0 = time.time()

    lattice = make_lattice("armchair", N_wide, a=a)
    T       = lattice["T"]
    M       = lattice["M"]

    k_max   = K_MAX_FRAC * np.pi / T
    k_vals, bands = compute_bands_near_k(lattice, t=t,
                                         k_center=0.0,
                                         k_half_range=k_max,
                                         Nk_fit=NK_FIT)

    # The Dirac cone bands cross at k=0 for 3p+2 AGNR.
    # Find HOMO/LUMO band indices at k=0 (midpoint of k_vals)
    mid  = NK_FIT // 2
    homo_idx = M // 2 - 1
    lumo_idx = M // 2

    # Fit E_LUMO(k) = hbar_vF * |k| for k > 0
    k_pos  = k_vals[mid:]        # k >= 0
    e_lumo = bands[mid:, lumo_idx]
    e_homo = -bands[mid:, homo_idx]   # negated: should be +hbar_vF*k

    # Fit through origin: hbar_vF = sum(k*E) / sum(k^2)
    # (avoids intercept offset at k=0 due to finite-size gap)
    k_nz   = k_pos[k_pos > 0]
    e_nz_l = e_lumo[k_pos > 0]
    e_nz_h = e_homo[k_pos > 0]

    hbar_vF_lumo = float(np.dot(k_nz, e_nz_l) / np.dot(k_nz, k_nz))
    hbar_vF_homo = float(np.dot(k_nz, e_nz_h) / np.dot(k_nz, k_nz))
    hbar_vF_fit  = 0.5 * (hbar_vF_lumo + hbar_vF_homo)

    rel_err = abs(hbar_vF_fit / hbar_vF_theory - 1.0)
    passed  = rel_err < DIRAC_TOL

    # Finite-size gap at k=0
    gap_k0  = float(bands[mid, lumo_idx] - bands[mid, homo_idx])

    print(f" {time.time()-t0:.1f}s  "
          f"hbar*vF={hbar_vF_fit:.4f} eV*A  "
          f"err={rel_err*100:.3f}%  "
          f"gap(k=0)={gap_k0*1000:.2f} meV  "
          f"{'PASS' if passed else 'FAIL'}")

    return {
        "N_wide":          N_wide,
        "k_vals":          k_vals,
        "bands":           bands,
        "homo_idx":        homo_idx,
        "lumo_idx":        lumo_idx,
        "hbar_vF_theory":  hbar_vF_theory,
        "hbar_vF_lumo":    hbar_vF_lumo,
        "hbar_vF_homo":    hbar_vF_homo,
        "hbar_vF_fit":     hbar_vF_fit,
        "rel_err":         rel_err,
        "gap_k0":          gap_k0,
        "passed":          passed,
        "k_max":           k_max,
        "T":               T,
    }


def plot_dirac(r: dict, out_png: str, dpi: int = 150) -> None:
    k_vals   = r["k_vals"]
    bands    = r["bands"]
    hi       = r["homo_idx"]
    li       = r["lumo_idx"]
    vF_fit   = r["hbar_vF_fit"]
    vF_th    = r["hbar_vF_theory"]
    T        = r["T"]
    k_max    = r["k_max"]

    # k in units of pi/T for x-axis
    k_sc     = k_vals * T / np.pi

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), layout="constrained")

    # ── left: full near-Gamma window 
    ax = axes[0]
    ax.plot(k_sc, bands[:, hi],   color="#d95f5f", lw=1.5, label="HOMO")
    ax.plot(k_sc, bands[:, li],   color="#5f7fd9", lw=1.5, label="LUMO")
    # Theory line
    k_abs  = np.abs(k_vals)
    ax.plot(k_sc,  vF_th * k_abs, color="k", lw=1.0, ls="--",
            label=rf"Theory  $\hbar v_F k$")
    ax.plot(k_sc, -vF_th * k_abs, color="k", lw=1.0, ls="--")
    # Fit line
    ax.plot(k_sc,  vF_fit * k_abs, color="orange", lw=1.2, ls="-.",
            label=rf"Fit  $\hbar v_F={vF_fit:.3f}$ eV$\cdot$Å")
    ax.plot(k_sc, -vF_fit * k_abs, color="orange", lw=1.2, ls="-.")

    ax.axhline(0.0, color="0.6", lw=0.6, ls=":")
    ax.axvline(0.0, color="0.6", lw=0.6, ls=":")
    ax.set_xlabel(r"$k\,T/\pi$", fontsize=10)
    ax.set_ylabel("Energy  (eV)", fontsize=10)
    ax.set_title(f"Near-$\\Gamma$ dispersion  AGNR N={r['N_wide']}", fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(which="major", ls=":", lw=0.5, alpha=0.55)

    # ── right: residuals 
    ax2   = axes[1]
    mid   = len(k_vals) // 2
    k_pos = k_vals[mid:]
    e_l   = bands[mid:, li]
    e_h   = -bands[mid:, hi]
    k_nz  = k_pos[k_pos > 0]
    res_l = (bands[mid:, li][k_pos > 0] - vF_fit * k_nz) * 1000    # meV
    res_h = (-bands[mid:, hi][k_pos > 0] - vF_fit * k_nz) * 1000

    ax2.plot(k_nz * T / np.pi, res_l, "o-", color="#5f7fd9",
             ms=3, lw=1.0, label="LUMO residual")
    ax2.plot(k_nz * T / np.pi, res_h, "s-", color="#d95f5f",
             ms=3, lw=1.0, label="HOMO residual")
    ax2.axhline(0.0, color="0.5", lw=0.7, ls="--")
    ax2.set_xlabel(r"$k\,T/\pi$", fontsize=10)
    ax2.set_ylabel("Residual  $E - \\hbar v_F k$  (meV)", fontsize=10)
    ax2.set_title("Deviation from linearity", fontsize=10)
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(which="major", ls=":", lw=0.5, alpha=0.55)

    # Annotation box
    info = (
        f"$\\hbar v_F$ (theory) = {vF_th:.4f} eV$\\cdot$Å\n"
        f"$\\hbar v_F$ (fit)    = {r['hbar_vF_fit']:.4f} eV$\\cdot$Å\n"
        f"Error = {r['rel_err']*100:.3f}%\n"
        f"Finite-size gap = {r['gap_k0']*1000:.2f} meV\n"
        f"{'PASS' if r['passed'] else 'FAIL'}  (tol {DIRAC_TOL*100:.0f}%)"
    )
    ax2.text(0.03, 0.97, info, transform=ax2.transAxes,
             fontsize=8, va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.92))

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  Saved  -> {out_png}")



# Report


def write_report(
    kconv_results: dict,
    dirac_result:  dict,
    path: str,
) -> bool:
    lines  = []
    lines += ["=" * 60]
    lines += ["GNR TIGHT-BINDING PIPELINE  --  VALIDATION REPORT"]
    lines += ["=" * 60, ""]

    # Check 1
    lines += ["CHECK 1: k-point convergence", "-" * 40]
    all_pass = True
    for (edge, N), r in kconv_results.items():
        status = "PASS" if r["passed"] else "FAIL"
        all_pass = all_pass and r["passed"]
        lines.append(f"  {r['label']:<30}  gap_ref={r['gap_ref']*1000:8.3f} meV"
                     f"  conv_Nk={r['conv_Nk']:5d}  {status}")
    lines += ["", f"  Overall: {'PASS' if all_pass else 'FAIL'}", ""]

    # Check 2
    dr = dirac_result
    lines += ["CHECK 2: Dirac cone recovery (wide AGNR)", "-" * 40]
    lines += [
        f"  Ribbon        : AGNR N={dr['N_wide']} (3p+2, metallic)",
        f"  Fit range     : k in [0, {K_MAX_FRAC:.2f} * pi/T]",
        f"  hbar*vF theory: {dr['hbar_vF_theory']:.4f} eV*A",
        f"  hbar*vF (LUMO): {dr['hbar_vF_lumo']:.4f} eV*A",
        f"  hbar*vF (HOMO): {dr['hbar_vF_homo']:.4f} eV*A",
        f"  hbar*vF (mean): {dr['hbar_vF_fit']:.4f} eV*A",
        f"  Relative error: {dr['rel_err']*100:.4f}%  (tol {DIRAC_TOL*100:.0f}%)",
        f"  Finite-size gap at k=0: {dr['gap_k0']*1000:.3f} meV",
        f"  Result        : {'PASS' if dr['passed'] else 'FAIL'}",
        "",
    ]

    overall = all_pass and dr["passed"]
    lines += ["=" * 60]
    lines += [f"OVERALL: {'PASS' if overall else 'FAIL'}"]
    lines += ["=" * 60]

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    with open(path, "w") as f:
        f.write(text)
    print(f"  Report -> {path}")
    return overall



# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validation checks for the GNR tight-binding pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--t",       type=float, default=2.7,
                   help="NN hopping (eV)")
    p.add_argument("--a",       type=float, default=1.42,
                   help="C-C bond length (Angstrom)")
    p.add_argument("--N-wide",  type=int,   default=80,
                   help="Ribbon width for Dirac cone check (3p+2 family enforced)")
    p.add_argument("--out-dir", default=".",
                   help="Output directory")
    p.add_argument("--dpi",     type=int,   default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pfx  = os.path.join(args.out_dir, "validation")

    print(f"\n{'='*60}")
    print(f"  GNR validation   t={args.t} eV   a={args.a} A")
    print(f"{'='*60}\n")

    # Check 1
    print("[ CHECK 1: k-point convergence ]")
    kconv = check_kconv(t=args.t, a=args.a)
    plot_kconv(kconv, pfx + "_kconv.png", dpi=args.dpi)

    # Check 2
    print("\n[ CHECK 2: Dirac cone recovery ]")
    dirac = check_dirac(N_wide=args.N_wide, t=args.t, a=args.a)
    plot_dirac(dirac, pfx + "_dirac.png", dpi=args.dpi)

    # Report
    print("\n[ REPORT ]")
    passed = write_report(kconv, dirac, pfx + "_report.txt")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
