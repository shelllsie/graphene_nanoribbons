"""
edge_state_visualise.py
=======================
Extract a Bloch eigenstate of H(k) for a graphene nanoribbon, compute its
probability density |\psi_i|^2 across the transverse direction, and plot the
spatial profile.

Eigenstate selection
--------------------
  (default)      : pick the state with |E| closest to zero at the chosen k
  --near-zero    : explicit alias for the default
  --idx N        : pick band N explicitly (0-based, sorted ascending in energy)

Transverse direction convention
--------------------------------
  ZGNR  (a1 ∥ x) : transverse = y  positions[:,1]
  AGNR  (a1 ∥ y) : transverse = x  positions[:,0]

Outputs
-------
  <stem>.png   two-panel figure
                 top    – per-atom |\psi_i|^2 vs transverse coord, A/B sublattice
                          coloured separately
                 bottom – per-slice \sum |\psi_i|^2 bar chart with exponential
                          localisation fit overlaid when state is edge-localised
  <stem>.csv   per-atom table  (with --save-csv)

Usage examples
--------------
  # ZGNR edge state at BZ boundary (flat band)
  python edge_state_visualise.py --npz zgnr_N8.npz --t 2.7 --k-frac 1.0

  # ZGNR at onset of flat band
  python edge_state_visualise.py --npz zgnr_N8.npz --t 2.7 --k-frac 0.667

  # AGNR, specific band index, mid-zone
  python edge_state_visualise.py --npz agnr_N10.npz --t 2.7 --idx 9 --k-frac 0.5

  # ZGNR with NNN hopping: finite localisation length
  python edge_state_visualise.py --npz zgnr_N12.npz --t 2.7 --t2 0.1 --k-frac 1.0
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from hamiltonian_k import load_lattice, build_Hk


#  Diagonalisation

def diagonalise(
    lattice: dict,
    k: float,
    t: float,
    t2: float = 0.0,
    delta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (eigenvalues, eigenvectors) of H(k), sorted ascending in energy.

    eigvecs[:, n]  is the n-th eigenstate (column vector, unit L2 norm).
    """
    H = build_Hk(lattice, t=t, k=k, t2=t2, delta=delta)
    vals, vecs = np.linalg.eigh(H)   # eigh exploits Hermitian structure
    order = np.argsort(vals)
    return vals[order], vecs[:, order]



#   Eigenstate selection


def select_band(
    vals: np.ndarray,
    vecs: np.ndarray,
    idx: int | None,
) -> tuple[float, np.ndarray, int]:
    """
    Return (energy, eigenvector, band_index).

    If idx is None, pick the eigenstate with |E| closest to zero.
    """
    M = len(vals)
    chosen = int(np.argmin(np.abs(vals))) if idx is None else idx
    if not (0 <= chosen < M):
        raise ValueError(f"--idx {chosen} out of range [0, {M-1}]")
    return float(vals[chosen]), vecs[:, chosen], chosen



#   Probability density and transverse profile


def site_probability(eigvec: np.ndarray) -> np.ndarray:
    """
    Per-site |\psi_i|^2, renormalised to sum to 1.

    eigh returns unit-norm vectors, so renormalisation is a no-op in practice,
    but guards against any tiny floating-point drift.
    """
    p = np.abs(eigvec) ** 2
    s = p.sum()
    return p / s if s > 0 else p


def transverse_coords(lattice: dict) -> tuple[np.ndarray, str]:
    """
    Transverse coordinate and axis label for each atom in the unit cell.

    ZGNR (a1 ∥ x): transverse = y  →  positions[:, 1]
    AGNR (a1 ∥ y): transverse = x  →  positions[:, 0]
    """
    a1  = lattice["a1"]
    pos = lattice["positions"]
    if abs(a1[0]) >= abs(a1[1]):
        return pos[:, 1].copy(), "Transverse  y"
    else:
        return pos[:, 0].copy(), "Transverse  x"


def slice_profile(
    trans: np.ndarray,
    prob:  np.ndarray,
    tol:   float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum probability over atoms sharing the same transverse coordinate (a "slice").

    Returns (sorted_unique_positions, summed_probabilities).
    """
    rounded = np.round(trans, 6)
    unique  = np.unique(rounded)
    rho     = np.array([prob[np.abs(rounded - u) < tol].sum() for u in unique])
    return unique, rho



#   Localisation diagnostics


def ipr(prob: np.ndarray) -> float:
    """
    Inverse Participation Ratio  IPR = \sum_i |\psi_i|^4.

    IPR = 1/M  → fully extended (uniform).
    IPR = 1    → perfectly localised on one site.
    Participation ratio PR = 1 / (M · IPR) ∈ [0, 1].
    """
    return float(np.dot(prob, prob))


def fit_localisation(
    sl_pos: np.ndarray,
    rho:    np.ndarray,
) -> dict | None:
    """
    Fit rho_slice(x) ~ A exp(-x/xi) from each edge.

    Fires independently on each edge when that edge weight exceeds 3x mean.
    If only a single slice carries weight (perfectly single-site localised),
    xi is reported as half the slice spacing rather than skipped — the state
    is still correctly flagged as edge-localised.

    Returns a dict with keys xi_left, xi_right, x_fit_*, y_fit_*,
    or None if neither edge is localised.
    """
    n = len(sl_pos)
    if n < 2:
        return None

    result: dict = {}
    half = max(n // 2, 2)
    spacing = float(np.min(np.diff(sl_pos))) if n > 1 else 1.0

    for side in ("left", "right"):
        if side == "left":
            x_loc = sl_pos[:half] - sl_pos[0]
            y_loc = rho[:half]
        else:
            x_loc = sl_pos[-1] - sl_pos[n - half:][::-1]
            y_loc = rho[n - half:][::-1]

        # Only proceed if this edge weight is significantly elevated
        if y_loc[0] < 3.0 * rho.mean():
            continue   # not localised on this edge; check the other

        good = y_loc > 1e-12 * y_loc[0]

        if good.sum() < 2:
            # Single-slice localisation: xi -> 0, report as half a spacing
            xi = spacing * 0.5
            A  = float(y_loc[0])
        else:
            try:
                slope, intercept = np.polyfit(x_loc[good], np.log(y_loc[good]), 1)
            except np.linalg.LinAlgError:
                continue
            if abs(slope) < 1e-12:
                continue
            xi = -1.0 / slope
            A  = float(np.exp(intercept))

        # Reject fits where xi > half the ribbon width
        W = float(sl_pos[-1] - sl_pos[0])
        if xi > W * 0.5:
            continue

        y_curve = A * np.exp(-x_loc / float(xi))
        # Clip curve to where it is above 1% of the peak value
        visible = y_curve >= 0.01 * A
        result[f"xi_{side}"]    = float(xi)
        result[f"A_{side}"]     = float(A)
        result[f"x_fit_{side}"] = x_loc[visible]
        result[f"y_fit_{side}"] = y_curve[visible]

    return result if result else None



#  Figure

_COLOR = {0: "#d95f5f", 1: "#5f7fd9"}   # sublattice A=red, B=blue


def plot(
    lattice:  dict,
    prob:     np.ndarray,
    energy:   float,
    band_idx: int,
    k_frac:   float,
    out_png:  str,
    dpi:      int = 150,
) -> None:
    """
    Two-panel figure saved to out_png.

    Top    – per-atom |\psi_i|² as a stem+scatter plot, coloured by sublattice.
    Bottom – per-slice bar chart \sum|\psi_i|², with exponential fit if localised.
    """
    labels        = lattice["labels"]
    trans, x_lbl  = transverse_coords(lattice)
    M             = lattice["M"]
    ipr_val       = ipr(prob)

    sl_pos, rho   = slice_profile(trans, prob)
    loc_fit       = fit_localisation(sl_pos, rho)

    state_type = "edge-localised" if loc_fit else "extended / bulk"
    pr         = 1.0 / (ipr_val * M)

    t_lo  = sl_pos[0]
    t_hi  = sl_pos[-1]
    W     = t_hi - t_lo
    pad   = max(W * 0.04, 0.2)

    # ── layout 
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(7.5, 6.5),
        layout="constrained",
    )
    fig.get_layout_engine().set(h_pad=0.08, hspace=0.05)

    # ── top: per-atom 
    for sub in (0, 1):
        mask = labels == sub
        col  = _COLOR[sub]
        lbl  = "A" if sub == 0 else "B"
        ax_top.vlines(trans[mask], 0, prob[mask],
                      colors=col, lw=1.2, alpha=0.75)
        ax_top.scatter(trans[mask], prob[mask],
                       color=col, s=28, zorder=5, label=f"Sub-lattice {lbl}")
    ax_top.axhline(0, color="0.65", lw=0.6)

    ax_top.set_xlim(t_lo - pad, t_hi + pad)
    ax_top.set_ylim(bottom=-0.015 * prob.max())
    ax_top.set_xlabel(x_lbl, fontsize=9)
    ax_top.set_ylabel(r"$|\psi_i|^2$", fontsize=10)
    ax_top.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_top.grid(which="major", ls=":", lw=0.5, alpha=0.55)
    ax_top.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Annotation box
    info = (
        f"$E = {energy:+.4f}$ eV\n"
        f"band {band_idx},   $k = {k_frac:.4f}\\,\\pi/T$\n"
        f"IPR $= {ipr_val:.4f}$,   PR $= {pr:.3f}$\n"
        f"({state_type})"
    )
    ax_top.text(
        0.02, 0.97, info, transform=ax_top.transAxes,
        fontsize=7.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.92),
    )

    # ── bottom: per-slice 
    bw = np.min(np.diff(sl_pos)) * 0.55 if len(sl_pos) > 1 else 0.4

    ax_bot.bar(
        sl_pos, rho, width=bw,
        color="#7aabcc", edgecolor="#3a6080", linewidth=0.5,
        label=r"$\rho_{\rm slice}$", zorder=2,
    )

    # Shade outermost 15% on each side to flag edge region
    for x0, x1 in [
        (t_lo, t_lo + 0.15 * W),
        (t_hi - 0.15 * W, t_hi),
    ]:
        ax_bot.axvspan(x0, x1, alpha=0.10, color="orange", zorder=0)

    # Exponential fit
    if loc_fit:
        fit_colors = {"left": "#c04040", "right": "#4040c0"}
        for side in ("left", "right"):
            if f"xi_{side}" not in loc_fit:
                continue
            xi   = loc_fit[f"xi_{side}"]
            xf   = loc_fit[f"x_fit_{side}"]
            yf   = loc_fit[f"y_fit_{side}"]
            # Map local x back to absolute transverse position
            if side == "left":
                x_abs = xf + sl_pos[0]
                y_abs = yf
            else:
                x_abs = sl_pos[-1] - xf[::-1]
                y_abs = yf[::-1]
            ax_bot.plot(
                x_abs, y_abs, "--", lw=1.6,
                color=fit_colors[side], zorder=4,
                label=rf"fit  $\xi_{{\rm {side}}} = {xi:.2f}$ Å",
            )
        ax_bot.legend(fontsize=8, framealpha=0.9)

    ax_bot.set_xlim(t_lo - pad, t_hi + pad)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlabel(x_lbl, fontsize=9)
    ax_bot.set_ylabel(
        r"$\rho_{\rm slice} = \sum_{i \in {\rm slice}} |\psi_i|^2$",
        fontsize=9,
    )
    ax_bot.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_bot.grid(which="major", ls=":", lw=0.5, alpha=0.55, zorder=1)

    # ── suptitle 
    edge_tag = lattice.get("edge_type", "GNR")
    t_tag    = lattice.get("t_nn", "?")
    t2_tag   = lattice.get("t2", 0.0)
    title    = f"Probability density  |  {edge_tag}   $t = {t_tag}$ eV"
    if t2_tag:
        title += f",  $t_2 = {t2_tag}$ eV"
    fig.suptitle(title, fontsize=10, y=0.988)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot → {out_png}")



#  CSV ouput


def save_csv(
    lattice:  dict,
    prob:     np.ndarray,
    energy:   float,
    band_idx: int,
    k:        float,
    path:     str,
) -> None:
    trans, _ = transverse_coords(lattice)
    lbl      = lattice["labels"]
    header   = ("atom_idx,transverse_coord_A,sublattice,"
                "prob_density,energy_eV,band_idx,k_invA")
    rows = [
        f"{i},{trans[i]:.6f},{lbl[i]},{prob[i]:.10e},"
        f"{energy:.6f},{band_idx},{k:.6f}"
        for i in range(lattice["M"])
    ]
    with open(path, "w") as fh:
        fh.write(header + "\n" + "\n".join(rows) + "\n")
    print(f"Saved CSV  → {path}")



#  CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise GNR eigenstate probability density |\psi_i|^2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz",   required=True,
                   help="Lattice .npz from lattice_generator.py")
    p.add_argument("--t",     type=float, default=2.7,
                   help="NN hopping  t  (eV)")
    p.add_argument("--t2",    type=float, default=0.0,
                   help="NNN hopping  t₂  (eV)")
    p.add_argument("--delta", type=float, default=0.0,
                   help="Staggered on-site potential  Δ  (eV)")
    p.add_argument("--k-frac", type=float, default=1.0,
                   help=(
                       "k as a fraction of pi/T.  "
                       "k=0 → \Gamma point,  k=1 → BZ boundary.  "
                       "ZGNR flat edge band occupies [2/3, 1]."
                   ))

    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--near-zero", action="store_true",
                     help="Select eigenstate with |E| closest to 0  (default).")
    sel.add_argument("--idx", type=int,
                     help="Select eigenstate by band index (0-based, "
                          "sorted ascending in energy).")

    p.add_argument("--out",      default=None,
                   help="Output file stem  (default: derived from --npz).")
    p.add_argument("--dpi",      type=int, default=150)
    p.add_argument("--save-csv", action="store_true",
                   help="Also write per-atom data to a CSV file.")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    lattice = load_lattice(args.npz)
    T       = lattice["T"]
    M       = lattice["M"]
    k       = args.k_frac * np.pi / T

    # Stash for plot title
    npz_stem             = os.path.splitext(os.path.basename(args.npz))[0]
    lattice["edge_type"] = npz_stem
    lattice["t_nn"]      = args.t
    lattice["t2"]        = args.t2

    print(f"\n{'='*58}")
    print(f"  Lattice  :  {args.npz}  (M={M} atoms,  T={T:.4f} Å)")
    print(f"  k        :  {args.k_frac:.4f} × π/T  =  {k:.5f} Å⁻¹")
    print(f"  t, t₂, Δ :  {args.t},  {args.t2},  {args.delta}  eV")
    print(f"{'='*58}\n")

    vals, vecs = diagonalise(lattice, k=k, t=args.t, t2=args.t2, delta=args.delta)

    homo_i = M // 2 - 1
    lumo_i = M // 2
    print(f"  Eigenvalues (eV)   "
          f"HOMO = {vals[homo_i]:+.4f}   LUMO = {vals[lumo_i]:+.4f}")
    print(f"  {'idx':>4}  {'E (eV)':>12}")
    print(f"  {'----':>4}  {'----------':>12}")
    for i, e in enumerate(vals):
        tag = ("← HOMO" if i == homo_i else
               "← LUMO" if i == lumo_i else "")
        print(f"  {i:>4}  {e:>+12.6f}  {tag}")
    print()

    energy, eigvec, chosen = select_band(vals, vecs, idx=args.idx)
    prob    = site_probability(eigvec)
    ipr_val = ipr(prob)

    print(f"  Selected  →  band {chosen},  E = {energy:+.6f} eV")
    print(f"  IPR = {ipr_val:.6f}   "
          f"(1/M = {1/M:.6f}  =  fully extended limit)")

    k_tag = f"{args.k_frac:.3f}".replace(".", "p")
    stem  = (args.out or npz_stem)
    fstem = f"{stem}_band{chosen}_k{k_tag}"

    plot(lattice, prob, energy, chosen, args.k_frac,
         fstem + ".png", dpi=args.dpi)

    if args.save_csv:
        save_csv(lattice, prob, energy, chosen, k, fstem + ".csv")

    print()


if __name__ == "__main__":
    main()
