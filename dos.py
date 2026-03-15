"""
dos.py
======
Compute the densiy of states (DoS) from GNR band energies.

Two methods are available and can be used simultaneously:
  Gaussian k-sum  -- each eigenvalue E_n(k) contributes a Gaussian of width
                     sigma to a fine energy grid.  Physically this represents
                     the spectral function broadened by disorder or temperature.
  Histogram       -- raw count of eigenvalues per energy bin (no broadening).

Inputs
------
  <bands>.csv  from bandstructure.py  (columns: k_invA, kT_over_pi, E_0 ...)

Outputs
-------
  <out>.png  -- DoS vs E plot (Gaussian and/or histogram)
  <out>.csv  -- energy grid and DoS values

Usage
-----
  # Default: Gaussian broadening only
  python dos.py --csv zgnr_N8_bands.csv --out zgnr_N8_dos

  # Both methods side by side
  python dos.py --csv agnr_N10_bands.csv --sigma 0.05 --histogram --out agnr_N10_dos

  # Narrow broadening to resolve van Hove singularities
  python dos.py --csv agnr_N10_bands.csv --sigma 0.01 --Ne 4000 --out agnr_N10_dos_fine
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



#  Load band energies from CSV


def load_bands_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Read bandstructure CSV (from bandstructure.py).

    Returns (k_vals, bands) where bands[ik, n] = E_n(k_vals[ik]).
    """
    data   = np.loadtxt(path, delimiter=",", skiprows=1)
    k_vals = data[:, 0]          # col 0: k in inverse Angstrom
    bands  = data[:, 2:]         # cols 2+: E_0, E_1, ...
    return k_vals, bands



#   Gaussian k-sum DoS


def dos_gaussian(
    bands:  np.ndarray,
    E_grid: np.ndarray,
    sigma:  float,
) -> np.ndarray:
    """
    DoS(E) = (1/N_k) * sum_{n,k}  G_sigma(E - E_n(k))

    where G_sigma is a normalised Gaussian.  The 1/N_k prefactor ensures the
    total integrated DoS equals the number of bands (= M states per unit cell
    per spin, ignoring spin degeneracy here).

    Implementation: vectorised over the energy grid; loops over k to avoid
    an O(Nk * M * Ne) memory allocation that would be large for fine grids.
    """
    Nk    = bands.shape[0]
    Ne    = len(E_grid)
    dos   = np.zeros(Ne)
    inv_s = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    inv_2s2 = 1.0 / (2.0 * sigma**2)

    # Flatten all eigenvalues at each k into one vector, then accumulate.
    for ik in range(Nk):
        evals = bands[ik, :]           # shape (M,)
        # Broadcast: (Ne, 1) - (1, M) = (Ne, M)
        dE  = E_grid[:, np.newaxis] - evals[np.newaxis, :]
        dos += (inv_s * np.exp(-dE**2 * inv_2s2)).sum(axis=1)

    dos /= Nk      # normalise by number of k-points
    return dos



#  Histogram DoS


def dos_histogram(
    bands:  np.ndarray,
    E_grid: np.ndarray,
) -> np.ndarray:
    """
    DoS as a simple histogram of all eigenvalues.

    Bin centres are taken as E_grid; bin edges are at the midpoints.
    The result is normalised so that sum(dos) * dE = M (total bands).
    """
    all_evals = bands.flatten()
    dE        = E_grid[1] - E_grid[0]
    edges     = np.concatenate([
        [E_grid[0] - dE / 2],
        (E_grid[:-1] + E_grid[1:]) / 2,
        [E_grid[-1] + dE / 2],
    ])
    counts, _ = np.histogram(all_evals, bins=edges)
    # Normalise: integral = M
    M   = bands.shape[1]
    Nk  = bands.shape[0]
    dos = counts / (Nk * dE)      # states per eV per unit cell
    return dos.astype(float)



#  Auto-choose sigma


def auto_sigma(bands: np.ndarray) -> float:
    """
    Heuristic default: sigma = 0.5 * total_bandwidth / Nk.

    This gives ~1-2 pixels of broadening at typical Nk=300, enough to smooth
    the k-discretisation noise without washing out band-edge structure.
    """
    Nk        = bands.shape[0]
    bandwidth = bands.max() - bands.min()
    return max(0.5 * bandwidth / Nk, 0.01)   # floor at 10 meV



#  Save CSV


def save_csv(E_grid: np.ndarray, dos_g: np.ndarray | None,
             dos_h: np.ndarray | None, path: str) -> None:
    cols    = [E_grid]
    header  = "energy_eV"
    if dos_g is not None:
        cols.append(dos_g)
        header += ",dos_gaussian"
    if dos_h is not None:
        cols.append(dos_h)
        header += ",dos_histogram"
    np.savetxt(path, np.column_stack(cols),
               delimiter=",", header=header, comments="")
    print(f"Saved CSV  -> {path}")



#  Plot


def _fermi_region(E_grid, bands):
    """Return (E_lo, E_hi) for the HOMO-LUMO gap shading."""
    M    = bands.shape[1]
    homo = bands[:, M // 2 - 1].max()
    lumo = bands[:, M // 2].min()
    return homo, lumo


def plot_dos(
    E_grid:    np.ndarray,
    dos_g:     np.ndarray | None,
    dos_h:     np.ndarray | None,
    bands:     np.ndarray,
    sigma:     float,
    out_png:   str,
    title:     str | None = None,
    dpi:       int = 150,
) -> None:
    """
    Single-panel DoS plot.

    Gaussian DoS: filled curve.  Histogram: step outline.
    Fermi region (gap) shaded if the gap > 10 meV.
    Van Hove singularity positions (band-edge energies) marked with ticks.
    """
    homo, lumo = _fermi_region(E_grid, bands)
    gap        = max(lumo - homo, 0.0)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), layout="constrained")

    # Histogram underlay
    if dos_h is not None:
        ax.fill_between(E_grid, dos_h, step="mid",
                        color="0.80", alpha=0.55, label="Histogram")
        ax.step(E_grid, dos_h, where="mid",
                color="0.55", lw=0.7, alpha=0.8)

    # Gaussian DoS
    if dos_g is not None:
        ax.fill_between(E_grid, dos_g,
                        color="#5f7fd9", alpha=0.35, label=f"Gaussian ($\u03c3={sigma:.3f}$ eV)")
        ax.plot(E_grid, dos_g, color="#3a5db0", lw=1.3)

    # Gap shading
    if gap > 0.01:
        ax.axvspan(homo, lumo, color="#ffd080", alpha=0.45, zorder=0,
                   label=f"Gap = {gap:.3f} eV")

    # Fermi level line
    ax.axvline(0.0, color="0.4", lw=0.8, ls="--", label="$E_F = 0$")

    # Van Hove tick marks: band edges = local extrema of each band
    vH_energies = set()
    for n in range(bands.shape[1]):
        b = bands[:, n]
        vH_energies.add(round(float(b.min()), 5))
        vH_energies.add(round(float(b.max()), 5))
    E_lo, E_hi = E_grid[0], E_grid[-1]
    vH_in_range = [e for e in vH_energies if E_lo <= e <= E_hi]
    y_max = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for e_vH in vH_in_range:
        ax.axvline(e_vH, color="0.7", lw=0.4, ls=":", zorder=0)

    ax.set_xlabel("Energy  (eV)", fontsize=10)
    ax.set_ylabel("DoS  (states / eV / unit cell)", fontsize=9)
    ax.set_xlim(E_lo, E_hi)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(which="major", ls=":", lw=0.5, alpha=0.55)
    ax.legend(fontsize=8, framealpha=0.9)

    if title:
        ax.set_title(title, fontsize=10)

    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot -> {out_png}")



# 7.  CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute GNR density of states from band CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv",   required=True,
                   help="Band CSV from bandstructure.py")
    p.add_argument("--sigma", type=float, default=None,
                   help="Gaussian broadening width (eV).  "
                        "Default: 0.5 * bandwidth / Nk (auto)")
    p.add_argument("--Ne",    type=int,   default=2000,
                   help="Number of energy grid points")
    p.add_argument("--Emin",  type=float, default=None,
                   help="Energy grid minimum (eV).  Default: auto")
    p.add_argument("--Emax",  type=float, default=None,
                   help="Energy grid maximum (eV).  Default: auto")
    p.add_argument("--histogram", action="store_true",
                   help="Also compute and overlay histogram DoS")
    p.add_argument("--out",   default=None,
                   help="Output file stem (default: derived from --csv)")
    p.add_argument("--dpi",   type=int, default=150)
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    k_vals, bands = load_bands_csv(args.csv)
    Nk, M   = bands.shape

    csv_stem = os.path.splitext(os.path.basename(args.csv))[0]
    stem     = args.out or csv_stem + "_dos"

    # Energy grid
    E_lo = args.Emin if args.Emin is not None else bands.min() - 0.5
    E_hi = args.Emax if args.Emax is not None else bands.max() + 0.5
    E_grid = np.linspace(E_lo, E_hi, args.Ne)
    dE     = E_grid[1] - E_grid[0]

    # Broadening
    sigma = args.sigma if args.sigma is not None else auto_sigma(bands)

    print(f"\n{'='*56}")
    print(f"  CSV    : {args.csv}  ({Nk} k-pts, {M} bands)")
    print(f"  E grid : [{E_lo:.3f}, {E_hi:.3f}] eV   Ne={args.Ne}   dE={dE*1000:.1f} meV")
    print(f"  sigma  : {sigma:.4f} eV  ({'auto' if args.sigma is None else 'user'})")
    print(f"  Output : {stem}.png / {stem}.csv")
    print(f"{'='*56}\n")

    # Compute
    print("  Computing Gaussian DoS ...", end=" ", flush=True)
    dos_g = dos_gaussian(bands, E_grid, sigma)
    print("done")

    dos_h = None
    if args.histogram:
        print("  Computing histogram DoS ...", end=" ", flush=True)
        dos_h = dos_histogram(bands, E_grid)
        print("done")

    # Check normalisation: integral of DoS should equal M (bands per k-point)
    integral = float(np.trapezoid(dos_g, E_grid))
    print(f"  Integral check: {integral:.3f}  (expected {M}  =  M bands)")

    homo = bands[:, M // 2 - 1].max()
    lumo = bands[:, M // 2].min()
    gap  = max(lumo - homo, 0.0)
    print(f"  HOMO={homo:+.4f} eV   LUMO={lumo:+.4f} eV   gap={gap:.4f} eV")

    # Outputs
    save_csv(E_grid, dos_g, dos_h, stem + ".csv")

    t2_tag = ""   # not stored in CSV; omit from title
    title = f"DoS  |  {csv_stem}  (\u03c3={sigma:.3f} eV)"
    plot_dos(E_grid, dos_g, dos_h, bands, sigma,
             stem + ".png", title=title, dpi=args.dpi)


if __name__ == "__main__":
    main()
