#!/usr/bin/env python3
"""
bandstructure.py

Compute and plot the band structure E_n(k) of a graphene nanoribbon using the
Hamiltonian builder in hamiltonian_k.py.

Produces:
  - PNG plot of bands (k axis shown as k*T/pi, BZ edges at \pm 1)
  - Optional CSV file with energies (rows = k points, columns = bands)

Usage examples:
  py bandstructure.py --npz armchair_N10.npz --t 2.7 --Nk 300 --out bands_armchair_N10
  py bandstructure.py --npz zigzag_N8.npz --t 2.7 --Nk 400 --out bands_zigzag_N8 --dpi 200
  py bandstructure.py --npz armchair_N10.npz --t 2.7 --t2 0.1 --delta 0.2 --save-csv

Notes:
  - For moderate unit-cell sizes (M < few hundreds) the dense diagonalisation
    (numpy.linalg.eigvalsh) is used and is fast. For very large M, pass --sparse
    to build H(k) in sparse format.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

# Import utilities
try:
    from hamiltonian_k import build_Hk, build_Hk_sparse, kpath, load_lattice
except ImportError as exc:
    sys.exit(
        f"[ERROR] Could not import from hamiltonian_k.py. "
        f"Make sure it is on your PYTHONPATH.\n  {exc}"
    )



# Band computation


def compute_bands(
    lattice: dict,
    t: float,
    k_vals: np.ndarray,
    use_sparse: bool = False,
    t2: float = 0.0,
    delta: float = 0.0,
) -> np.ndarray:
    """
    Diagonalise H(k) at every k in *k_vals* and return sorted eigenvalues.

    Parameters
    ----------
    lattice    : dict returned by load_lattice()
    t          : nearest-neighbour hopping (eV)
    k_vals     : 1-D array of k points (Å^-1)
    use_sparse : if True, build H(k) via build_Hk_sparse then convert to dense
    t2         : next-nearest-neighbour hopping (eV), default 0
    delta      : staggered on-site potential (eV), default 0

    Returns
    -------
    bands : (Nk, M) float array — eigenvalues sorted ascending for each k
    """
    M = lattice["M"]
    bands = np.empty((len(k_vals), M), dtype=float)

    for idx, k in enumerate(k_vals):
        if use_sparse:
            H = build_Hk_sparse(lattice, t=t, k=k, t2=t2, delta=delta).toarray()
        else:
            H = build_Hk(lattice, t=t, k=k, t2=t2, delta=delta)
        bands[idx] = np.linalg.eigvalsh(H)   # eigvalsh returns sorted ascending

    return bands



# Plotting


def plot_bands(
    k_vals: np.ndarray,
    bands: np.ndarray,
    lattice_T: float,
    out_png: str,
    title: str | None = None,
    dpi: int = 150,
) -> None:
    """
    Plot the band structure.

    The x-axis is k·T/π so that the Brillouin zone edges fall at ±1.
    A horizontal line marks E = 0 (Fermi level reference).
    Bands are coloured with a continuous colormap to avoid cycle repetition
    for large M.
    """
    k_scaled = k_vals * lattice_T / np.pi          # dimensionless, range [-1, +1]
    M = bands.shape[1]

    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    # Continuous colormap so colours never repeat for large M
    cmap = plt.cm.viridis
    colours = cmap(np.linspace(0.0, 1.0, M))

    for band_idx in range(M):
        ax.plot(k_scaled, bands[:, band_idx], linewidth=0.9, color=colours[band_idx])

    # Fermi level reference
    ax.axhline(0.0, color="crimson", linestyle="--", linewidth=0.9, label=r"$E = 0$")

    # BZ boundary markers
    ax.axvline( 1.0, color="0.55", linestyle=":", linewidth=0.7)
    ax.axvline(-1.0, color="0.55", linestyle=":", linewidth=0.7)
    ax.axvline( 0.0, color="0.55", linestyle="--", linewidth=0.7)

    ax.set_xlim(k_scaled[0], k_scaled[-1])
    ax.set_xlabel(r"$k \, T / \pi$  (BZ edges at $\pm 1$)", fontsize=11)
    ax.set_ylabel("Energy  (eV)", fontsize=11)
    if title:
        ax.set_title(title, fontsize=10)

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(which="major", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.7)

    plt.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"  Band plot  → {out_png}")



# CSV export


def save_bands_csv(
    k_vals: np.ndarray,
    bands: np.ndarray,
    lattice_T: float,
    out_csv: str,
) -> None:
    """
    Write k values and band energies to a CSV file.

    Columns
    -------
    k_phys  : k in physical units (Å^-1)
    k_scaled: k·T/pi (dimensionless; BZ edges at \pm 1, matches plot x-axis)
    band_0 … band_{M-1}: eigenvalues in eV, sorted ascending
    """
    k_scaled = k_vals * lattice_T / np.pi
    M = bands.shape[1]

    header = "k_phys(1/Å),k_scaled(kT/pi)," + ",".join(f"band_{i}" for i in range(M))
    data = np.hstack([k_vals.reshape(-1, 1), k_scaled.reshape(-1, 1), bands])
    np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
    print(f"  Bands CSV  → {out_csv}")



# CLI


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute and plot the band structure of a GNR from a .npz lattice file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--npz",     required=True,          help="Lattice .npz from lattice_generator.py")
    p.add_argument("--t",       type=float, default=2.7, help="Nearest-neighbour hopping (eV)")
    p.add_argument("--t2",      type=float, default=0.0, help="Next-nearest-neighbour hopping (eV)")
    p.add_argument("--delta",   type=float, default=0.0, help="Staggered on-site potential (eV)")
    p.add_argument("--Nk",      type=int,   default=300, help="Number of k-points across BZ")
    p.add_argument("--kstart",  type=float, default=None, help="Custom k start (Å⁻¹); overrides BZ default")
    p.add_argument("--kend",    type=float, default=None, help="Custom k end   (Å⁻¹); overrides BZ default")
    p.add_argument("--sparse",  action="store_true",     help="Build H(k) as sparse matrix")
    p.add_argument("--out",     default="bands",         help="Output filename stem (no extension)")
    p.add_argument("--dpi",     type=int,   default=150, help="Figure resolution (DPI)")
    p.add_argument("--save-csv", action="store_true",    help="Also export bands to CSV")
    return p.parse_args()


def build_title(npz_path: str, t: float, t2: float, delta: float, M: int) -> str:
    """Construct a plot title that includes all non-trivial physical parameters."""
    base = os.path.basename(npz_path)
    parts = [f"t = {t} eV"]
    if t2 != 0.0:
        parts.append(f"t₂ = {t2} eV")
    if delta != 0.0:
        parts.append(f"Δ = {delta} eV")
    params = ",  ".join(parts)
    return f"{base}  |  {params}  |  M = {M}"


def main() -> None:
    args = parse_args()

    # --- Load lattice with a clear error if file is missing ---
    if not os.path.isfile(args.npz):
        sys.exit(f"[ERROR] Lattice file not found: {args.npz}")
    lattice = load_lattice(args.npz)
    T = lattice["T"]
    M = lattice["M"]
    print(f"Loaded lattice : {args.npz}  (M = {M}, T = {T:.5f} Å)")

    # --- Build k grid ---
    if args.kstart is not None and args.kend is not None:
        k_vals = np.linspace(args.kstart, args.kend, args.Nk)
    else:
        k_vals = kpath(args.Nk, T)

    # --- Compute ---
    print(f"Computing bands for {len(k_vals)} k-points …")
    bands = compute_bands(
        lattice,
        t=args.t,
        k_vals=k_vals,
        use_sparse=args.sparse,
        t2=args.t2,
        delta=args.delta,
    )

    # --- Plots ---
    title = build_title(args.npz, args.t, args.t2, args.delta, M)
    plot_bands(k_vals, bands, lattice_T=T, out_png=args.out + ".png", title=title, dpi=args.dpi)

    # --- Opt CSV ---
    if args.save_csv:
        save_bands_csv(k_vals, bands, T, args.out + ".csv")

    print("Done.")


if __name__ == "__main__":
    main()
