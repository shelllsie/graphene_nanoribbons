"""
gap_vs_width.py
---------------
Sweeps ribbon widths N = [Nmin..Nmax] and computes fundamental band gap
for graphene nanoribbon (GNR) tight-binding model.

The gap is defined as:
    E_gap = min_k E_{LUMO}(k) - max_k E_{HOMO}(k)

where HOMO/LUMO are the highest occupied / lowest unoccupied bands at
half-filling (particle-hole symmetric filling, i.e. the M/2 bands below
and above zero energy).

Outputs
-------
  <out>.csv   : columns  N, gap_eV
  <out>.png   : two-panel figure — linear gap vs N (top), log gap vs N (bottom)

Usage
-----
  python gap_vs_width.py --edge armchair --Nmin 3 --Nmax 30 --t 2.7 --Nk 300
  python gap_vs_width.py --edge zigzag   --Nmin 2 --Nmax 20 --t 2.7 --Nk 400
  python gap_vs_width.py --edge armchair --Nmin 3 --Nmax 60 --out agnr_gap --t2 0.1

"""
from __future__ import annotations

import argparse
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lattice_generator import generate
from hamiltonian_k import build_Hk, kpath



# main computations

def compute_gap(edge_type: str, N: int, t: float, Nk: int,
                t2: float = 0.0, delta: float = 0.0,
                a: float = 1.42) -> float:
    """
    Return fundamental band gap (eV) for a GNR of given edge type and width.

    Build the lattice in-memory (no .npz I/O) and diagonalise H(k) on a
    grid of Nk points spanning the full Brillouin zone.
    """
    positions, labels, a1, neighbors = generate(edge_type, N, a=a, out=None)
    T = float(np.linalg.norm(a1))
    M = len(positions)

    lattice = dict(
        M         = M,
        T         = T,
        a1        = a1,
        positions = positions,
        labels    = labels,
        neighbors = neighbors,
        nnn       = [],
    )

    homo_idx = M // 2 - 1   # highest occupied band 
    lumo_idx = M // 2       # lowest  unoccupied band

    homo_max = -np.inf
    lumo_min =  np.inf

    Nk_odd = Nk if Nk % 2 == 1 else Nk + 1

    for k in kpath(Nk_odd, T, endpoint=True):
        evals = np.linalg.eigvalsh(
            build_Hk(lattice, t=t, k=k, t2=t2, delta=delta)
        )
        if evals[homo_idx] > homo_max:
            homo_max = evals[homo_idx]
        if evals[lumo_idx] < lumo_min:
            lumo_min = evals[lumo_idx]

    gap = lumo_min - homo_max

    # Clamp floating-point noise on gapless systems. 
    NOISE_FLOOR = 1e-10   # eV — well below any physically meaningful gap
    return 0.0 if abs(gap) < NOISE_FLOOR else float(gap)


def sweep(edge_type: str, N_values: list[int], t: float, Nk: int,
          t2: float = 0.0, delta: float = 0.0,
          a: float = 1.42) -> tuple[np.ndarray, np.ndarray]:
    """
    Run compute_gap for each N in N_values.  Returns (N_arr, gap_arr).
    """
    N_arr   = np.array(N_values, dtype=int)
    gap_arr = np.empty(len(N_values), dtype=float)
    w       = len(str(len(N_values)))

    for idx, N in enumerate(N_values):
        gap = compute_gap(edge_type, N, t=t, Nk=Nk, t2=t2, delta=delta, a=a)
        gap_arr[idx] = gap
        print(f"  [{idx+1:{w}}/{len(N_values)}]  "
              f"{edge_type.upper()} N={N:>4d}  gap = {gap:.6f} eV")

    return N_arr, gap_arr



# helpers


def save_csv(N_arr: np.ndarray, gap_arr: np.ndarray, path: str) -> None:
    np.savetxt(path,
               np.column_stack([N_arr, gap_arr]),
               delimiter=",", header="N,gap_eV", comments="",
               fmt=["%d", "%.8f"])
    print(f"Saved CSV  → {path}")


def plot_gap(N_arr: np.ndarray, gap_arr: np.ndarray,
             edge_type: str, t: float, t2: float,
             out_png: str, dpi: int = 150) -> None:
    """
    Two-panel figure: linear (top) and log-scale (bottom) gap vs N.

    For armchair ribbons the three families 3p / 3p+1 / 3p+2 are
    highlighted with distinct markers and colours.  Metallic ribbons
    (gap == 0) are shown as downward-pointing arrows on the log panel
    rather than floating at an arbitrary floor value.

    When all ribbons are metallic (e.g. ZGNR at NN level), the log panel
    is replaced by a text annotation to avoid a meaningless flat line.
    """
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    fig.subplots_adjust(hspace=0.06)

    # Separate metallic (gap == 0) from semiconducting entries
    metallic  = gap_arr == 0.0
    semico    = ~metallic
    all_metal = metallic.all()

    # For log panel, plot semiconducting points; annotate metallic
    # ones with downward arrows at a fixed y position.
    LOG_ARROW_Y = None  

    # ── colour / marker scheme ───────────────────────────────────────────
    def _family_scatter(ax, y_data, mask_extra=None):
        """Plot points with family colouring (AGNR) or uniform (ZGNR)."""
        if edge_type == "armchair":
            families = [
                (N_arr % 3 == 0, "o", "#e05c5c", r"$3p$"),
                (N_arr % 3 == 1, "s", "#5c8be0", r"$3p+1$"),
                (N_arr % 3 == 2, "^", "#5cb85c", r"$3p+2$"),
            ]
            for fmask, marker, color, label in families:
                sel = fmask if mask_extra is None else fmask & mask_extra
                if sel.any():
                    ax.scatter(N_arr[sel], y_data[sel],
                               marker=marker, color=color, label=label,
                               s=48, zorder=3, linewidths=0.5, edgecolors="k")
            ax.legend(title="Family", fontsize=8, title_fontsize=8, framealpha=0.85)
        else:
            sel = np.ones(len(N_arr), dtype=bool) if mask_extra is None else mask_extra
            if sel.any():
                ax.scatter(N_arr[sel], y_data[sel], marker="o", color="#5c8be0",
                           s=48, zorder=3, linewidths=0.5, edgecolors="k")
                ax.plot(N_arr[sel], y_data[sel],
                        color="#5c8be0", lw=0.8, alpha=0.5, zorder=2)

    # ── linear panel (top)
    ax_lin = axes[0]
    _family_scatter(ax_lin, gap_arr)
    ax_lin.set_yscale("linear")
    ax_lin.set_ylabel("Gap (eV)", fontsize=10)
    ax_lin.axhline(0, color="0.75", lw=0.8, ls="--")
    ax_lin.grid(which="major", ls=":", lw=0.5, alpha=0.6)

    # ── log panel (bottom) 
    ax_log = axes[1]

    if all_metal:
        # Nothing meaningful, remove the axes entirely,
        # expand the linear panel to fill the figure, and add note below.
        fig.delaxes(ax_log)
        ax_lin.set_position([0.12, 0.18, 0.83, 0.68])
        fig.text(0.5, 0.07,
                 "All ribbons metallic (gap = 0 at NN level)  —  "
                 "add NNN hopping (--t2) to open a gap",
                 ha="center", va="center", fontsize=9,
                 color="0.45", style="italic",
                 bbox=dict(boxstyle="round,pad=0.35", fc="0.97", ec="0.82"))
    else:
        # Plot only semiconducting points on log scale
        log_y = gap_arr.copy()
        _family_scatter(ax_log, log_y, mask_extra=semico)

        ax_log.set_yscale("log")
        ax_log.set_ylabel("Gap (eV)", fontsize=10)
        ax_log.grid(which="major", ls=":", lw=0.5, alpha=0.6)
        ax_log.grid(which="minor", ls=":", lw=0.3, alpha=0.4)
        ax_log.yaxis.set_major_formatter(
            ticker.LogFormatterSciNotation(labelOnlyBase=False))

        # Annotate metallic points with downward arrows below the plot minimum.
        # Must be placed after set_yscale so axis limits are meaningful.
        if metallic.any():
            y_lo, y_hi = ax_log.get_ylim()
            arrow_tip  = y_lo * 0.6          # just below visible range
            arrow_base = y_lo * 2.5
            for n_val in N_arr[metallic]:
                ax_log.annotate("", xy=(n_val, arrow_tip),
                                xytext=(n_val, arrow_base),
                                arrowprops=dict(arrowstyle="-|>",
                                                color="0.55", lw=1.2))
            ax_log.set_ylim(bottom=y_lo * 0.3)   # make room for arrows
            ax_log.plot([], [], color="0.55", marker="|", lw=0,
                        markersize=8, label="metallic (gap = 0)")
            handles, labels_leg = ax_log.get_legend_handles_labels()
            if handles:
                ax_log.legend(handles=handles, labels=labels_leg,
                              title="Family", fontsize=8, title_fontsize=8,
                              framealpha=0.85)

    # xlabel goes on whichever is the bottom-most visible axes
    x_ax = ax_lin if all_metal else ax_log
    x_ax.set_xlabel("Width parameter $N$", fontsize=10)
    x_ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=12))
    x_ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    title = f"{edge_type.capitalize()} GNR  |  $t = {t}$ eV"
    if t2:
        title += f"  |  $t_2 = {t2}$ eV"
    ax_lin.set_title(title, fontsize=10)

    if all_metal:
        # tight_layout would override the manual ax_lin position set above,
        # so save directly without calling it.
        fig.savefig(out_png, dpi=dpi)
    else:
        plt.tight_layout()
        fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Saved plot → {out_png}")



# CLI


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep GNR width N and compute the fundamental band gap."
    )
    p.add_argument("--edge",  default="armchair", choices=["zigzag", "armchair"],
                   help="Edge type (default: armchair)")
    p.add_argument("--Nmin",  type=int, default=2,
                   help="Minimum N (default: 2)")
    p.add_argument("--Nmax",  type=int, default=30,
                   help="Maximum N (default: 30)")
    p.add_argument("--t",     type=float, default=2.7,
                   help="NN hopping in eV (default: 2.7)")
    p.add_argument("--t2",    type=float, default=0.0,
                   help="NNN hopping in eV (default: 0)")
    p.add_argument("--delta", type=float, default=0.0,
                   help="Staggered on-site potential in eV (default: 0)")
    p.add_argument("--Nk",    type=int, default=300,
                   help="k-grid resolution (default: 300)")
    p.add_argument("--a",     type=float, default=1.42,
                   help="C-C bond length in Å (default: 1.42)")
    p.add_argument("--out",   default=None,
                   help="Output stem (default: <edge>_gap)")
    p.add_argument("--dpi",   type=int, default=150,
                   help="Figure DPI (default: 150)")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.out or f"{args.edge}_gap"

    if args.Nmin < 1:
        print("Error: --Nmin must be >= 1", file=sys.stderr); sys.exit(1)
    if args.Nmax < args.Nmin:
        print("Error: --Nmax must be >= --Nmin", file=sys.stderr); sys.exit(1)

    N_values = list(range(args.Nmin, args.Nmax + 1))

    print(f"\n{'='*52}")
    print(f"  Edge   : {args.edge.upper()}")
    print(f"  N      : {args.Nmin} → {args.Nmax}  ({len(N_values)} ribbons)")
    print(f"  t      : {args.t} eV   t2 = {args.t2} eV   Δ = {args.delta} eV")
    print(f"  Nk     : {args.Nk}   a = {args.a} Å")
    print(f"  Output : {out}.csv / {out}.png")
    print(f"{'='*52}\n")

    N_arr, gap_arr = sweep(args.edge, N_values,
                           t=args.t, Nk=args.Nk,
                           t2=args.t2, delta=args.delta, a=args.a)

    save_csv(N_arr, gap_arr, out + ".csv")
    plot_gap(N_arr, gap_arr, args.edge, t=args.t, t2=args.t2,
             out_png=out + ".png", dpi=args.dpi)

    # Summary
    print(f"\n  Max gap : {gap_arr.max():.4f} eV  (N = {N_arr[np.argmax(gap_arr)]})")
    print(f"  Min gap : {gap_arr.min():.6f} eV  (N = {N_arr[np.argmin(gap_arr)]})")
    print()


if __name__ == "__main__":
    main()
