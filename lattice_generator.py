"""
lattice_generator.py
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



# Geometry builders


def build_zigzag(N: int, a: float):
    """
    ZGNR with N zigzag chains.

    Convention:
      – translation direction : x
      – ribbon width          : y
      – unit cell period      : T = sqrt(3)*a  along x

    The unit cell contains 2*N atoms (N per sublattice).

    Atom layout (y increases upward):
      The y-positions follow alternating gaps of a/2 (intra-dimer bond)
      and a (inter-chain bond): 0, a/2, 3a/2, 2a, 3a, 7a/2, …
      x-positions alternate in pairs: 0, T/2, T/2, 0, 0, T/2, …
    """
    T = np.sqrt(3) * a
    positions, labels = [], []
    y = 0.0
    for n in range(N):
        if n % 2 == 0:
            positions.append([0.0,     y])
            positions.append([T / 2.0, y + a / 2.0])
        else:
            positions.append([T / 2.0, y])
            positions.append([0.0,     y + a / 2.0])
        labels += [0, 1]
        y += a / 2.0 + a          # gap to next chain = inter-chain bond length a

    positions = np.array(positions)
    labels    = np.array(labels, dtype=int)
    a1 = np.array([T, 0.0])
    a2 = np.array([0.0, 0.0])
    return positions, labels, a1, a2


def build_armchair(N: int, a: float):
    """
    AGNR with N dimer lines.

    Convention:
      – translation direction : y
      – ribbon width          : x
      – unit cell period      : T = 3*a  along y

    The unit cell contains 2*N atoms.
    """
    T  = 3.0 * a
    dx = a * np.sqrt(3) / 2.0
    positions, labels = [], []
    for n in range(N):
        x = n * dx
        if n % 2 == 0:
            positions.append([x, 0.0])
            positions.append([x, a])
        else:
            positions.append([x, 3.0 * a / 2.0])  
            positions.append([x, 5.0 * a / 2.0])  
        labels += [0, 1]    

    positions = np.array(positions)
    labels    = np.array(labels, dtype=int)
    a1 = np.array([0.0, T])
    a2 = np.array([0.0, 0.0])
    return positions, labels, a1, a2



# Neighbour finder


def find_neighbors(positions: np.ndarray, labels, a1: np.ndarray, a: float,
                   tol: float = 0.05):
    """
    Return list of nearest-neighbour pairs considering the unit cell and
    images at +/- a1.

    Storage convention (important for Hamiltonian builder):
      cell == 0  : intra-cell bonds, stored ONCE with i < j.
      cell == +1 : forward inter-cell bonds (atom j in next cell).
      cell == -1 : reverse inter-cell bonds (atom j in previous cell).
      Inter-cell bonds are stored in BOTH directions so that the Hamiltonian
      builder can iterate without manually adding conjugates for
      cross-cell terms.
    """
    bond_len = a
    M = len(positions)
    neighbors = []

    for cell_image in [0, 1, -1]:
        shift = cell_image * a1
        for i in range(M):
            for j in range(M):
                if labels[i] == labels[j]:
                    continue
                if cell_image == 0 and i >= j:
                    continue
                disp = positions[j] + shift - positions[i]
                dist = np.linalg.norm(disp)
                if abs(dist - bond_len) < tol * bond_len:
                    neighbors.append(dict(
                        i    = i,
                        j    = j,
                        cell = cell_image,
                        dx   = float(disp[0]),
                        dy   = float(disp[1]),
                    ))
    return neighbors



# Visualisation


def plot_ribbon(positions, labels, a1, neighbors, edge_type, N, out_stem):
    """Save a schematic PNG showing three unit cells side by side."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: "#e05c5c", 1: "#5c8be0"}

    for rep in range(3):
        shift = rep * a1
        for nb in neighbors:
            img = nb["cell"]
            if img == 0 or (img == 1 and rep < 2) or (img == -1 and rep > 0):
                xi, yi = positions[nb["i"]] + shift
                xj = positions[nb["j"]][0] + shift[0] + img * a1[0]
                yj = positions[nb["j"]][1] + shift[1] + img * a1[1]
                ax.plot([xi, xj], [yi, yj], "k-", lw=1.2, zorder=1)
        for idx, (pos, lab) in enumerate(zip(positions, labels)):
            x, y = pos + shift
            ax.scatter(x, y, s=120, c=colors[lab], zorder=2,
                       edgecolors="k", linewidths=0.5)
            if rep == 0:
                ax.text(x, y + 0.08, str(idx), ha="center",
                        va="bottom", fontsize=6)

    if np.abs(a1[0]) > 0:
        for rep in range(4):
            ax.axvline(rep * a1[0], ls="--", color="gray", lw=0.8)
    else:
        for rep in range(4):
            ax.axhline(rep * a1[1], ls="--", color="gray", lw=0.8)

    legend = [mpatches.Patch(color=colors[0], label="Sublattice A"),
               mpatches.Patch(color=colors[1], label="Sublattice B")]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(f"{edge_type.capitalize()} GNR, N={N}  (3 unit cells shown)")
    ax.set_aspect("equal")
    ax.set_xlabel("x  (Å)")
    ax.set_ylabel("y  (Å)")
    ax.margins(0.1)
    plt.tight_layout()

    png_path = out_stem + ".png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"  Schematic saved → {png_path}")
    return png_path



# Save & load helpers


def save_npz(positions, labels, a1, a2, neighbors, path):
    nb_arr = np.array(
        [(nb["i"], nb["j"], nb["cell"], nb["dx"], nb["dy"]) for nb in neighbors],
        dtype=[("i", int), ("j", int), ("cell", int), ("dx", float), ("dy", float)],
    )
    np.savez(path,
             positions         = positions,
             sublattice_labels = labels,
             unit_cell_vectors = np.stack([a1, a2]),
             neighbors         = nb_arr)
    print(f"  Data saved       → {path}.npz")



# Main

def generate(edge_type: str, N: int, a: float = 1.42, out: str = None):
    edge_type = edge_type.lower()
    if edge_type == "zigzag":
        positions, labels, a1, a2 = build_zigzag(N, a)
    elif edge_type == "armchair":
        positions, labels, a1, a2 = build_armchair(N, a)
    else:
        raise ValueError(f"Unknown edge_type '{edge_type}'. Choose 'zigzag' or 'armchair'.")

    neighbors = find_neighbors(positions, labels, a1, a, tol=0.05)

    print(f"\n{'='*50}")
    print(f"  {edge_type.upper()} GNR   N={N}   a={a} Å")
    print(f"  Atoms per unit cell : {len(positions)}")
    print(f"  NN pairs found      : {len(neighbors)}")
    print(f"  Translation vector  : {a1}")

    if out:
        save_npz(positions, labels, a1, a2, neighbors, out)
        plot_ribbon(positions, labels, a1, neighbors, edge_type, N, out)

    return positions, labels, a1, neighbors


def main():
    parser = argparse.ArgumentParser(
        description="Generate GNR lattice data (.npz) and schematic (.png).")
    parser.add_argument("--edge", default="armchair",
                        choices=["zigzag", "armchair"])
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--a", type=float, default=1.42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = args.out or f"{args.edge}_N{args.N}"
    generate(args.edge, args.N, args.a, out)


if __name__ == "__main__":
    main()
