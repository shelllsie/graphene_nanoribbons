"""
hamiltonian_k.py
----------------------
Build the k-dependent Bloch Hamiltonian H(k) for a graphene nanoribbon (GNR)
from the lattice data produced by lattice_generator.py.

Physical model
--------------
Nearest-neighbour tight-binding on the honeycomb lattice following the
conventions of Castro Neto et al., Rev. Mod. Phys. 81, 109 (2009).

  H(k) = -t  Σ_{<i,j>, n}  e^{ikn}  |i><j|  +  h.c.

where the sum runs over all NN pairs (i,j) in the neighbour list,
n ∈ {0, ±1} is the cell-image index along the periodic direction, and
k is the 1-D Bloch wavevector conjugate to the translation vector a1.

k convention
------------
k is the physical wavevector in units of Å⁻¹.  The Brillouin zone
boundary is at k = ±π/T where T = |a1| is the unit-cell period.

Helper `kpath(N_k, T)` returns a uniformly spaced array
spanning the full BZ.

Extensions
---------------------------
--nnn / t2   : add next-nearest-neighbour hopping 
--onsite M   : add a staggered on-site potential ±M on A/B sublattices               

Outputs
-------
  Dense  : numpy ndarray, shape (M, M), dtype complex128
  Sparse : scipy.sparse.csr_matrix  (via --sparse flag or build_Hk_sparse())

The eigenvalues of H(k) give the band structure.  Use `np.linalg.eigvalsh`
to exploit Hermitian structure and guarantee real output.

CLI usage
---------
# Print eigenvalues at a single k point:
python hamiltonian_builder.py --npz zgnr_N4.npz --t 2.7 --k 0.0

# Save H(k) matrix (dense) to .npz at k = π/(2T):
python hamiltonian_builder.py --npz zgnr_N4.npz --t 2.7 --k 1.28 --out H_k

# Use sparse format:
python hamiltonian_builder.py --npz zgnr_N4.npz --t 2.7 --k 0.0 --sparse

Module usage
------------
from hamiltonian_builder import load_lattice, build_Hk, kpath

lattice = load_lattice("zgnr_N4.npz")
k_arr   = kpath(200, lattice["T"])
bands   = np.array([np.linalg.eigvalsh(build_Hk(lattice, t=2.7, k=k))
                    for k in k_arr])
"""

import argparse
import numpy as np
import scipy.sparse as sp



#  Load lattice data


def load_lattice(npz_path: str) -> dict:
    """
    Load the .npz produced by lattice_generator.py and return a dict with
    pre-computed quantities needed by the Hamiltonian builder.

    Keys in returned dict
    ---------------------
    M          : int    – number of atoms per unit cell
    T          : float  – unit-cell period |a1| in Å
    a1         : (2,)   – translation vector
    positions  : (M,2)  – Cartesian atom coordinates
    labels     : (M,)   – sublattice index (0=A, 1=B)
    neighbors  : list of dicts with keys i, j, cell, dx, dy
                 (cell = 0 for intra-cell, ±1 for inter-cell along a1)
    nnn        : list of dicts – next-nearest neighbours (same sublattice)
                 populated lazily by find_nnn(); empty until then.
    """
    data = np.load(npz_path, allow_pickle=True)

    a1   = data["unit_cell_vectors"][0]
    T    = float(np.linalg.norm(a1))
    M    = len(data["positions"])

    # Reconstruct neighbor list from structured array
    nb_arr = data["neighbors"]
    neighbors = [
        {"i": int(r["i"]), "j": int(r["j"]),
         "cell": int(r["cell"]),
         "dx": float(r["dx"]), "dy": float(r["dy"])}
        for r in nb_arr
    ]

    return dict(
        M         = M,
        T         = T,
        a1        = a1,
        positions = data["positions"],
        labels    = data["sublattice_labels"],
        neighbors = neighbors,
        nnn       = [],           # filled by find_nnn() on request
    )


#  Next-nearest-neighbour finder

def find_nnn(lattice: dict, a: float, tol: float = 0.05) -> None:
    """
    Populate lattice["nnn"] with next-nearest-neighbour pairs.

    NNN distance = a*sqrt(3).  Only same-sublattice pairs are included
    (bipartite honeycomb: NNN connects A-A or B-B).

    Modifies `lattice` in-place; safe to call multiple times.
    """
    positions = lattice["positions"]
    labels    = lattice["labels"]
    a1        = lattice["a1"]
    nnn_len   = a * np.sqrt(3)
    M         = lattice["M"]
    nnn       = []

    for cell_image in [0, 1, -1]:
        shift = cell_image * a1
        for i in range(M):
            for j in range(M):
                if cell_image == 0 and i >= j:
                    continue
                if labels[i] != labels[j]:
                    continue          # NNN connects same sublattice only
                disp = positions[j] + shift - positions[i]
                dist = np.linalg.norm(disp)
                if abs(dist - nnn_len) < tol * nnn_len:
                    nnn.append(dict(
                        i    = i,  j    = j,
                        cell = cell_image,
                        dx   = float(disp[0]), dy = float(disp[1]),
                    ))

    lattice["nnn"] = nnn



#  Core Hamiltonian builder


def build_Hk(
    lattice : dict,
    t       : float,
    k       : float,
    t2      : float = 0.0,
    delta   : float = 0.0,
) -> np.ndarray:
    """
    Assembles the M×M Bloch Hamiltonian H(k) as a dense complex128 array.

    Parameters
    ----------
    lattice : dict
        Output of load_lattice().
    t : float
        Nearest-neighbour hopping energy in eV (2.7 eV for graphene).
    k : float
        Bloch wavevector in Å⁻¹.  BZ spans [-π/T, π/T].
    t2 : float, optional
        Next-nearest-neighbour hopping in eV (t' ≈ 0.1*t, default 0).
        If non-zero and lattice["nnn"] is empty, find_nnn() is called
        automatically.  Requires the bond length a to be inferred from
        lattice geometry (first NN bond length used).
    delta : float, optional
        Staggered on-site potential in eV: +delta on sublattice A,
        -delta on sublattice B  Default 0.

    Returns
    -------
    H : ndarray, shape (M, M), dtype complex128
        Hermitian Hamiltonian matrix.  Use np.linalg.eigvalsh(H) for bands.
    """
    M         = lattice["M"]
    T         = lattice["T"]
    neighbors = lattice["neighbors"]
    labels    = lattice["labels"]

    H = np.zeros((M, M), dtype=np.complex128)

    # --- NN hopping (off-diagonal, A-B) ---
    # The neighbour list from lattice_generator has an asymmetric storage
    # convention that must be respected here:
    #
    #   cell == 0  (intra-cell): stored ONCE with i < j only.
    #              must add both H[i,j] and its conjugate H[j,i].
    #
    #   cell == ±1 (inter-cell): stored TWICE — once as (i, j, cell=+1)
    #              and once as (i, j, cell=-1) for the reverse direction.

    for nb in neighbors:
        i, j, n = nb["i"], nb["j"], nb["cell"]
        phase = np.exp(1j * k * n * T)
        H[i, j] += -t * phase
        if n == 0:
            H[j, i] += -t * np.conj(phase)  # intra-cell only: add H.c. explicitly

    # --- NNN hopping (diagonal blocks, A-A and B-B) ---
    # Same storage convention applies: intra-cell NNN pairs stored once (i<j),
    # inter-cell NNN pairs stored in both directions.
    if t2 != 0.0:
        if not lattice["nnn"]:
            # Infer bond length from first NN entry
            nb0  = neighbors[0]
            a_nn = np.hypot(nb0["dx"], nb0["dy"])
            find_nnn(lattice, a_nn)
        for nb in lattice["nnn"]:
            i, j, n = nb["i"], nb["j"], nb["cell"]
            phase = np.exp(1j * k * n * T)
            H[i, j] += -t2 * phase
            if n == 0:
                H[j, i] += -t2 * np.conj(phase)

    # --- Staggered on-site potential ---
    if delta != 0.0:
        for i in range(M):
            H[i, i] += delta * (1 if labels[i] == 0 else -1)

    return H


def build_Hk_sparse(
    lattice : dict,
    t       : float,
    k       : float,
    t2      : float = 0.0,
    delta   : float = 0.0,
) -> sp.csr_matrix:
    """
    Return H(k) as a scipy.sparse.csr_matrix.

    For NN-only GNRs the fill fraction is O(3/M), so sparsity becomes
    worthwhile for M > 200 (N > 100 zigzag chains).  For band-structure
    sweeps over many k, the dense version is usually faster due to
    per-call overhead.
    """
    return sp.csr_matrix(build_Hk(lattice, t, k, t2=t2, delta=delta))



#  k-path utility


def kpath(N_k: int, T: float, endpoint: bool = False) -> np.ndarray:
    """
    Return N_k evenly spaced k points spanning the 1-D Brillouin zone
    [-π/T, +π/T].

    Parameters
    ----------
    N_k      : number of k points
    T        : unit-cell period in Å
    endpoint : if True, include +π/T (useful for plotting; creates
               a duplicate if used in periodic calculations)
    """
    k_max = np.pi / T
    return np.linspace(-k_max, k_max, N_k, endpoint=endpoint)



#  Validation helpers


def check_hermitian(H: np.ndarray, tol: float = 1e-10) -> bool:
    """Return True if H is Hermitian to within tol"""
    return bool(np.allclose(H, H.conj().T, atol=tol))


def check_particle_hole(bands: np.ndarray, tol: float = 1e-8) -> bool:
    """
    Check particle-hole symmetry: eigenvalues come in +/- E pairs (valid when
    t2=0 and delta=0 on a bipartite lattice).
    """
    for row in bands:
        es = np.sort(row)
        if not np.allclose(es, -es[::-1], atol=tol):
            return False
    return True



#  CLI


def main():
    parser = argparse.ArgumentParser(
        description="Build H(k) for a GNR from a lattice_generator .npz file."
    )
    parser.add_argument("--npz",    required=True,
                        help="Path to .npz from lattice_generator.py")
    parser.add_argument("--t",      type=float, default=2.7,
                        help="NN hopping in eV (default: 2.7)")
    parser.add_argument("--k",      type=float, default=0.0,
                        help="Wavevector in Å⁻¹ (default: 0)")
    parser.add_argument("--t2",     type=float, default=0.0,
                        help="NNN hopping in eV (default: 0, i.e. NN only)")
    parser.add_argument("--delta",  type=float, default=0.0,
                        help="Staggered on-site potential in eV (default: 0)")
    parser.add_argument("--sparse", action="store_true",
                        help="Print sparsity info and use sparse format")
    parser.add_argument("--out",    default=None,
                        help="Stem for output .npz  (saves H_real, H_imag, "
                             "eigenvalues)")
    args = parser.parse_args()

    lattice = load_lattice(args.npz)
    T       = lattice["T"]
    M       = lattice["M"]

    print(f"\n{'='*54}")
    print(f"  Loaded  : {args.npz}")
    print(f"  M       : {M} atoms/cell   T = {T:.5f} Å")
    print(f"  BZ edge : ±π/T = ±{np.pi/T:.5f} Å⁻¹")
    print(f"  k       : {args.k:.5f} Å⁻¹   (k·T/π = {args.k*T/np.pi:.4f})")
    print(f"  t       : {args.t} eV   t2 = {args.t2} eV   Δ = {args.delta} eV")

    H = build_Hk(lattice, t=args.t, k=args.k, t2=args.t2, delta=args.delta)

    # --- Validation ---
    herm_ok = check_hermitian(H)
    print(f"\n  Hermitian check : {'PASS ✓' if herm_ok else 'FAIL ✗'}")
    if not herm_ok:
        dev = np.max(np.abs(H - H.conj().T))
        print(f"    max |H - H†| = {dev:.2e}")

    eigenvalues = np.linalg.eigvalsh(H)   # guaranteed real for Hermitian H
    print(f"\n  Eigenvalues (eV):")
    for i, e in enumerate(eigenvalues):
        print(f"    band {i:>3d} :  {e:+.6f}")

    if args.t2 == 0.0 and args.delta == 0.0:
        ph_ok = check_particle_hole(eigenvalues[None], tol=1e-6)
        print(f"\n  Particle-hole symmetry : {'PASS ✓' if ph_ok else 'FAIL ✗'}")

    if args.sparse:
        Hs = build_Hk_sparse(lattice, t=args.t, k=args.k,
                              t2=args.t2, delta=args.delta)
        nnz  = Hs.nnz
        fill = nnz / M**2 * 100
        print(f"\n  Sparse info : nnz = {nnz}  ({fill:.1f}% fill)")

    if args.out:
        np.savez(
            args.out,
            H_real      = H.real,
            H_imag      = H.imag,
            eigenvalues = eigenvalues,
            k           = np.array([args.k]),
            t           = np.array([args.t]),
        )
        print(f"\n  Saved → {args.out}.npz")

    print()


if __name__ == "__main__":
    main()
