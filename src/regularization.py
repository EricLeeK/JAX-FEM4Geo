"""
Regularization functions for heterogeneous parameter field inversion.

I1: Total Variation (TV) — piecewise constant fields, sharp interfaces
I2: Laplacian smoothness — smooth fields, suppresses oscillations
"""

import jax.numpy as np
import numpy as onp


# ---------------------------------------------------------------------------
# Neighbor pair construction for structured QUAD4 meshes
# ---------------------------------------------------------------------------

def build_structured_neighbor_pairs(Nx, Ny):
    """Build neighbor pair indices for a structured Nx×Ny QUAD4 mesh.

    Cell ordering: cell_idx = ix * Ny + iy (matches rectangle_mesh).
    Returns pairs for edges sharing in x-direction and y-direction.

    Returns
    -------
    neighbor_pairs : (num_pairs, 2) int array
    """
    pairs = []
    for ix in range(Nx):
        for iy in range(Ny):
            idx = ix * Ny + iy
            # Right neighbor (x-direction)
            if ix < Nx - 1:
                pairs.append([idx, (ix + 1) * Ny + iy])
            # Top neighbor (y-direction)
            if iy < Ny - 1:
                pairs.append([idx, ix * Ny + (iy + 1)])
    return onp.array(pairs, dtype=onp.int32)


def build_laplacian_matrix(Nx, Ny):
    """Build graph Laplacian matrix for a structured Nx×Ny QUAD4 mesh.

    L_ii = degree(i), L_ij = -1 if (i,j) are neighbors.

    Returns
    -------
    L : (num_cells, num_cells) dense array
    """
    nc = Nx * Ny
    L = onp.zeros((nc, nc))
    for ix in range(Nx):
        for iy in range(Ny):
            idx = ix * Ny + iy
            neighbors = []
            if ix > 0:
                neighbors.append((ix - 1) * Ny + iy)
            if ix < Nx - 1:
                neighbors.append((ix + 1) * Ny + iy)
            if iy > 0:
                neighbors.append(ix * Ny + (iy - 1))
            if iy < Ny - 1:
                neighbors.append(ix * Ny + (iy + 1))
            L[idx, idx] = len(neighbors)
            for j in neighbors:
                L[idx, j] = -1.0
    return L


# ---------------------------------------------------------------------------
# I1: Total Variation regularization
# ---------------------------------------------------------------------------

def tv_regularizer(E_field, neighbor_pairs, eps_tv=1e-6, E_ref=1.0):
    """Smooth Total Variation regularization (differentiable approximation).

    R_TV(E) = (1/n_pairs) Σ_{(i,j)} sqrt(((E_i - E_j)/E_ref)^2 + eps_tv)

    Normalized by number of pairs and a reference E scale so that the
    regularization weight λ is interpretable across different meshes.

    Parameters
    ----------
    E_field : (num_cells,) parameter field
    neighbor_pairs : (num_pairs, 2) int array
    eps_tv : float, smoothing parameter for differentiability
    E_ref : float, reference E scale for normalization

    Returns
    -------
    scalar TV value
    """
    n_pairs = neighbor_pairs.shape[0]
    diffs = (E_field[neighbor_pairs[:, 0]] - E_field[neighbor_pairs[:, 1]]) / E_ref
    return np.sum(np.sqrt(diffs ** 2 + eps_tv)) / n_pairs


# ---------------------------------------------------------------------------
# I2: Laplacian smoothness regularization
# ---------------------------------------------------------------------------

def laplacian_regularizer(E_field, laplacian_matrix, E_ref=1.0):
    """Laplacian smoothness regularization.

    R_Lap(E) = (1/n) ||L · (E/E_ref)||^2

    Normalized by number of cells and a reference E scale.

    Parameters
    ----------
    E_field : (num_cells,) parameter field
    laplacian_matrix : (num_cells, num_cells) Laplacian matrix
    E_ref : float, reference E scale for normalization

    Returns
    -------
    scalar smoothness value
    """
    n = E_field.shape[0]
    Lx = laplacian_matrix @ (E_field / E_ref)
    return np.sum(Lx ** 2) / n
