"""Tunnel cross-section mesh generator for 2D plane strain FEM analysis.

Generates a rectangular domain with a circular tunnel opening using gmsh.
Compatible with JAX-FEM's Mesh class.
"""

import os
import numpy as onp
import meshio
import gmsh


def tunnel_mesh_gmsh(data_dir, W=40.0, H=40.0, R=3.0,
                     cx=None, cy=None,
                     mesh_size_far=2.0, mesh_size_near=0.5,
                     ele_type='QUAD4'):
    """Generate a 2D mesh of rectangular domain with circular tunnel.

    Parameters
    ----------
    data_dir : str
        Directory for mesh cache.
    W : float
        Domain width [m].
    H : float
        Domain height [m].
    R : float
        Tunnel radius [m].
    cx : float or None
        Tunnel center x-coordinate (default: W/2).
    cy : float or None
        Tunnel center y-coordinate (default: H/2).
    mesh_size_far : float
        Mesh size far from tunnel.
    mesh_size_near : float
        Mesh size near tunnel surface.
    ele_type : str
        'QUAD4' or 'TRI3'.

    Returns
    -------
    mesh : meshio.Mesh
        Mesh with 'quad' or 'triangle' cells.
    """
    if cx is None:
        cx = W / 2.0
    if cy is None:
        cy = H / 2.0

    assert R > 0, f"Tunnel radius must be positive, got {R}"
    assert 0 < cx - R and cx + R < W, "Tunnel must be inside domain (x-direction)"
    assert 0 < cy - R and cy + R < H, "Tunnel must be inside domain (y-direction)"

    msh_dir = os.path.join(data_dir, 'msh')
    os.makedirs(msh_dir, exist_ok=True)
    base_name = (f'tunnel_W{W}_H{H}_R{R}_cx{cx}_cy{cy}'
                 f'_far{mesh_size_far}_near{mesh_size_near}_{ele_type}')
    msh_file = os.path.join(msh_dir, f'{base_name}.msh')

    if os.path.isfile(msh_file):
        print(f"Reusing cached mesh file: {msh_file}")
    else:
        _generate_tunnel_msh(msh_file, W, H, R, cx, cy,
                             mesh_size_far, mesh_size_near, ele_type)
        print(f"Generated mesh file: {msh_file}")

    mesh = meshio.read(msh_file)
    return mesh


def _generate_tunnel_msh(msh_file, W, H, R, cx, cy,
                         mesh_size_far, mesh_size_near, ele_type):
    """Use gmsh Python API to create the tunnel mesh and write to .msh file."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)
    gmsh.model.add("tunnel")

    # Rectangle (domain boundary)
    rect = gmsh.model.occ.addRectangle(0, 0, 0, W, H)

    # Circle (tunnel opening)
    circle = gmsh.model.occ.addDisk(cx, cy, 0, R, R)

    # Boolean cut: rectangle minus circle
    result, _ = gmsh.model.occ.cut([(2, rect)], [(2, circle)])
    gmsh.model.occ.synchronize()

    # Mesh size field: refine near tunnel
    gmsh.model.mesh.field.add("Distance", 1)
    # Get all curves — find the tunnel boundary (circle arcs)
    curves = gmsh.model.getEntities(1)
    curve_tags = [c[1] for c in curves]
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", curve_tags)
    gmsh.model.mesh.field.setNumber(1, "Sampling", 100)

    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", mesh_size_near)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", mesh_size_far)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", min(W, H) / 4.0)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    if ele_type == 'QUAD4':
        gmsh.option.setNumber("Mesh.Algorithm", 8)  # Frontal-Delaunay for quads
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # simple full-quad
        gmsh.option.setNumber("Mesh.RecombineAll", 1)

    gmsh.model.mesh.generate(2)
    gmsh.write(msh_file)
    gmsh.finalize()
