"""P2-7: Single-point driver for the DP + isotropic hardening model.

Exercises the hardening constitutive law directly (no FEM solve) to verify it
behaves physically: under increasing axial compression, a positive hardening
modulus H should raise the post-yield stress (stiffening response) relative to
ideal plasticity (H=0), and a negative H should lower it (softening).

This bypasses the FEM solver to isolate the constitutive behaviour — the
correct place to validate the return-mapping math. The single-step limitation
documented in differentiable_dp_hardening.py applies: H's effect is muted by
the elastic return-mapping modulus within one step.

Run:
    python examples/run_hardening_driver.py
"""

import os
import sys
import numpy as onp

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.differentiable_dp_hardening import DifferentiableDPHardening


def stress_at(epsilon, E, k0, H, nu=0.3, alpha=0.3):
    """Evaluate the hardening DP stress for a full strain tensor (single step).

    Calls the model's tensor_map directly with zero history (single-step).
    Returns the stress tensor as a numpy array.
    """
    import jax
    import jax.numpy as np

    # Build a throwaway 1-cell "problem" just to access get_tensor_map(); the
    # tensor map is a pure function of (u_grad, sigma_old, eps_old) + params.
    class _Probe(DifferentiableDPHardening):
        def custom_init(self):
            self.fe = self.fes[0]
            self.epsilons_old = np.zeros((1, self.fe.num_quads, self.fe.vec, self.dim))
            self.sigmas_old = np.zeros_like(self.epsilons_old)
            self.internal_vars = [self.sigmas_old, self.epsilons_old]

    from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh
    rd = os.path.join(project_root, 'results', '_probe')
    os.makedirs(rd, exist_ok=True)
    mm = box_mesh_gmsh(Nx=1, Ny=1, Nz=1, domain_x=1., domain_y=1., domain_z=1.,
                       data_dir=rd, ele_type='HEX8')
    mesh = Mesh(mm.points, mm.cells_dict[get_meshio_cell_type('HEX8')])

    def bottom(p): return np.isclose(p[2], 0., atol=1e-5)
    def top(p): return np.isclose(p[2], 1., atol=1e-5)
    probe = _Probe(mesh, vec=3, dim=3,
                   dirichlet_bc_info=[[bottom, top], [2, 2],
                                      [lambda p: 0., lambda p: 0.]])
    probe.set_params(np.array([E, k0, H]))
    stress_fn = probe.get_tensor_map()
    u_grad = np.array(epsilon)
    sigma_old = np.zeros((3, 3))
    eps_old = np.zeros((3, 3))
    return onp.array(stress_fn(u_grad, sigma_old, eps_old))


def main():
    E, k0, nu, alpha = 70.0e3, 250.0, 0.3, 0.3
    # Uniaxial-strain loading: eps_zz increases, eps_xx=eps_yy=0 (oedometric).
    print("P2-7: Hardening constitutive driver (single-point)")
    print("=" * 60)
    print(f"E={E}, k0={k0}, nu={nu}, alpha={alpha}")
    print()
    print(f"{'eps_zz':>10} {'H=0':>12} {'H=+5e4':>12} {'H=-5e4':>12}")
    print("-" * 50)

    for eps_zz in [-0.001, -0.005, -0.01, -0.02, -0.03, -0.05]:
        szz = {}
        for H, label in [(0., 'H=0'), (5e4, 'H=+5e4'), (-5e4, 'H=-5e4')]:
            eps = onp.diag([0., 0., eps_zz]).astype(float)
            sig = stress_at(eps.tolist(), E, k0, H, nu, alpha)
            szz[label] = sig[2, 2]
        print(f"{eps_zz*100:>9.2f}% {szz['H=0']:>12.1f} "
              f"{szz['H=+5e4']:>12.1f} {szz['H=-5e4']:>12.1f}")

    print()
    print("Expected: post-yield, H>0 gives higher |sigma_zz| (stiffening),")
    print("          H<0 gives lower |sigma_zz| (softening), vs H=0 (ideal).")


if __name__ == "__main__":
    main()
