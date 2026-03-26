import jax
import jax.numpy as np
import os
import sys

# Path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
jax_fem_path = os.path.join(project_root, 'jax-fem-main')
if jax_fem_path not in sys.path:
    sys.path.append(jax_fem_path)

from jax_fem.problem import Problem
from jax_fem.solver import ad_wrapper
from jax_fem.generate_mesh import box_mesh_gmsh, get_meshio_cell_type, Mesh


class DiffElasticity(Problem):
    def custom_init(self):
        self.fe = self.fes[0]
        self.E_field = 70.0e3 * np.ones((len(self.fe.cells), self.fe.num_quads))
        self.internal_vars = [self.E_field]

    def set_params(self, params):
        self.E_field = params[0] * np.ones((len(self.fe.cells), self.fe.num_quads))
        self.internal_vars = [self.E_field]

    def get_tensor_map(self):
        nu = 0.3

        def stress_fn(u_grad, E_local):
            epsilon = 0.5 * (u_grad + u_grad.T)
            mu = E_local / (2. * (1. + nu))
            lmbda = E_local * nu / ((1. + nu) * (1. - 2. * nu))
            return lmbda * np.trace(epsilon) * np.eye(self.dim) + 2. * mu * epsilon

        return stress_fn


def volume_avg_sigma_zz(problem, sol):
    fe = problem.fe
    u_grads = fe.sol_to_grad(sol)
    stress_fn = problem.get_tensor_map()
    sig = jax.vmap(jax.vmap(stress_fn))(u_grads, problem.E_field)
    JxW = fe.JxW
    return np.sum(sig[..., 2, 2] * JxW) / np.sum(JxW)


def run_elastic_grad_test(displacement):
    Lx, Ly, Lz = 10., 10., 10.
    Nx, Ny, Nz = 2, 2, 2

    data_dir = os.path.join(project_root, 'results', 'test_output_elastic')
    os.makedirs(data_dir, exist_ok=True)

    ele_type = 'HEX8'
    cell_type = get_meshio_cell_type(ele_type)
    meshio_mesh = box_mesh_gmsh(
        Nx=Nx, Ny=Ny, Nz=Nz, domain_x=Lx, domain_y=Ly, domain_z=Lz,
        data_dir=data_dir, ele_type=ele_type,
    )
    mesh = Mesh(meshio_mesh.points, meshio_mesh.cells_dict[cell_type])

    def bottom(p):
        return np.isclose(p[2], 0.)

    def top(p):
        return np.isclose(p[2], Lz)

    def corner(p):
        return np.logical_and(
            np.logical_and(np.isclose(p[0], 0.), np.isclose(p[1], 0.)),
            np.isclose(p[2], 0.),
        )

    location_fns = [bottom, top, corner, corner]
    vecs = [2, 2, 0, 1]
    value_fns = [lambda p: 0., lambda p: displacement, lambda p: 0., lambda p: 0.]
    dirichlet_bc_info = [location_fns, vecs, value_fns]

    problem = DiffElasticity(mesh, vec=3, dim=3, dirichlet_bc_info=dirichlet_bc_info)

    solver_options = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
    fwd_pred = ad_wrapper(problem, solver_options=solver_options, adjoint_solver_options=solver_options)

    def loss_fn(params):
        sol_list = fwd_pred(params)
        problem.set_params(params)
        return volume_avg_sigma_zz(problem, sol_list[0])

    print(f"\nElasticity AD vs FD (loss = volume avg sigma_zz, displacement = {displacement})")

    params_init = np.array([70.0e3])
    loss_val, grad = jax.value_and_grad(loss_fn)(params_init)
    grad_E = grad[0]

    eps = 100.0
    loss_plus = loss_fn(params_init + np.array([eps]))
    loss_minus = loss_fn(params_init - np.array([eps]))
    grad_fd = (loss_plus - loss_minus) / (2. * eps)

    abs_err = np.abs(grad_E - grad_fd)
    rel_err = abs_err / (np.abs(grad_fd) + 1e-10)

    print(f"  Loss: {loss_val:.6e}")
    print(f"  dLoss/dE (AD): {grad_E:.6e}")
    print(f"  dLoss/dE (FD): {grad_fd:.6e}")
    print(f"  Relative error: {rel_err:.2e}")

    assert rel_err < 1e-3 or abs_err < 1e-3, f"Elastic gradient mismatch: rel={rel_err}, abs={abs_err}"


def test_elastic_gradient():
    for disp in [-0.01, -0.05, -0.1]:
        run_elastic_grad_test(disp)


if __name__ == "__main__":
    test_elastic_gradient()
