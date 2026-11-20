import types
import numpy as np
from sde_md.iterators.dpp_based_cont_act_iterations import ContinuousActionMirrorDescent
from sde_md.iterators.dpp_based_cont_act_pia import ContinuousActionPolicyIteration

def test_md_fixed_point_is_policy_improvement_action():
    s = types.SimpleNamespace(action_radius=10.0, cost_factor=0.01, ent_coef=0.0)
    g = np.array([[0.1, -0.05], [0.02, 0.0], [1.0, 2.0]])  # interior, interior, boundary optimum
    gx, gy = g[:, 0], g[:, 1]
    u_star = ContinuousActionPolicyIteration._update_primal_space_params(s, np.zeros((3, 2)), gx, gx, gy, gy)
    Z = np.zeros((3, 2))
    for _ in range(5000):
        Z = ContinuousActionMirrorDescent._update_dual_space_params(s, Z, gx, gx, gy, gy, md_lambda=0.5)
    u = 100.0 * Z / (1 + np.sqrt(1 + 100.0 * np.sum(Z**2, axis=1, keepdims=True)))
    np.testing.assert_allclose(u, u_star, atol=1e-3)