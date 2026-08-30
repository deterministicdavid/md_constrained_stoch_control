import pytest
import numpy as np

# Adjust imports to match your new module structure
from sde_md.envs.sde_env import step_sde

# Safely import the C++ extension
try:
    import sde_cpp
except ImportError:
    try:
        # Fallback in case it's built explicitly inside the envs module
        from sde_md.envs import sde_cpp
    except ImportError:
        pytest.fail("Could not import sde_cpp. Make sure the C++ extension is built.")


def test_cpp_python_equivalence():
    """
    Validates that the C++ SDE step implementation perfectly matches
    the Python reference implementation for 100 random transitions.
    """
    # Set a fixed seed so the test is fully deterministic and reproducible
    np.random.seed(42)

    n_tests = 100
    state_dim = 2
    action_dim = 2
    dt = 0.01
    radius = 2.0
    T = 10.0
    sigma = 0.1
    space_cost_factor_terminal = 3.5
    control_cost_factor = 0.033

    for i in range(n_tests):
        # 1. Random Inputs
        state = np.random.uniform(-2, 2, size=(state_dim,)).astype(np.float32)
        action = np.random.uniform(-1, 1, size=(action_dim,)).astype(np.float32)
        dW = np.random.normal(0, 1, size=(state_dim,)).astype(np.float32)
        current_time = float(np.random.uniform(0, T))

        # 2. Run Python Reference
        py_state, py_reward, py_time, py_term, py_trunc = step_sde(
            state,
            action,
            dW,
            dt,
            sigma,
            radius,
            current_time,
            T,
            control_cost_factor,
            space_cost_factor_terminal,
        )

        # 3. Run C++ Implementation
        cpp_state, cpp_reward, cpp_time, cpp_term, cpp_trunc = sde_cpp.step_sde(
            state,
            action,
            dW,
            dt,
            sigma,
            radius,
            current_time,
            T,
            control_cost_factor,
            space_cost_factor_terminal,
        )

        # 4. Compare
        # Tolerances: float32 precision is roughly 1e-7
        rtol = 1e-3
        atol = 1e-3

        np.testing.assert_allclose(
            py_state,
            cpp_state,
            rtol=rtol,
            atol=atol,
            err_msg=f"State Mismatch on test {i}",
        )
        np.testing.assert_allclose(
            py_reward,
            cpp_reward,
            rtol=rtol,
            atol=atol,
            err_msg=f"Reward Mismatch on test {i}",
        )
        np.testing.assert_allclose(
            py_time,
            cpp_time,
            rtol=rtol,
            atol=atol,
            err_msg=f"Time Mismatch on test {i}",
        )
        assert (
            py_term == cpp_term
        ), f"Terminated Mismatch on test {i}: Py={py_term}, C++={cpp_term}"
        assert (
            py_trunc == cpp_trunc
        ), f"Truncated Mismatch on test {i}: Py={py_trunc}, C++={cpp_trunc}"
