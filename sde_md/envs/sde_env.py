import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numba import njit
from typing import Optional, Union

# Import the compiled C++ module
try:
    from . import sde_cpp
except ImportError:
    try:
        import sde_cpp
    except ImportError:
        raise ImportError(
            "Please compile the C++ extension first using `python setup.py build_ext --inplace`"
        )


# Define the dynamics and cost functions
@njit
def drift_fn(x, a):
    # Example: Attempt to steer towards origin against a current
    return 0.0 * x + a


@njit
def diffusion_fn(x, a, sigma):
    # Constant noise intensity
    return (sigma * np.eye(2)).astype(np.float32)


@njit
def running_cost_fn(x, a, control_cost_factor):
    # Quadratic cost on state and control
    # return np.sum(x**2) + 0.01*np.sum(a**2)
    return control_cost_factor * np.sum(a**2)


@njit
def terminal_cost_fn(x, space_cost_factor_terminal):
    return space_cost_factor_terminal * np.sum(x**2)


# @njit
def step_sde(
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
):
    # 1. Calculate Dynamics (Euler-Maruyama)
    # dX = b(x,u)dt + sigma(x,u)dW

    # Brownian increment dW ~ N(0, dt)
    # Standard deviation is sqrt(dt)

    drift = drift_fn(state, action)
    diffusion = diffusion_fn(state, action, sigma)

    # diffusion_term = diffusion @ dW
    diffusion_term = np.dot(diffusion, dW)

    # Update State
    # X_{t+h} = X_t + b * h + sigma * dW
    next_state = state + drift * dt + diffusion_term * np.sqrt(dt)

    # Update Time
    current_time += dt

    # 2. Check Constraints / Termination
    dist_sq = np.sum(next_state**2)
    is_outside = dist_sq > (radius**2)
    time_limit_reached = current_time >= T

    terminated = False
    truncated = False
    reward = 0.0

    if is_outside or time_limit_reached:
        # Process left the ball
        terminated = True
        # Apply terminal cost g (negative reward)
        reward = -terminal_cost_fn(state, space_cost_factor_terminal)
    else:
        # Normal step inside the ball
        step_cost = running_cost_fn(state, action, control_cost_factor)
        # Reward is negative cost integrated over time step h
        reward = -step_cost * dt

    return next_state, reward, current_time, terminated, truncated


def validate_cpp_implementation(state_dim, action_dim, dt, radius, T, n_tests=100):
    print(f"Running {n_tests} validation tests (C++ vs Python)... ", end="")

    sigma = 0.1
    space_cost_factor_terminal = 3.5
    control_cost_factor = 0.033

    for i in range(n_tests):
        # 1. Random Inputs
        state = np.random.uniform(-2, 2, size=(state_dim,)).astype(np.float32)

        # Generate continuous action for testing physics directly
        # (bypassing the discrete grid lookup for this test)
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
            control_cost_factor=control_cost_factor,
            space_cost_factor_terminal=space_cost_factor_terminal,
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

        try:
            np.testing.assert_allclose(
                py_state, cpp_state, rtol=rtol, atol=atol, err_msg="State Mismatch"
            )
            np.testing.assert_allclose(
                py_reward, cpp_reward, rtol=rtol, atol=atol, err_msg="Reward Mismatch"
            )

            np.testing.assert_allclose(
                py_time, cpp_time, rtol=rtol, atol=atol, err_msg="Time Mismatch"
            )
            assert (
                py_term == cpp_term
            ), f"Terminated Mismatch: Py={py_term}, C++={cpp_term}"
            assert (
                py_trunc == cpp_trunc
            ), f"Truncated Mismatch: Py={py_trunc}, C++={cpp_trunc}"
        except AssertionError as e:
            print(f"\nFAILED! sample {i}")
            print("Inputs:")
            print(f"  State: {state}")
            print(f"  Action: {action}")
            print(f"  dW: {dW}")
            print(f"  Time: {current_time}")
            print("\nOutputs (Python vs C++):")
            print(f"  Next State: {py_state} vs {cpp_state}")
            print(f"  Reward: {py_reward} vs {cpp_reward}")
            print(f"  Terminated: {py_term} vs {cpp_term}")
            raise e

    print("PASSED.")


class SDEDiffusionEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        env_type: str = "continuous",  # purely for config sanity checking
        radius: float = 2.0,
        sigma: float = 0.1,
        T: float = 10.0,
        dt: float = 0.01,
        state_dim: int = 2,
        action_dim: int = 2,
        action_bound: float = 1.0,
        control_cost_factor: float = 0.1,
        space_cost_factor_terminal: float = 2.0,
        run_validation: bool = False,
    ):
        """
        Args:
            drift_fn: Function b(x, a) returning numpy array of shape (state_dim,)
            diffusion_fn: Function sigma(x, a) returning numpy array of shape (state_dim, state_dim) or (state_dim,)
            cost_running_fn: Function c(x, a) returning scalar cost
            cost_terminal: Scalar cost g for exiting the ball
            radius: Radius R of the ball
            T: Terminal time
            dt: Step size h
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_bound: Box bound for actions (assumes symmetric -bound to +bound)
        """
        if env_type != "continuous":
            raise ValueError(f"SDEDiffusionEnv must have env_type=continuous")

        if run_validation:
            validate_cpp_implementation(
                state_dim=state_dim, action_dim=action_dim, dt=dt, radius=radius, T=T
            )

        super().__init__()

        # Physics / Dynamics parameters
        self.drift_fn = drift_fn
        self.diffusion_fn = diffusion_fn
        self.dt = dt
        self.T = T
        self.radius = radius
        self.sigma = sigma

        # Dimensions
        self.spatial_dim = state_dim
        self.action_dim = action_dim

        # Spaces
        # Observation is the state X_t + current time t (often useful for time-dependent policies)
        # Here we just return X_t, but you can append t if needed.
        # We set bounds to infinity as SDEs are unbounded, though we terminate at R.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.spatial_dim + 1,), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-action_bound, high=action_bound, shape=(action_dim,), dtype=np.float32
        )

        self.control_cost_factor = control_cost_factor
        self.space_cost_factor_terminal = space_cost_factor_terminal

        # State initialization
        self.state = None
        self.current_time = 0.0

    # def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
    #     super().reset(seed=seed)
    #     x0_spatial = self.np_random.uniform(
    #         -self.radius, self.radius, size=(self.spatial_dim,)
    #     ).astype(np.float32)
    #     self.state = np.array(x0_spatial, dtype=np.float32)
    #     # self.current_time = 0.0
    #     self.current_time = self.np_random.uniform(0, self.T - self.dt)
    #     obs = np.concatenate([self.state, [self.current_time]]).astype(np.float32)

    #     return obs, {}
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            super().reset(seed=seed)
        # x0_spatial = self.np_random.uniform(-self.radius, self.radius, size=(self.spatial_dim,)).astype(np.float32)

        if options is not None and "x0" in options:
            x0_spatial = np.array(options["x0"], dtype=np.float32)
            if x0_spatial.shape != (self.spatial_dim,):
                raise ValueError(
                    f"Custom x0 has shape {x0_spatial.shape}, expected {(self.spatial_dim,)}"
                )
        else:
            x0_spatial = self.np_random.uniform(
                -self.radius, self.radius, size=(self.spatial_dim,)
            ).astype(np.float32)

        if options is not None and "t0" in options:
            self.current_time = float(options["t0"])
        else:
            self.current_time = self.np_random.uniform(0, self.T - self.dt)

        self.state = np.array(x0_spatial, dtype=np.float32)
        obs = np.concatenate([self.state, [self.current_time]]).astype(np.float32)
        return obs, {}

    def step(self, action):
        dW = self.np_random.normal(loc=0.0, scale=1.0, size=self.spatial_dim).astype(
            np.float32
        )

        # next_state, reward, current_time, terminated, truncated = step_sde(
        #     self.state,
        #     action,
        #     dW,
        #     self.dt,
        #     self.sigma,
        #     self.radius,
        #     self.current_time,
        #     self.T,
        # )

        next_state, reward, current_time, terminated, truncated = sde_cpp.step_sde(
            self.state,
            action,
            dW,
            self.dt,
            self.sigma,
            self.radius,
            self.current_time,
            self.T,
            self.control_cost_factor,
            self.space_cost_factor_terminal,
        )

        self.state = next_state.astype(np.float32)
        self.current_time = current_time

        obs = np.concatenate([self.state, [self.current_time]]).astype(np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        # Optional: simple print for debugging
        print(f"t={self.current_time:.2f}, x={self.state}")


class SDEDiffusionEnvDiscAct(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        env_type: str = "discrete",
        radius: float = 2.0,
        sigma: float = 0.1,
        T: float = 10.0,
        dt: float = 0.01,
        state_dim: int = 2,
        action_dim: int = 2,
        action_min: Union[float, list] = -1.0,
        action_max: Union[float, list] = 1.0,
        n_actions_per_dim: int = 3,  # e.g., 3 creates {-1, 0, 1}
        control_cost_factor: float = 0.1,
        space_cost_factor_terminal: float = 2.0,
        run_validation: bool = False,
    ):
        if env_type != "discrete":
            raise ValueError(f"SDEDiffusionEnvDiscAct must have env_type=discrete")

        if run_validation:
            validate_cpp_implementation(
                state_dim=state_dim, action_dim=action_dim, dt=dt, radius=radius, T=T
            )

        super().__init__()

        # Physics Parameters
        self.dt = dt
        self.T = T
        self.radius = radius
        self.sigma = sigma
        self.spatial_dim = state_dim
        self.action_dim = action_dim

        # --- 1. Setup Discrete Action Grid ---
        # Handle scalar inputs by repeating them for each dimension
        if np.isscalar(action_min):
            action_min = [action_min] * action_dim
        if np.isscalar(action_max):
            action_max = [action_max] * action_dim

        # Validate
        assert len(action_min) == action_dim
        assert len(action_max) == action_dim

        # Create the mapping: Dimension -> Index -> Float Value
        # self.action_grids will be a list of arrays, e.g., [array([-1, 0, 1]), array([-1, 0, 1])]
        self.action_grids = []
        for i in range(action_dim):
            grid = np.linspace(
                action_min[i], action_max[i], n_actions_per_dim, dtype=np.float32
            )
            self.action_grids.append(grid)

        # Store as array for fast lookup if all dims are same size (they are here)
        self.action_grids = np.array(self.action_grids)

        # Observation Space (Continuous State)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.spatial_dim + 1,), dtype=np.float32
        )

        # Action Space (Discrete Indices)
        # MultiDiscrete allows independent discrete choices for each dimension.
        # e.g., [3, 3] means action is a vector [a, b] where a,b in {0,1,2}
        self.action_space = spaces.MultiDiscrete([n_actions_per_dim] * action_dim)

        self.control_cost_factor = control_cost_factor
        self.space_cost_factor_terminal = space_cost_factor_terminal

        # State initialization
        self.state = None
        self.current_time = 0.0

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            super().reset(seed=seed)
        # x0_spatial = self.np_random.uniform(-self.radius, self.radius, size=(self.spatial_dim,)).astype(np.float32)

        if options is not None and "x0" in options:
            x0_spatial = np.array(options["x0"], dtype=np.float32)
            if x0_spatial.shape != (self.spatial_dim,):
                raise ValueError(
                    f"Custom x0 has shape {x0_spatial.shape}, expected {(self.spatial_dim,)}"
                )
        else:
            x0_spatial = self.np_random.uniform(
                -self.radius, self.radius, size=(self.spatial_dim,)
            ).astype(np.float32)

        if options is not None and "t0" in options:
            self.current_time = float(options["t0"])
        else:
            self.current_time = self.np_random.uniform(0, self.T - self.dt)

        self.state = np.array(x0_spatial, dtype=np.float32)
        obs = np.concatenate([self.state, [self.current_time]]).astype(np.float32)
        return obs, {}

    def step(self, action_indices):
        """
        action_indices: A numpy array of integers, e.g., [0, 2]
        """
        # --- 2. Map Indices to Continuous Values ---
        # We need to convert the integer indices into physical float values.
        # action_indices is shape (2,), self.action_grids is (2, 3)
        # We want to pick the specific float for each dimension.

        # Efficient NumPy indexing to lookup values
        # row_indices = [0, 1], col_indices = [action_indices[0], action_indices[1]]
        continuous_action = self.action_grids[
            np.arange(self.action_dim), action_indices
        ]

        # Ensure it is float32 for Numba
        continuous_action = continuous_action.astype(np.float32)

        dW = self.np_random.normal(loc=0.0, scale=1.0, size=self.spatial_dim).astype(
            np.float32
        )

        # next_state, reward, current_time, terminated, truncated = step_sde(self.state,
        #                                                                    continuous_action,
        #                                                                    dW,
        #                                                                    self.dt,
        #                                                                    self.radius,
        #                                                                    self.current_time,
        #                                                                    self.T)

        next_state, reward, current_time, terminated, truncated = sde_cpp.step_sde(
            self.state,
            continuous_action,
            dW,
            self.dt,
            self.sigma,
            self.radius,
            self.current_time,
            self.T,
            self.control_cost_factor,
            self.space_cost_factor_terminal,
        )

        self.state = next_state.astype(np.float32)
        self.current_time = current_time

        obs = np.concatenate([self.state, [self.current_time]]).astype(np.float32)

        return obs, reward, terminated, truncated, {}

    def render(self):
        # Optional: simple print for debugging
        print(f"t={self.current_time:.2f}, x={self.state}")


if __name__ == "__main__":
    env = SDEDiffusionEnvDiscAct(radius=2.0, run_validation=True)
