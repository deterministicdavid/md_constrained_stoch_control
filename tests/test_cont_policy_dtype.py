"""
Checks for the float64 option of the continuous-action tabular policies.

- QuantizedTabularContActionPolicy and QuantizedTabularZPolicy default to float32
  (as used by the existing tests) and accept param_dtype=torch.float64;
- setup_env_and_policy gives float64 tables for cont_act_pia, cont_act_md and
  cont_act_proj_pg, and still builds the projected-LQR baseline.

Run from anywhere:  pytest tests/test_cont_policy_dtype.py -v
"""
import pathlib
import sys

import pytest
import torch
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sde_md.envs.sde_env import SDEDiffusionEnv  # noqa: E402
from sde_md.policies.quantizedtabularcontpolicy import QuantizedTabularContActionPolicy  # noqa: E402
from sde_md.policies.quantizedtabularzpolicy import QuantizedTabularZPolicy  # noqa: E402

ENV_CONFIG = ROOT / "configs" / "env_config_cont_act.yaml"


@pytest.fixture(scope="module")
def env_and_cfg():
    with open(ENV_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg.pop("env_type", None)
    return SDEDiffusionEnv(**cfg), cfg


def assert_tables(policy, dtype):
    for name, p in policy.tabular_params.items():
        assert p.dtype == dtype, f"{name}: {p.dtype} != {dtype}"
        assert torch.isfinite(p).all(), f"{name} has non-finite entries"


@pytest.mark.parametrize("cls", [QuantizedTabularContActionPolicy, QuantizedTabularZPolicy])
@pytest.mark.parametrize("dtype", [None, torch.float64])
def test_policy_dtype(env_and_cfg, cls, dtype):
    env, cfg = env_and_cfg
    extra = {} if dtype is None else {"param_dtype": dtype}
    policy = cls(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.0,
        grid_bins=10,
        radius=cfg["radius"],
        action_radius=10.0,
        dt=0.05,
        T=cfg["T"],
        **extra,
    )
    assert_tables(policy, torch.float32 if dtype is None else dtype)
    obs = torch.tensor([[0.5, -0.5, 0.1], [-2.0, 1.0, 0.7]], dtype=torch.float32)
    actions = policy._predict(obs, deterministic=True)
    assert actions.shape[0] == obs.shape[0]


@pytest.mark.parametrize(
    "dpp_config",
    ["solve_cont_act_pia.yaml", "solve_cont_act_md.yaml", "solve_cont_act_proj_pg.yaml"],
)
def test_dpp_setup_is_float64(dpp_config):
    import run_dpp_based

    env_cfg = run_dpp_based.load_yaml(str(ENV_CONFIG))
    dpp_cfg = run_dpp_based.load_yaml(str(ROOT / "configs" / dpp_config))
    _, policy, *_ = run_dpp_based.setup_env_and_policy(dict(env_cfg), dpp_cfg)
    assert_tables(policy, torch.float64)


def test_projected_lqr_still_builds():
    import run_dpp_based

    env_cfg = run_dpp_based.load_yaml(str(ENV_CONFIG))
    dpp_cfg = run_dpp_based.load_yaml(str(ROOT / "configs" / "solve_proj_lqr.yaml"))
    _, policy, *_ = run_dpp_based.setup_env_and_policy(dict(env_cfg), dpp_cfg)
    assert policy is not None
