"""
Checks for the float64 option of QuantizedTabularPolicy (param_dtype).

1. RL code paths are unchanged:
   - the default dtype of the tabular parameters is float32;
   - constructing the policy consumes the global torch RNG exactly as the
     original (pre-patch) _build did, so seeded RL runs are reproducible as before;
   - RL models that use QuantizedTabularPolicy (TabularMD, and PPO with the tabular
     policy) build and run a few updates, and their tables stay float32 and finite.
2. The DPP solvers get float64 tables for SPI and exact MD; projected PG is unchanged.

Run from anywhere:  pytest tests/test_tabular_policy_dtype.py -v
"""
import pathlib
import sys

import pytest
import torch
import torch.nn as nn
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sde_md.envs.sde_env import SDEDiffusionEnvDiscAct  # noqa: E402
from sde_md.policies.quantizedtabularpolicy import QuantizedTabularPolicy  # noqa: E402
from sde_md.rl_based.tabularmd import TabularMD  # noqa: E402

RL_ENV_CONFIG = ROOT / "configs" / "env_config_disc_act_rl.yaml"
GRID_BINS = 10


def load_env_cfg(path=RL_ENV_CONFIG):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.pop("env_type", None)
    return cfg


def policy_kwargs(cfg, **extra):
    return dict(grid_bins=GRID_BINS, radius=cfg["radius"], T=cfg["T"], dt=cfg["dt"], **extra)


def make_policy(env, cfg, cls=QuantizedTabularPolicy, **extra):
    return cls(
        observation_space=env.observation_space,
        action_space=env.action_space,
        lr_schedule=lambda _: 0.01,
        **policy_kwargs(cfg, **extra),
    )


class LegacyQuantizedTabularPolicy(QuantizedTabularPolicy):
    """_build exactly as it was before the float64 patch (reference for RNG test)."""

    def _build(self, lr_schedule):
        self.tabular_params = nn.ParameterDict(
            {
                "action_logits": nn.Parameter(
                    torch.randn(self.total_states, self.action_dim) * 0.0
                ),
                "value_table": nn.Parameter(torch.randn(self.total_states, 1) * 0.0),
            }
        )
        for name in ("mlp_extractor", "action_net", "value_net"):
            if hasattr(self, name):
                delattr(self, name)
        if self.action_logits_to_optimizer:
            params = self.parameters()
        else:
            params = [self.tabular_params["value_table"]]
        self.optimizer = self.optimizer_class(params, lr=lr_schedule(1), **self.optimizer_kwargs)


def assert_tables(policy, dtype):
    for name, p in policy.tabular_params.items():
        assert p.dtype == dtype, f"{name}: {p.dtype} != {dtype}"
        assert torch.isfinite(p).all(), f"{name} has non-finite entries"


@pytest.fixture(scope="module")
def env_and_cfg():
    cfg = load_env_cfg()
    return SDEDiffusionEnvDiscAct(**cfg), cfg


# ---------------------------------------------------------------- policy level

def test_default_dtype_is_float32(env_and_cfg):
    env, cfg = env_and_cfg
    assert_tables(make_policy(env, cfg), torch.float32)


def test_float64_option(env_and_cfg):
    env, cfg = env_and_cfg
    policy = make_policy(env, cfg, param_dtype=torch.float64)
    assert_tables(policy, torch.float64)
    obs = torch.tensor([[0.5, -0.5, 0.1], [-2.0, 1.0, 0.7]], dtype=torch.float32)
    probs, log_probs = policy.get_action_probs(obs)
    assert probs.dtype == torch.float64 and torch.isfinite(probs).all()
    actions = policy._predict(obs, deterministic=False)
    assert actions.shape[0] == obs.shape[0]


@pytest.mark.parametrize("logits_to_optimizer", [False, True])
def test_rng_stream_unchanged(env_and_cfg, logits_to_optimizer):
    env, cfg = env_and_cfg
    torch.manual_seed(1234)
    new = make_policy(env, cfg, action_logits_to_optimizer=logits_to_optimizer)
    after_new = torch.rand(8)

    torch.manual_seed(1234)
    old = make_policy(env, cfg, cls=LegacyQuantizedTabularPolicy,
                      action_logits_to_optimizer=logits_to_optimizer)
    after_old = torch.rand(8)

    assert torch.equal(after_new, after_old), "policy construction changed the RNG stream"
    for name in ("action_logits", "value_table"):
        assert new.tabular_params[name].shape == old.tabular_params[name].shape
    n_new = sum(len(g["params"]) for g in new.optimizer.param_groups)
    n_old = sum(len(g["params"]) for g in old.optimizer.param_groups)
    assert n_new == n_old


# ------------------------------------------------------------------- RL level

def make_vec(cfg):
    return make_vec_env(SDEDiffusionEnvDiscAct, n_envs=1, seed=0,
                        vec_env_cls=DummyVecEnv, env_kwargs=cfg)


def test_tabular_md_builds_and_learns(env_and_cfg):
    _, cfg = env_and_cfg
    vec_env = make_vec(cfg)
    model = TabularMD(
        policy=QuantizedTabularPolicy,
        env=vec_env,
        learning_rate=0.2,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=1.0,
        md_lambda=1.0,
        ent_coef=1e-4,
        policy_kwargs=policy_kwargs(cfg, action_logits_to_optimizer=False),
        seed=0,
        device="cpu",
    )
    assert_tables(model.policy, torch.float32)
    model.learn(total_timesteps=128)
    assert_tables(model.policy, torch.float32)
    vec_env.close()


def test_ppo_with_tabular_policy_builds_and_learns(env_and_cfg):
    _, cfg = env_and_cfg
    vec_env = make_vec(cfg)
    model = PPO(
        policy=QuantizedTabularPolicy,
        env=vec_env,
        learning_rate=0.3,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        gamma=1.0,
        ent_coef=1e-4,
        vf_coef=0.05,
        policy_kwargs=policy_kwargs(cfg, action_logits_to_optimizer=True),
        seed=0,
        device="cpu",
    )
    assert_tables(model.policy, torch.float32)
    model.learn(total_timesteps=128)
    assert_tables(model.policy, torch.float32)
    vec_env.close()


# ------------------------------------------------------------------ DPP level

@pytest.mark.parametrize(
    "dpp_config, expected",
    [
        ("solve_spi.yaml", torch.float64),
        ("solve_md.yaml", torch.float64),
        ("solve_proj_pg.yaml", torch.float32),  # QuantizedTabularProbPolicy, unchanged
    ],
)
def test_dpp_setup_dtypes(dpp_config, expected):
    import run_dpp_based

    env_cfg = run_dpp_based.load_yaml(str(ROOT / "configs" / "env_config_disc_act.yaml"))
    dpp_cfg = run_dpp_based.load_yaml(str(ROOT / "configs" / dpp_config))
    _, policy, *_ = run_dpp_based.setup_env_and_policy(dict(env_cfg), dpp_cfg)
    assert_tables(policy, expected)