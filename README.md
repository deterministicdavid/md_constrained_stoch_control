# sde_md

Code for numerical experiments to validate some of the theory in [Mirror descent for constrained stochastic control problems](https://arxiv.org/abs/2506.02564).

## Structure
Reusable code is in `sde_md`, which is a Python package.
- `envs` the code for Euler--Maruyama SDE stepping wrapped into a gymnasium environment.
- `evaluation` code for plotting results.
- `iterators` use the FDM solvers from `solvers` to implement algorithms like (soft) policy iteration and mirror descent stepping.
- `policies` classes for storing the quantized policies and value functions etc.
- `rl_based` all the sample based methods that "learn".
- `solvers` finite difference method (FDM) solvers.

Pre-saved experiment configurations are in `configs` and tests are in `tests`.

## Setup

1. Create a virtual environment and install the packages in `requirements.txt`. Run this from the repo root, since `requirements.txt` installs `sde_md` itself in editable mode:
```
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```
Activate the environment (`source venv/bin/activate`) in every new terminal.
On Linux without a GPU, installing the CPU build of PyTorch first avoids downloading the CUDA libraries:
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

2. Build the C++ lib for SDE simulations (needs a C++ compiler):
```
cd sde_md/envs/
python build_sde_core.py build_ext --inplace
cd ../../
```
Then, from the repo root (where `run_rl.py`, `run_dpp_based.py`, etc. sit), check that the environment can be initialised:
```
python sde_md/envs/sde_env.py
```

3. Run the tests:
```
pytest tests/
```

## Running individual RL experiments
First train:
```
python run_rl.py --env_config configs/env_config_disc_act_rl.yaml --learn_config configs/learn_md_adv_rate_critic.yaml --train
```
The algorithm produces Tensorboard logs in `logs_tb` so progress can be checked.

Once training finishes, the model weights are saved and can be used for evaluation:
```
python run_rl.py --env_config configs/env_config_disc_act_rl.yaml --learn_config configs/learn_md_adv_rate_critic.yaml --evaluate --model logs_md_adv_rate_critic/md_adv_rate_critic_model.zip
```

The training curves can also be plotted:
```
python plot_tb.py --logdir ./logs_tb/ --experiments md_adv_rate_critic_envs64_20260830_094743 --labels MD\ with\ advantage\ rate\ critic  --smooth 0.99 --scalars rollout/ep_rew_mean train/q_critic_loss --titles Episode\ reward\ mean Critic\ loss
```

## Running individual dynamic programming principle (DPP) experiments

These are moderately fast, so it's easiest to run solving and evaluation in one go:
```
python run_dpp_based.py --solve_and_evaluate --env_config configs/env_config_disc_act.yaml --dpp_config configs/solve_spi.yaml
```
Soft policy iteration and exact mirror descent store their policy tables in double precision (`param_dtype` in `QuantizedTabularPolicy`). In single precision the value function of a policy is only reproducible to about `1e-6`, which hides the convergence below that level. The RL code uses the default single precision.

## Running the convergence tests for DPP experiments

These take longer. To reproduce the convergence figures in the paper for both examples:
```
python run_convergence_experiment.py --config ./configs/convergence_config.yaml
```
Useful options: `--exp NAME` runs a single experiment from the config, `--skip_mc` skips the Monte Carlo evaluation of the returns, and `--load_ref` reuses a previously saved reference solution instead of recomputing it.

### Exponential convergence for larger entropy regularisation

To reproduce the semi-log figure showing exponential convergence of mirror descent for the entropy-regularised problem with `tau` in `{0.1, 0.05, 0.02}`, run the three experiments (the reference for each is computed by soft policy iteration with the same `tau`):
```
python run_convergence_experiment.py --config configs/convergence_config_tau.yaml --exp Disc_tau_0.1 --skip_mc
python run_convergence_experiment.py --config configs/convergence_config_tau.yaml --exp Disc_tau_0.05 --skip_mc
python run_convergence_experiment.py --config configs/convergence_config_tau.yaml --exp Disc_tau_0.02 --skip_mc
```
and then plot:
```
python plot_semilog.py --floor 1e-11 --runs \
    convergence_results_disc_tau_0.1/convergence.csv:0.1 \
    convergence_results_disc_tau_0.05/convergence.csv:0.05 \
    convergence_results_disc_tau_0.02/convergence.csv:0.02
```
This writes `convergence_semilog.pdf` and prints the measured decay rates, which can be compared with the predicted rate `tau * eta` (dual step size `eta = md_lambda = 1`).