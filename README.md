# sde_md

Code for numerical experiments to validate some of the theory in [Mirror descent for constrained stochastic control problems](https://arxiv.org/abs/2506.02564).

## Structure
Resuable code is in `sde_md` which is a python package. 
- `envs` the code for Euler--Maruyama SDE stepping wrapped into gymnasium environment. 
- `evaluation` code for plotting results.
- `iterators` use FDM solvers from `solver` to implement algorithms like policy iteration and mirror descent stepping. 
- `policies` classes for storing the quantized policies and value functions etc. 
- `rl_based` all the sample based methods that "learn"
- `solvers` finite difference method (FDM) solvers 

Pre-saved experiment configurations are in `configs`. 

## Setup

1. Setup a virtual environment for python and install packages in `requirements.txt`. 

2. Build and test the C++ lib for SDE simulations
```
cd sde_md/envs/
python build_sde_core.py build_ext --inplace
```
then, while still in the same directory, check we can init the env and run tests
```
python sde_env.py 
```
Then `cd ../../` to go back to the level on which `run_rl.py`, `run_dpp_based.py`, etc. sit. 

## Running individual RL experiments 
First train: 
```
python run_rl.py --env_config configs/env_config_disc_act_rl.yaml --learn_config configs/learn_md_adv_rate_critic.yaml --train 
```
The algorithm produces Tensorboard logs in `logs_tb` so progress can be checked. 

Once training finishes model weights will be saved and can be executed in evalution:
```
python run_rl.py --env_config configs/env_config_disc_act_rl.yaml --learn_config configs/learn_md_adv_rate_critic.yaml --evaluate --model logs_md_adv_rate_critic/md_adv_rate_critic_model.zip
```

The training curves can also be plotted:
```
python plot_tb.py --logdir ./logs_tb/ --experiments md_adv_rate_critic_envs64_20260830_094743 --labels MD\ with\ advantage\ rate\ critic  --smooth 0.99 --scalars rollout/ep_rew_mean train/q_critic_loss --titles Episode\ reward\ mean Critic\ loss
```

## Running individual Dynamic programming principle DPP experiments 

These are moderately fast so it's easiest to run training and evaluation in one go
```
python run_dpp_based.py --solve_and_evaluate --env_config configs/env_config_disc_act.yaml --dpp_config configs/solve_spi.yaml 
```

## Running the convergence test for DPP experiments

This takes longer but is launched via:
```
python run_convergence_experiment.py --config ./configs/convergence_config.yaml
```
This is the one to run to reproduce the convergence figures in the preprint. 