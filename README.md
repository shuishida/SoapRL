# SOAP-RL: Sequential Option Advantage Propagation for Reinforcement Learning in POMDP Environments
Shu Ishida, João F. Henriques (Visual Geometry Group, University of Oxford)

Paper: [SOAP-RL: Sequential Option Advantage Propagation for Reinforcement Learning in POMDP Environments](https://arxiv.org/abs/2407.18913)

This repository contains the code for [SOAP-RL](https://arxiv.org/abs/2407.18913).

This work compares ways of extending Reinforcement Learning algorithms to POMDPs with options. 
Options function as memory that allows the agent to retain historical information beyond the policy's context window. 
While option assignment could be handled using heuristics and hand-crafted objectives, learning temporally consistent options and associated sub-policies without explicit supervision is a challenge. 

Two algorithms, PPOEM and SOAP, are proposed and studied in depth to address this problem. 
PPOEM applies the forward-backward algorithm to optimize the expected returns for an option-augmented policy. 
However, it was shown that this learning approach is unstable during on-policy rollouts, and unsuited for learning causal policies without the knowledge of future trajectories, since option assignments are optimized for offline sequences where the entire episode is available. 
As an alternative approach, SOAP evaluates the policy gradient for an optimal option assignment. 
It extends the concept of the GAE to propagate option advantages through time, which is an analytical equivalent to performing temporal back-propagation of option policy gradients. 
With this approach, the option policy is only conditional on the history of the agent.

Evaluated against competing baselines, SOAP exhibited the most robust performance, correctly discovering options for POMDP corridor environments, as well as on standard benchmarks including Atari and MuJoCo, outperforming PPOEM, as well as LSTM and Option-Critic baselines. 

### Citation
If you find the code or paper useful, please consider citing:

```
@article{ishida2024soap,
  title={SOAP-RL: Sequential Option Advantage Propagation for Reinforcement Learning in POMDP Environments},
  author={Ishida, Shu and Henriques, Jo{\~a}o F},
  journal={arXiv preprint arXiv:2407.18913},
  year={2024}
} 
```

## Setup

Create a conda environment with Python 3.8 or above and install the required packages:

```bash
git clone https://github.com/shuishida/SoapRL.git

pip install stable-baselines3[extra]==2.2.1 rl_zoo3==2.2.1 wandb einops
```

### Installing Mujoco 2.1.0

1. Download the MuJoCo version 2.1 binaries for [Linux](https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz) or [OSX](https://mujoco.org/download/mujoco210-macos-x86_64.tar.gz).
2. Extract the downloaded `mujoco210` directory into `~/.mujoco/mujoco210`.
3. Run the following installation commands
    ```
    conda install -c conda-forge glew
    conda install -c conda-forge mesalib
    conda install -c menpo glfw3
    conda install patchelf
    pip install "cython<3"
    pip install mujoco-py==2.1.0
    ```
4. Add the following lines to your `~/.bashrc` file:
    ```
    export CPATH=$CONDA_PREFIX/include
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib:/home/<username>/.mujoco/mujoco210/bin
    ```
5. Run `source ~/.bashrc` to update your environment variables.
6. Run `python -c "import mujoco_py"` to test the installation.

## Run experiments

```bash
bash run_training.sh <env-name> <algo-name>
```

#### Examples of arguments that can be passed:

- `<env-name>`: Environment name
  - CartPole-v1
  - LunarLander-v2
  - Corridor environments
    - corridor-l<length> (e.g. `corridor-l3`, `corridor-l10`, `corridor-l20`)
  - Atari environments
    - BeamRiderNoFrameskip-v4
    - BreakoutNoFrameskip-v4
    - EnduroNoFrameskip-v4
    - PongNoFrameskip-v4
    - QbertNoFrameskip-v4
    - SeaquestNoFrameskip-v4
    - SpaceInvadersNoFrameskip-v4
    - MsPacmanNoFrameskip-v4
    - AsteroidsNoFrameskip-v4 
    - RoadRunnerNoFrameskip-v4
  - Mujoco environments
    - Ant-v3
    - HalfCheetah-v3
    - Humanoid-v3
    - Reacher-v3
    - Swimmer-v3
    - Walker2d-v3
  - many other RL environments from Gymnasium
- `<algo-name>`: Algorithm name
  - `soap`: SOAP-RL
  - `ppoem`: PPO-EM
  - `ppo`: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
  - `ppoc`: [Proximal Policy Option Critic](https://arxiv.org/abs/1712.00004)
  - `dac`: [Double Actor-Critic](https://arxiv.org/abs/1904.12691)
  - `ppo-lstm`: PPO with LSTM
  - many other algorithms from stable-baselines3
