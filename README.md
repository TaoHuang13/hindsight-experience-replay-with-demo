# Hindsight Experience Replay with Demonstrations
PyTorch implementation of the paper [Overcoming Exploration in Reinforcement Learning with Demonstrations](https://arxiv.org/abs/1709.1008) in surgical robot manipulation tasks.
<p align="left">
  <img width="98%" src="https://i.imgur.com/sVsrFtg.png">
</p>

## Acknowledgement
- [OpenAI Baselines](https://github.com/openai/baselines) for the tensorflow -based implementation.
- [SurRoL](https://github.com/med-air/SurRoL]) for the training and testing simulation platform.
- [DrQv2](https://github.com/facebookresearch/drqv2) for the coding structure and utils modules.
## Setup
We use Python 3.8 and Anaconda3 for development. To create an environment and install dependencies, run the following steps:
```shell
# Clone and cd into herdemo
git clone https://github.com/TaoHuang13/hindsight-experience-replay-with-demo.git
cd hindsight-experience-replay-with-demo

# Create and activate environment
conda create -n herdemo python=3.8 -y
conda activate herdemo

# Install dependencies
pip install -e .
```

Then add one line of code in `gym/gym/envs/__init__.py` to register SurRoL tasks:
```python
import surrol.gym
```


Run the following command to collect expert demonstration via the scripted policy in the individual task file:
```shell
python surrol/data/data_generation.py --env env_name
```
Here we have already provided demonstrations of several tasks. 

## Code Navigation
At a high-level, our code relies on the generic python script: `train.py` for training and evaluating RL agent. We use [hydra](https://hydra.cc/) for hyperparameterize this script with experiment-specific configuration. Specifically, all experiments should be configured in the directory `configs/` or command lines. 

The rest of code is organized as follows:
- `configs/` config files for launching expriments.
- `rl/` core implementation of [HER+DEMO](https://arxiv.org/abs/1709.1008) adopted from [OpenAI Baselines](https://github.com/openai/baselines).
- `surrol/` simulation platform for surgical robotic manipulation based on [PyBullet](https://github.com/bulletphysics/bullet3).
- `scripts/` bash scripts to running a batch of experiments.
- `train.py` generic python script for training and evaluating RL agent.

To simply start a experiment, run the following command:
```shell
sh scripts/run_herdemo.sh
```
