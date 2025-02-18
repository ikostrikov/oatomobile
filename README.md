# OATomobile: A research framework for autonomous driving

  **[Overview](#overview)**
| **[Installation](#installation)**
| **[Baselines]**
| **[Paper]**

![PyPI Python Version](https://img.shields.io/pypi/pyversions/oatomobile)
![PyPI version](https://badge.fury.io/py/oatomobile.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2006.14911-b31b1b.svg)](https://arxiv.org/abs/2006.14911)
[![GitHub license](https://img.shields.io/pypi/l/oatomobile)](./LICENSE)

OATomobile is a library for autonomous driving research.
OATomobile strives to expose simple, efficient, well-tuned and readable agents, that serve both as reference implementations of popular algorithms and as strong baselines, while still providing enough flexibility to do novel research.

## Overview

If you just want to get started using OATomobile quickly, the first thing to know about the framework is that we wrap [CARLA] towns and scenarios in OpenAI [gym]s:

```python
import oatomobile

# Initializes a CARLA environment.
environment = oatomobile.envs.CARLAEnv(town="Town01")

# Makes an initial observation.
observation = environment.reset()
done = False

while not done:
  # Selects a random action.
  action = environment.action_space.sample()
  observation, reward, done, info = environment.step(action)

  # Renders interactive display.
  environment.render(mode="human")

# Book-keeping: closes
environment.close()
```

[Baselines] can also be used out-of-the-box:

```python
# Rule-based agents.
import oatomobile.baselines.rulebased

agent = oatomobile.baselines.rulebased.AutopilotAgent(environment)
action = agent.act(observation)

# Imitation-learners.
import torch
import oatomobile.baselines.torch

models = [oatomobile.baselines.torch.ImitativeModel() for _ in range(4)]
ckpts = ... # Paths to the model checkpoints.
for model, ckpt in zip(models, ckpts):
  model.load_state_dict(torch.load(ckpt))
agent = oatomobile.baselines.torch.RIPAgent(
  environment=environment,
  models=models,
  algorithm="WCM",
)
action = agent.act(observation)
```

## Installation

```bash
echo 'export CARLA_ROOT=$HOME/Carla_Latest/' >>~/.bashrc
source ~/.bashrc

mkdir $CARLA_ROOT
cd $CARLA_ROOT

wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/Dev/CARLA_Latest.tar.gz

tar xvf CARLA_Latest.tar.gz
rm CARLA_Latest.tar.gz

# Installs CARLA Python API.
echo 'export PYTHONPATH=$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg:$PYTHONPATH' >>~/.bashrc

source ~/.bashrc

mkdir -p ~/GitHub/
cd ~/GitHub/
git clone git@github.com:ikostrikov/oatomobile.git
cd oatomobile
pip install -e .
```

## Citing OATomobile

If you use OATomobile in your work, please cite the accompanying
[technical report][Paper]:

```bibtex
@inproceedings{filos2020can,
    title={Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?},
    author={Filos, Angelos and
            Tigas, Panagiotis and
            McAllister, Rowan and
            Rhinehart, Nicholas and
            Levine, Sergey and
            Gal, Yarin},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2020}
}
```

[Baselines]: oatomobile/baselines/
[Examples]: examples/
[CARLA]: https://carla.readthedocs.io/
[Paper]: https://arxiv.org/abs/2006.14911
[TensorFlow]: https://tensorflow.org
[PyTorch]: http://pytorch.org
[gym]: https://github.com/openai/gym
