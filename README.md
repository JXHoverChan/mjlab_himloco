# mjlab HIM Go2

This repository contains the HIM locomotion adaptation for Unitree Go2 on top
of [mjlab](https://github.com/mujocolab/mjlab). The repository registers one
task: `Unitree-Go2-HIM`.

The implementation includes:

- a history-based source encoder;
- a target encoder and prototype-based contrastive objective;
- source velocity prediction loss;
- a HIM-aware actor, rollout storage, and PPO algorithm;
- the Go2 MJCF and mesh assets required by the environment.

## Installation

Use Python 3.10-3.13 and install the project in editable mode:

```bash
git clone https://github.com/JXHoverChan/mjlab_himloco.git
cd mjlab_himloco
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The project pins `mjlab==1.6.0`. That version supplies the matching
`rsl-rl-lib`, `mujoco-warp`, MuJoCo, and other runtime dependencies.

## Verify the task

Run a small CPU smoke test:

```bash
python scripts/train.py Unitree-Go2-HIM \
  --gpu-ids None \
  --env.scene.num-envs 8 \
  --agent.max-iterations 1 \
  --agent.logger tensorboard
```

For a real training run, increase `--env.scene.num-envs` and
`--agent.max-iterations`. Training outputs are written under `logs/`, which
is intentionally ignored by Git.

## HIM task layout

```text
src/tasks/velocity/config/go2/   Go2 environment and HIM configuration
src/tasks/velocity/rl/himactor.py
src/tasks/velocity/rl/himencoder.py
src/tasks/velocity/rl/himppo.py
src/tasks/velocity/rl/himstorage.py
src/assets/robots/unitree_go2/   Go2 MJCF and mesh assets
```
