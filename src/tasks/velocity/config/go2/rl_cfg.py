"""RL configuration for Unitree Go2 velocity task."""

from dataclasses import dataclass, field
from typing import Any

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


@dataclass
class HIMEncoderCfg:
  """Configuration for one source or target HIM encoder."""

  class_name: str = "src.tasks.velocity.rl.himencoder:HIMEncoder"
  hidden_dims: tuple[int, ...] = (128, 64)
  latent_dim: int = 16
  velocity_dim: int = 3
  activation: str = "elu"
  obs_normalization: bool = True
  predict_velocity: bool = False


@dataclass
class HIMAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO configuration plus the HIM auxiliary objective."""

  class_name: str = "src.tasks.velocity.rl.himppo:HIM_PPO"
  encoder_learning_rate: float = 1.0e-3
  encoder_max_grad_norm: float = 10.0
  num_prototypes: int = 32
  temperature: float = 3.0
  sinkhorn_epsilon: float = 0.05
  sinkhorn_iterations: int = 3
  estimation_loss_coef: float = 1.0
  swap_loss_coef: float = 1.0
  source_obs_group: str = "actor"
  target_obs_group: str = "target_encoder"
  velocity_obs_group: str = "velocity"
  source_encoder: HIMEncoderCfg = field(
    default_factory=lambda: HIMEncoderCfg(predict_velocity=True)
  )
  target_encoder: HIMEncoderCfg = field(default_factory=HIMEncoderCfg)
  rnd_cfg: dict[str, Any] | None = None
  symmetry_cfg: dict[str, Any] | None = None


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_go2_him_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the HIM training configuration for Unitree Go2."""
  return RslRlOnPolicyRunnerCfg(
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    actor=RslRlModelCfg(
      class_name="src.tasks.velocity.rl.himactor:HIMActor",
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=HIMAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      encoder_learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      encoder_max_grad_norm=10.0,
      num_prototypes=32,
      temperature=3.0,
      sinkhorn_epsilon=0.05,
      sinkhorn_iterations=3,
      estimation_loss_coef=1.0,
      swap_loss_coef=1.0,
    ),
    experiment_name="go2_him",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )
