"""Actor model that consumes the HIM source encoder features."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, HiddenState, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable

from .himencoder import HIMEncoder


class HIMActor(nn.Module):
  """Policy model using the latest proprioception and detached HIM features."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    source_encoder: HIMEncoder,
    source_obs_group: str = "actor",
    hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
  ) -> None:
    super().__init__()
    del obs_groups, obs_set

    if source_obs_group not in obs:
      raise KeyError(f"Source observation group '{source_obs_group}' is not present in {list(obs.keys())}")
    source_observation = obs[source_obs_group]
    if source_observation.ndim != 3:
      raise ValueError(
        "HIMActor requires the source observation group to have shape [N, H, D]. "
        f"Got {tuple(source_observation.shape)} for '{source_obs_group}'."
      )
    if source_observation.shape[1] != source_encoder.history_length:
      raise ValueError(
        f"Source history length mismatch: observation has {source_observation.shape[1]}, "
        f"encoder expects {source_encoder.history_length}"
      )
    if source_observation.shape[2] != source_encoder.input_dim:
      raise ValueError(
        f"Source observation dimension mismatch: observation has {source_observation.shape[2]}, "
        f"encoder expects {source_encoder.input_dim}"
      )
    if not source_encoder.predict_velocity:
      raise ValueError("HIMActor requires a source encoder with predict_velocity=True")
    if not hidden_dims:
      raise ValueError("hidden_dims must contain at least one layer")

    self.source_obs_group = source_obs_group
    self.source_encoder = source_encoder
    self.one_step_obs_dim = int(source_observation.shape[-1])
    self.history_length = int(source_observation.shape[-2])
    self.obs_dim = self.one_step_obs_dim

    self.obs_normalization = obs_normalization
    if obs_normalization:
      self.obs_normalizer: nn.Module = EmpiricalNormalization(self.one_step_obs_dim)
    else:
      self.obs_normalizer = nn.Identity()

    self.policy_input_dim = self.one_step_obs_dim + source_encoder.velocity_dim + source_encoder.latent_dim

    if distribution_cfg is not None:
      distribution_cfg = copy.deepcopy(distribution_cfg)
      dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
      self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
      mlp_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      mlp_output_dim = output_dim

    self.mlp = MLP(self.policy_input_dim, mlp_output_dim, hidden_dims, activation)
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _source_history(self, obs: TensorDict) -> torch.Tensor:
    history = obs[self.source_obs_group]
    if history.ndim == 2 and self.history_length == 1:
      history = history.unsqueeze(1)
    if history.ndim != 3:
      raise ValueError(f"Expected source history with 3 dimensions, got {tuple(history.shape)}")
    return history

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
  ) -> torch.Tensor:
    del masks, hidden_state
    history = self._source_history(obs)
    current_observation = self.obs_normalizer(history[:, -1, :])

    # HIM encoders are updated only by the contrastive/velocity objective.
    with torch.no_grad():
      velocity, latent = self.source_encoder.policy_features(history)
    return torch.cat((current_observation, velocity, latent), dim=-1)

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    latent = self.get_latent(obs, masks, hidden_state)
    mlp_output = self.mlp(latent)
    if self.distribution is None:
      return mlp_output
    if stochastic_output:
      self.distribution.update(mlp_output)
      return self.distribution.sample()
    return self.distribution.deterministic_output(mlp_output)

  def update_normalization(self, obs: TensorDict) -> None:
    if not self.obs_normalization:
      return
    history = self._source_history(obs)
    normalization = self.obs_normalizer
    assert isinstance(normalization, EmpiricalNormalization)
    normalization.update(history[:, -1, :])

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    del dones, hidden_state

  def get_hidden_state(self) -> HiddenState:
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    assert self.distribution is not None
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.kl_divergence(old_params, new_params)

  def as_jit(self) -> nn.Module:
    return _HIMActorExport(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    del verbose
    return _HIMActorExport(self)


class _HIMActorExport(nn.Module):
  """Tensor-only deterministic policy wrapper for JIT and ONNX export."""

  def __init__(self, actor: HIMActor) -> None:
    super().__init__()
    self.source_normalizer = copy.deepcopy(actor.source_encoder.obs_normalizer)
    self.source_mlp = copy.deepcopy(actor.source_encoder.mlp)
    self.obs_normalizer = copy.deepcopy(actor.obs_normalizer)
    self.mlp = copy.deepcopy(actor.mlp)
    assert actor.distribution is not None
    self.deterministic_output = actor.distribution.as_deterministic_output_module()
    self.history_length = actor.history_length
    self.one_step_obs_dim = actor.one_step_obs_dim
    self.flat_source_dim = actor.source_encoder.flat_input_dim
    self.velocity_dim = actor.source_encoder.velocity_dim

  def forward(self, history: torch.Tensor) -> torch.Tensor:
    flat_history = history.reshape(history.shape[0], self.flat_source_dim)
    source_output = self.source_mlp(self.source_normalizer(flat_history))
    velocity = source_output[..., : self.velocity_dim]
    latent = source_output[..., self.velocity_dim :]
    latent = torch.nn.functional.normalize(latent, dim=-1, p=2.0)
    current_observation = self.obs_normalizer(history[:, -1, :])
    policy_input = torch.cat((current_observation, velocity, latent), dim=-1)
    return self.deterministic_output(self.mlp(policy_input))

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.history_length, self.one_step_obs_dim),)

  @property
  def input_names(self) -> list[str]:
    return ["obs_history"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
