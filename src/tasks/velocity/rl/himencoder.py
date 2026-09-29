"""History and privileged-observation encoders for HIM training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from rsl_rl.modules import EmpiricalNormalization, MLP


class HIMEncoder(nn.Module):
  """Encode a history or a one-step target observation into a HIM latent.

  The source encoder can also predict base linear velocity. The target encoder
  uses the same class with ``predict_velocity=False``. Source and target are
  instantiated separately, so their parameters are independent.
  """

  def __init__(
    self,
    input_dim: int,
    history_length: int,
    hidden_dims: tuple[int, ...] | list[int] = (128, 64),
    latent_dim: int = 16,
    velocity_dim: int = 3,
    activation: str = "elu",
    obs_normalization: bool = False,
    predict_velocity: bool = False,
  ) -> None:
    super().__init__()

    if input_dim <= 0:
      raise ValueError(f"input_dim must be positive, got {input_dim}")
    if history_length <= 0:
      raise ValueError(f"history_length must be positive, got {history_length}")
    if latent_dim <= 0:
      raise ValueError(f"latent_dim must be positive, got {latent_dim}")
    if not hidden_dims:
      raise ValueError("hidden_dims must contain at least one layer")

    self.input_dim = input_dim
    self.history_length = history_length
    self.flat_input_dim = input_dim * history_length
    self.latent_dim = latent_dim
    self.velocity_dim = velocity_dim
    self.predict_velocity = predict_velocity

    self.obs_normalization = obs_normalization
    self.obs_normalizer: nn.Module
    if obs_normalization:
      self.obs_normalizer = EmpiricalNormalization(self.flat_input_dim)
    else:
      self.obs_normalizer = nn.Identity()

    output_dim = latent_dim + (velocity_dim if predict_velocity else 0)
    self.mlp = MLP(self.flat_input_dim, output_dim, hidden_dims, activation)

  def _flatten_input(self, observations: torch.Tensor) -> torch.Tensor:
    """Convert ``[B, H, D]`` or ``[B, H*D]`` input to ``[B, H*D]``."""
    if observations.ndim == 3:
      if observations.shape[1] != self.history_length or observations.shape[2] != self.input_dim:
        raise ValueError(
          f"Expected observations with shape [B, {self.history_length}, {self.input_dim}], "
          f"got {tuple(observations.shape)}"
        )
      return observations.reshape(observations.shape[0], self.flat_input_dim)

    if observations.ndim == 2:
      if observations.shape[1] != self.flat_input_dim:
        raise ValueError(
          f"Expected flattened observations with last dimension {self.flat_input_dim}, "
          f"got {tuple(observations.shape)}"
        )
      return observations

    raise ValueError(f"Expected a 2D or 3D observation tensor, got {tuple(observations.shape)}")

  def _forward_raw(self, observations: torch.Tensor) -> torch.Tensor:
    flat_observations = self._flatten_input(observations)
    normalized_observations = self.obs_normalizer(flat_observations)
    return self.mlp(normalized_observations)

  def encode(self, observations: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Return optional velocity prediction and normalized latent."""
    output = self._forward_raw(observations)
    if self.predict_velocity:
      velocity = output[..., : self.velocity_dim]
      latent = output[..., self.velocity_dim :]
    else:
      velocity = None
      latent = output
    latent = F.normalize(latent, dim=-1, p=2)
    return velocity, latent

  def policy_features(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return source velocity and latent features for the actor."""
    if not self.predict_velocity:
      raise RuntimeError("policy_features() requires a source encoder with predict_velocity=True")
    velocity, latent = self.encode(observations)
    assert velocity is not None
    return velocity, latent

  def forward(self, observations: torch.Tensor) -> torch.Tensor:
    """Return the normalized latent embedding."""
    return self.encode(observations)[1]

  @torch.no_grad()
  def update_normalization(self, observations: torch.Tensor) -> None:
    """Update the encoder's running observation statistics."""
    if not self.obs_normalization:
      return
    flat_observations = self._flatten_input(observations)
    normalization = self.obs_normalizer
    assert isinstance(normalization, EmpiricalNormalization)
    normalization.update(flat_observations)
