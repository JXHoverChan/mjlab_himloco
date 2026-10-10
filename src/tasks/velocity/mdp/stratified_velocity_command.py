"""Command-manager adapter for the sampling curriculum in curriculums.py."""

import math
from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand, UniformVelocityCommandCfg,
)

from .curriculums import (
  DEFAULT_COMMAND_PROBABILITIES, STRATIFIED_COMMAND_MODES,
  sample_stratified_velocity_commands,
)


class StratifiedVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    if cfg.rel_world_envs != 0.0 or cfg.rel_forward_envs != 0.0:
      raise ValueError("Stratified commands use body-frame tasks; disable world/forward overrides")
    if cfg.init_velocity_prob != 0.0:
      raise ValueError("Stratified startup tasks require init_velocity_prob=0")
    self.mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    shape = (self.num_envs, len(STRATIFIED_COMMAND_MODES))
    self.task_steps = torch.zeros(shape, device=self.device)
    self.task_xy_error_sum = torch.zeros_like(self.task_steps)
    self.task_yaw_error_sum = torch.zeros_like(self.task_steps)
    self._record_samples = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.configure_sampling(DEFAULT_COMMAND_PROBABILITIES, (0.2, 0.8), (0.05, 1.0))

  def configure_sampling(self, probabilities, linear_speed_band, yaw_speed_band):
    if set(probabilities) != set(STRATIFIED_COMMAND_MODES):
      raise ValueError("mode_probabilities must specify all seven command modes")
    weights = [float(probabilities[name]) for name in STRATIFIED_COMMAND_MODES]
    if any(not math.isfinite(w) or w < 0 for w in weights) or not math.isclose(sum(weights), 1.0):
      raise ValueError("mode_probabilities must be finite, nonnegative and sum to one")
    if probabilities["combined"] <= 0.0:
      raise ValueError("combined probability must be positive for range adaptation")
    for band in (linear_speed_band, yaw_speed_band):
      if len(band) != 2 or not all(math.isfinite(v) for v in band) or not 0 < band[0] <= band[1]:
        raise ValueError("speed bands need finite 0 < minimum <= maximum")
    ranges = self.cfg.ranges
    for bounds, minimum in ((ranges.lin_vel_x, linear_speed_band[0]),
                             (ranges.lin_vel_y, linear_speed_band[0]),
                             (ranges.ang_vel_z, yaw_speed_band[0])):
      if bounds[0] > -minimum or bounds[1] < minimum:
        raise ValueError("current ranges must contain both directions of each minimum speed")
    self.probabilities = torch.tensor(weights, device=self.device)
    self.linear_speed_band = tuple(linear_speed_band)
    self.yaw_speed_band = tuple(yaw_speed_band)

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    values, modes = sample_stratified_velocity_commands(
      self.cfg.ranges, self.probabilities, len(env_ids),
      self.linear_speed_band, self.yaw_speed_band,
    )
    self.vel_command_b[env_ids] = values
    self.mode[env_ids] = modes
    combined = modes == STRATIFIED_COMMAND_MODES.index("combined")
    # Only the full-range task retains the original heading feedback mode.
    self.is_heading_env[env_ids] &= combined
    self.is_standing_env[env_ids] = modes == STRATIFIED_COMMAND_MODES.index("standing")
    self.is_world_env[env_ids] = False
    self.is_forward_env[env_ids] = False

  def compute(self, dt, env_ids=None):
    # reset uses dt=0: do not count its initial observation as a physical step.
    if env_ids is None:
      self._record_samples[:] = dt > 0
    else:
      self._record_samples.zero_()
    super().compute(dt, env_ids)

  def _update_metrics(self):
    super()._update_metrics()
    ids = self._record_samples.nonzero(as_tuple=False).flatten()
    modes = self.mode[ids]
    xy_error = torch.linalg.vector_norm(
      self.vel_command_b[ids, :2] - self.robot.data.root_link_lin_vel_b[ids, :2], dim=-1,
    )
    yaw_error = (
      self.vel_command_b[ids, 2] - self.robot.data.root_link_ang_vel_b[ids, 2]
    ).abs()
    self.task_steps[ids, modes] += 1
    self.task_xy_error_sum[ids, modes] += xy_error
    self.task_yaw_error_sum[ids, modes] += yaw_error

  def sampling_metrics(self, env_ids):
    steps = self.task_steps[env_ids].sum(0)
    xy_sum = self.task_xy_error_sum[env_ids].sum(0)
    yaw_sum = self.task_yaw_error_sum[env_ids].sum(0)
    packed = torch.stack((steps / steps.sum().clamp_min(1),
                          xy_sum / steps.clamp_min(1),
                          yaw_sum / steps.clamp_min(1), steps), dim=1).cpu().tolist()
    metrics = {}
    for name, (fraction, xy, yaw, samples) in zip(STRATIFIED_COMMAND_MODES, packed, strict=True):
      metrics[f"fraction_{name}"] = fraction
      metrics[f"xy_error_{name}"] = xy
      metrics[f"yaw_error_{name}"] = yaw
      metrics[f"samples_{name}"] = samples
    return metrics

  def reset(self, env_ids):
    extras = super().reset(env_ids)
    self.task_steps[env_ids] = 0
    self.task_xy_error_sum[env_ids] = 0
    self.task_yaw_error_sum[env_ids] = 0
    return extras


@dataclass(kw_only=True)
class StratifiedVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env):
    return StratifiedVelocityCommand(self, env)
