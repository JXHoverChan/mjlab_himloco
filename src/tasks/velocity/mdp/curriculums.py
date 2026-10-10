from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.curriculum_manager import CurriculumTermCfg

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")

STRATIFIED_COMMAND_MODES = (
  "standing", "low_forward", "low_backward", "lateral", "pure_yaw",
  "translation", "combined",
)
DEFAULT_COMMAND_PROBABILITIES = {
  "standing": 0.05, "low_forward": 0.15, "low_backward": 0.15,
  "lateral": 0.10, "pure_yaw": 0.20, "translation": 0.10, "combined": 0.25,
}


def sample_stratified_velocity_commands(
  ranges: object,
  probabilities: torch.Tensor,
  count: int,
  linear_speed_band: tuple[float, float],
  yaw_speed_band: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Sample a mixture of basic skills and the current full-range task.

  Basic skill bands stay fixed as the range curriculum advances, intersected
  with its current bounds. Signed skills choose the two directions equally;
  combined commands retain the uniform Cartesian sampling distribution.
  """
  device = probabilities.device
  modes = torch.multinomial(probabilities, count, replacement=True)
  values = torch.zeros(count, 3, device=device)
  bounds = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
  random = torch.rand(count, 3, device=device)
  for axis, (lo, hi) in enumerate(bounds):
    values[:, axis] = lo + random[:, axis] * (hi - lo)

  def signed_speed(axis: int, band: tuple[float, float]) -> torch.Tensor:
    lo, hi = bounds[axis]
    positive = torch.rand(count, device=device) < 0.5
    limit = torch.where(positive, min(hi, band[1]), min(-lo, band[1]))
    magnitude = band[0] + torch.rand(count, device=device) * (limit - band[0])
    return torch.where(positive, magnitude, -magnitude)

  basic = modes != STRATIFIED_COMMAND_MODES.index("combined")
  values[basic] = 0.0
  forward = modes == STRATIFIED_COMMAND_MODES.index("low_forward")
  backward = modes == STRATIFIED_COMMAND_MODES.index("low_backward")
  low, high = linear_speed_band
  values[forward, 0] = low + random[forward, 0] * (min(bounds[0][1], high) - low)
  values[backward, 0] = -(low + random[backward, 0] * (min(-bounds[0][0], high) - low))
  lateral = modes == STRATIFIED_COMMAND_MODES.index("lateral")
  values[lateral, 1] = signed_speed(1, linear_speed_band)[lateral]
  turning = modes == STRATIFIED_COMMAND_MODES.index("pure_yaw")
  values[turning, 2] = signed_speed(2, yaw_speed_band)[turning]
  translating = modes == STRATIFIED_COMMAND_MODES.index("translation")
  along_x = torch.rand(count, device=device) < 0.5
  for axis in (0, 1):
    selected = translating & (along_x if axis == 0 else ~along_x)
    full_band = (low, max(-bounds[axis][0], bounds[axis][1]))
    values[selected, axis] = signed_speed(axis, full_band)[selected]
  return values, modes


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def _resolve_env_ids(
  env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None
) -> torch.Tensor:
  """Convert the curriculum manager's reset selector to explicit environment ids."""
  all_env_ids = torch.arange(env.num_envs, device=env.device)
  if env_ids is None:
    return all_env_ids
  if isinstance(env_ids, slice):
    return all_env_ids[env_ids]
  return env_ids


def _restore_episode_mean_error(
  env: ManagerBasedRlEnv,
  command_term: object,
  env_ids: torch.Tensor,
  metric_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Restore a per-step command error from UniformVelocityCommand's accumulator."""
  metrics = getattr(command_term, "metrics").get(metric_name)
  if metrics is None:
    raise KeyError(f"Command term does not expose {metric_name!r}")

  episode_lengths = env.episode_length_buf[env_ids]
  valid = episode_lengths > 0
  errors = torch.zeros_like(metrics[env_ids])
  if not torch.any(valid):
    return errors, valid

  resampling_time_range = getattr(command_term, "cfg").resampling_time_range
  max_command_steps = resampling_time_range[1] / env.step_dt
  if max_command_steps <= 0.0:
    raise ValueError("resampling_time_range must have a positive upper bound")

  errors[valid] = (
    metrics[env_ids][valid]
    * max_command_steps
    / episode_lengths[valid].to(dtype=metrics.dtype)
  )
  return errors, valid


def _classify_terrain_update(
  distance: torch.Tensor,
  expected_distance: torch.Tensor,
  linear_error: torch.Tensor,
  yaw_error: torch.Tensor,
  active: torch.Tensor,
  terrain_up_distance: float,
  linear_error_up_threshold: float,
  linear_error_down_threshold: float,
  yaw_error_up_threshold: float,
  yaw_error_down_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Classify per-episode terrain updates using separate task metrics."""
  move_up = (
    active
    & (distance > terrain_up_distance)
    & (linear_error < linear_error_up_threshold)
    & (yaw_error < yaw_error_up_threshold)
  )
  move_down = (
    active
    & ~move_up
    & (
      (distance < expected_distance)
      | (linear_error > linear_error_down_threshold)
      | (yaw_error > yaw_error_down_threshold)
    )
  )
  return move_up, move_down


class AdaptiveTerrainCurriculum:
  """Adapt terrain levels from terrain, planar tracking, and yaw performance.

  Horizontal displacement is used only as the terrain-passage metric. Planar
  velocity and yaw velocity errors are evaluated independently from the command
  term's completed-episode accumulators. Episodes with no meaningful planar
  command do not change terrain levels, because a pure-yaw or standing episode
  does not test terrain traversal.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    self._command_name = str(cfg.params["command_name"])
    self._command_activity_threshold = float(
      cfg.params.get("command_activity_threshold", 0.1)
    )
    self._terrain_down_distance_ratio = float(
      cfg.params.get("terrain_down_distance_ratio", 0.5)
    )
    self._linear_error_up_threshold = float(cfg.params["linear_error_up_threshold"])
    self._linear_error_down_threshold = float(
      cfg.params["linear_error_down_threshold"]
    )
    self._yaw_error_up_threshold = float(cfg.params["yaw_error_up_threshold"])
    self._yaw_error_down_threshold = float(cfg.params["yaw_error_down_threshold"])

    if self._command_activity_threshold < 0.0:
      raise ValueError("command_activity_threshold must be non-negative")
    if not 0.0 < self._terrain_down_distance_ratio <= 1.0:
      raise ValueError("terrain_down_distance_ratio must be in (0, 1]")
    if self._linear_error_up_threshold >= self._linear_error_down_threshold:
      raise ValueError(
        "linear_error_up_threshold must be lower than linear_error_down_threshold"
      )
    if self._yaw_error_up_threshold >= self._yaw_error_down_threshold:
      raise ValueError(
        "yaw_error_up_threshold must be lower than yaw_error_down_threshold"
      )
    if self._linear_error_up_threshold <= 0.0 or self._yaw_error_up_threshold <= 0.0:
      raise ValueError("tracking error thresholds must be positive")

    command_term = env.command_manager.get_term(self._command_name)
    if command_term is None:
      raise ValueError(f"Command term {self._command_name!r} does not exist")
    self._command_term = command_term

  @staticmethod
  def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not torch.any(mask):
      return 0.0
    return float(values[mask].mean().item())

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
    **_kwargs: object,
  ) -> dict[str, float]:
    selected_env_ids = _resolve_env_ids(env, env_ids)
    asset: Entity = env.scene[asset_cfg.name]

    terrain = env.scene.terrain
    assert terrain is not None
    terrain_generator = terrain.cfg.terrain_generator
    assert terrain_generator is not None

    command = env.command_manager.get_command(self._command_name)
    assert command is not None
    command = command[selected_env_ids]

    distance = torch.norm(
      asset.data.root_link_pos_w[selected_env_ids, :2]
      - env.scene.env_origins[selected_env_ids, :2],
      dim=1,
    )
    command_xy_norm = torch.norm(command[:, :2], dim=1)
    active = command_xy_norm > self._command_activity_threshold

    linear_error, linear_valid = _restore_episode_mean_error(
      env, self._command_term, selected_env_ids, "error_vel_xy"
    )
    yaw_error, yaw_valid = _restore_episode_mean_error(
      env, self._command_term, selected_env_ids, "error_vel_yaw"
    )
    valid = linear_valid & yaw_valid
    active &= valid

    terrain_up_distance = terrain_generator.size[0] / 2.0
    expected_distance = (
      command_xy_norm
      * env.max_episode_length_s
      * self._terrain_down_distance_ratio
    )
    move_up, move_down = _classify_terrain_update(
      distance=distance,
      expected_distance=expected_distance,
      linear_error=linear_error,
      yaw_error=yaw_error,
      active=active,
      terrain_up_distance=terrain_up_distance,
      linear_error_up_threshold=self._linear_error_up_threshold,
      linear_error_down_threshold=self._linear_error_down_threshold,
      yaw_error_up_threshold=self._yaw_error_up_threshold,
      yaw_error_down_threshold=self._yaw_error_down_threshold,
    )
    terrain.update_env_origins(selected_env_ids, move_up, move_down)

    active_count = active.sum()
    valid_count = valid.sum()
    changed = move_up | move_down
    active_denominator = torch.clamp(active_count, min=1)
    valid_denominator = torch.clamp(valid_count, min=1)
    terrain_reached = active & (distance > terrain_up_distance)
    linear_good = active & (linear_error < self._linear_error_up_threshold)
    yaw_good = active & (yaw_error < self._yaw_error_up_threshold)

    return {
      "level": float(torch.mean(terrain.terrain_levels.float()).item()),
      "mean_distance": self._masked_mean(distance, valid),
      "mean_expected_distance": self._masked_mean(expected_distance, active),
      "mean_linear_error": self._masked_mean(linear_error, valid),
      "mean_yaw_error": self._masked_mean(yaw_error, valid),
      "active_rate": float((active.float().sum() / valid_denominator).item()),
      "terrain_reached_rate": float(
        (terrain_reached.float().sum() / active_denominator).item()
      ),
      "linear_good_rate": float(
        (linear_good.float().sum() / active_denominator).item()
      ),
      "yaw_good_rate": float((yaw_good.float().sum() / active_denominator).item()),
      "move_up_rate": float((move_up.float().sum() / active_denominator).item()),
      "move_down_rate": float((move_down.float().sum() / active_denominator).item()),
      "updated": float((changed.float().sum() / torch.clamp(valid.sum(), min=1)).item()),
    }


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


class AdaptiveVelocityCurriculum:
  """Adapt velocity command ranges from completed-episode tracking error.

  This is a stateful curriculum term. The curriculum manager calls it during
  environment reset, before command metrics are cleared, so each completed
  episode can contribute its normalized linear-velocity error to a window.
  """

  _RANGE_NAMES = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    self._env = env
    self._command_name = str(cfg.params["command_name"])
    self._initial_ranges = self._read_ranges(cfg.params["initial_ranges"])
    self._maximum_ranges = self._read_ranges(cfg.params["maximum_ranges"])

    for name in self._RANGE_NAMES:
      initial = self._initial_ranges[name]
      maximum = self._maximum_ranges[name]
      if maximum[0] > initial[0] or maximum[1] < initial[1]:
        raise ValueError(
          f"maximum_ranges[{name!r}] must contain initial_ranges[{name!r}]"
        )

    self._error_up_threshold = float(cfg.params["error_up_threshold"])
    self._error_down_threshold = float(cfg.params["error_down_threshold"])
    self._level_step = float(cfg.params["level_step"])
    self._update_every_episodes = int(cfg.params["update_every_episodes"])
    if self._error_up_threshold >= self._error_down_threshold:
      raise ValueError("error_up_threshold must be lower than error_down_threshold")
    if self._error_up_threshold <= 0.0 or self._level_step <= 0.0:
      raise ValueError("error_up_threshold and level_step must be positive")
    if self._update_every_episodes < 1:
      raise ValueError("update_every_episodes must be at least one")

    command_term = env.command_manager.get_term(self._command_name)
    if command_term is None:
      raise ValueError(f"Command term {self._command_name!r} does not exist")
    self._command_term = command_term

    self._level = 0.0
    # 续练临时调整成 1
    # self._level = 1.0


    self._window_error_sum = 0.0
    self._window_episode_count = 0
    self._last_mean_error = 0.0
    self._last_update_episode_count = 0

    # The command config starts at the maximum range in the base environment.
    # Apply the adaptive curriculum's initial range before the first reset.
    self._apply_level()

  @staticmethod
  def _read_ranges(value: object) -> dict[str, tuple[float, float]]:
    if not isinstance(value, Mapping):
      raise TypeError("velocity ranges must be mappings")

    ranges: dict[str, tuple[float, float]] = {}
    for name in AdaptiveVelocityCurriculum._RANGE_NAMES:
      if name not in value:
        raise KeyError(f"velocity range {name!r} is missing")
      raw_range = value[name]
      if isinstance(raw_range, (str, bytes)) or not isinstance(raw_range, Sequence):
        raise TypeError(f"velocity range {name!r} must be a sequence")
      if len(raw_range) != 2:
        raise ValueError(f"velocity range {name!r} must contain two values")
      lower, upper = float(raw_range[0]), float(raw_range[1])
      if lower > upper:
        raise ValueError(f"velocity range {name!r} has lower bound above upper bound")
      ranges[name] = (lower, upper)
    return ranges

  def _apply_level(self) -> None:
    level = min(max(self._level, 0.0), 1.0)
    ranges = self._command_term.cfg.ranges
    for name in self._RANGE_NAMES:
      initial = self._initial_ranges[name]
      maximum = self._maximum_ranges[name]
      current = (
        initial[0] + level * (maximum[0] - initial[0]),
        initial[1] + level * (maximum[1] - initial[1]),
      )
      setattr(ranges, name, current)

  def _collect_completed_episodes(
    self, env_ids: torch.Tensor | slice | None
  ) -> tuple[float, int]:
    if env_ids is None:
      env_ids = slice(None)

    metrics = self._command_term.metrics.get("error_vel_xy")
    if metrics is None:
      raise KeyError(
        f"Command term {self._command_name!r} does not expose error_vel_xy"
      )

    episode_lengths = self._env.episode_length_buf[env_ids]
    episode_errors = metrics[env_ids]
    valid = episode_lengths > 0
    if not torch.any(valid):
      return 0.0, 0

    max_command_steps = (
      self._command_term.cfg.resampling_time_range[1] / self._env.step_dt
    )
    if max_command_steps <= 0.0:
      raise ValueError("resampling_time_range must have a positive upper bound")

    # UniformVelocityCommand stores sum(error / max_command_steps). Restore the
    # episode mean so early terminations are comparable to full episodes.
    mean_errors = (
      episode_errors[valid]
      * max_command_steps
      / episode_lengths[valid].to(dtype=episode_errors.dtype)
    )
    return float(mean_errors.sum().item()), int(valid.sum().item())

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    **_kwargs: object,
  ) -> dict[str, float]:
    del env
    error_sum, episode_count = self._collect_completed_episodes(env_ids)
    self._window_error_sum += error_sum
    self._window_episode_count += episode_count

    updated = 0.0
    if self._window_episode_count >= self._update_every_episodes:
      self._last_mean_error = self._window_error_sum / self._window_episode_count
      if self._last_mean_error < self._error_up_threshold:
        self._level = min(1.0, self._level + self._level_step)
      elif self._last_mean_error > self._error_down_threshold:
        self._level = max(0.0, self._level - self._level_step)
      self._last_update_episode_count = self._window_episode_count
      self._window_error_sum = 0.0
      self._window_episode_count = 0
      self._apply_level()
      updated = 1.0

    return {
      "level": self._level,
      "last_mean_error": self._last_mean_error,
      "episodes_since_update": float(self._window_episode_count),
      "last_update_episodes": float(self._last_update_episode_count),
      "updated": updated,
    }


class StratifiedVelocityCurriculum(AdaptiveVelocityCurriculum):
  """Range adaptation with persistent basic-skill sampling and separate scoring.

  Only completed-episode samples of the combined task advance the range.
  Standing and pure yaw must not lower the range curriculum's planar error.
  All modes retain their own measured exposure and tracking errors for logging.
  """

  def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
    super().__init__(cfg, env)
    from .stratified_velocity_command import StratifiedVelocityCommand

    if not isinstance(self._command_term, StratifiedVelocityCommand):
      raise TypeError("StratifiedVelocityCurriculum requires StratifiedVelocityCommandCfg")
    self._command_term.configure_sampling(
      cfg.params["mode_probabilities"],
      cfg.params["linear_speed_band"],
      cfg.params["yaw_speed_band"],
    )

  def _collect_completed_episodes(self, env_ids):
    ids = _resolve_env_ids(self._env, env_ids)
    index = STRATIFIED_COMMAND_MODES.index("combined")
    steps = self._command_term.task_steps[ids, index]
    valid = (steps > 0) & (self._env.episode_length_buf[ids] > 0)
    if not torch.any(valid):
      return 0.0, 0
    means = self._command_term.task_xy_error_sum[ids, index][valid] / steps[valid]
    return float(means.sum().item()), int(valid.sum().item())

  def __call__(self, env, env_ids, **kwargs):
    state = super().__call__(env, env_ids, **kwargs)
    ids = _resolve_env_ids(env, env_ids)
    state.update(self._command_term.sampling_metrics(ids))
    for name in self._RANGE_NAMES:
      lo, hi = getattr(self._command_term.cfg.ranges, name)
      state[f"{name}_min"], state[f"{name}_max"] = lo, hi
    return state


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])
