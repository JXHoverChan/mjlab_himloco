"""Explicit HIM reward variants for training and controlled ablations.

Profiles change rewards only. They never alter observations, actuators,
termination conditions, terrain, or command sampling.
"""

import math
from typing import Literal

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from . import rewards

RewardProfile = Literal[
  "legacy", "standing_pose", "split_tracking", "revised", "revised_strong_action", "baseline_v2"
]


def apply_him_reward_profile(cfg: ManagerBasedRlEnvCfg, profile: RewardProfile) -> None:
  """Apply a complete, repeatable profile, including restoration of old weights."""
  required = (
    "pose", "track_linear_velocity", "track_angular_velocity", "body_ang_vel",
    "angular_momentum", "body_orientation_l2", "action_rate_l2", "stand_still",
  )
  if any(name not in cfg.rewards for name in required):
    raise ValueError("HIM reward profiles require the Go2 velocity reward configuration.")
  if profile not in ("legacy", "standing_pose", "split_tracking", "revised", "revised_strong_action", "baseline_v2"):
    raise ValueError(f"Unknown HIM reward profile: {profile}")

  # Restore the pre-revision definitions before applying the selected variant.
  cfg.rewards["pose"].func = rewards.variable_posture
  cfg.rewards["track_linear_velocity"].func = rewards.track_linear_velocity
  cfg.rewards["track_angular_velocity"].func = rewards.track_angular_velocity
  linear_params = cfg.rewards["track_linear_velocity"].params
  linear_params.pop("std_x", None)
  linear_params.pop("std_y", None)
  linear_params["std"] = 0.5
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.5)
  cfg.rewards["body_ang_vel"].func = rewards.body_angular_velocity_penalty
  cfg.rewards["stand_still"].func = rewards.stand_still
  cfg.rewards["stand_still"].params["command_threshold"] = 0.1
  cfg.rewards["body_orientation_l2"].weight = -1.0
  cfg.rewards["angular_momentum"].weight = -0.025
  cfg.rewards["action_rate_l2"].weight = -0.1
  cfg.rewards.pop("vertical_velocity_l2", None)

  if profile in ("standing_pose", "revised", "revised_strong_action", "baseline_v2"):
    cfg.rewards["pose"].func = rewards.standing_posture
  if profile in ("split_tracking", "revised", "revised_strong_action", "baseline_v2"):
    cfg.rewards["track_linear_velocity"].func = rewards.track_planar_velocity
    cfg.rewards["track_angular_velocity"].func = rewards.track_yaw_velocity
    cfg.rewards["body_ang_vel"].func = rewards.roll_pitch_velocity_l2
    cfg.rewards["vertical_velocity_l2"] = RewardTermCfg(
      func=rewards.vertical_velocity_l2, weight=-0.2,
    )
  if profile in ("revised", "revised_strong_action", "baseline_v2"):
    cfg.rewards["body_orientation_l2"].weight = -0.2
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["action_rate_l2"].weight = -0.01
  if profile == "revised_strong_action":
    cfg.rewards["action_rate_l2"].weight = -0.1
  if profile == "baseline_v2":
    cfg.rewards["pose"].func = rewards.commanded_standing_posture
    cfg.rewards["stand_still"].func = rewards.stand_still_on_zero_command
    cfg.rewards["stand_still"].params.pop("command_threshold")
    cfg.rewards["track_linear_velocity"].func = rewards.track_planar_velocity_anisotropic
    linear_params.pop("std")
    linear_params.update(std_x=0.5, std_y=0.15)
    cfg.rewards["track_angular_velocity"].params["std"] = 0.2
