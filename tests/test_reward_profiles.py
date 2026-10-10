from types import SimpleNamespace
import math

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from src.tasks.velocity.config.go2.env_cfgs import HIMloco_Go2_env_cfg
from src.tasks.velocity.mdp import rewards
from src.tasks.velocity.mdp.reward_profiles import apply_him_reward_profile


def test_tracking_and_stability_are_independent() -> None:
  command = torch.tensor([[1., 0., 0.5], [1., 0., 0.5]])
  robot = SimpleNamespace(data=SimpleNamespace(
    root_link_lin_vel_b=torch.tensor([[1., 0., 0.], [1., 0., 1.]]),
    root_link_ang_vel_b=torch.tensor([[0., 0., 0.5], [1., 2., 0.5]]),
  ))
  env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(
    get_command=lambda _: command,
  ))
  assert torch.equal(rewards.track_planar_velocity(env, 0.5, "twist"), torch.ones(2))
  assert torch.equal(rewards.track_yaw_velocity(env, 0.5, "twist"), torch.ones(2))
  assert torch.equal(rewards.vertical_velocity_l2(env), torch.tensor([0., 1.]))
  assert torch.equal(rewards.roll_pitch_velocity_l2(env), torch.tensor([0., 5.]))
  assert rewards.track_linear_velocity(env, 0.5, "twist")[1] < 1.0
  assert rewards.track_angular_velocity(env, 0.5, "twist")[1] < 1.0


def test_pose_is_rewarded_only_at_rest_including_pure_yaw_commands() -> None:
  command = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 0., 1.]])
  robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(3, 1)))
  env = SimpleNamespace(scene={"robot": robot}, command_manager=SimpleNamespace(
    get_command=lambda _: command,
  ))
  pose = object.__new__(rewards.standing_posture)
  pose.default_joint_pos = torch.zeros(3, 1)
  pose.std_standing = pose.std_walking = pose.std_running = torch.ones(1)
  result = pose(env, {}, {}, {}, SceneEntityCfg("robot"), "twist", 0.1, 1.5)
  assert torch.equal(result, torch.tensor([1., 0., 0.]))


def test_profiles_restore_legacy_and_do_not_change_other_task_semantics() -> None:
  cfg = HIMloco_Go2_env_cfg()
  observations, commands, terminations = cfg.observations, cfg.commands, cfg.terminations
  apply_him_reward_profile(cfg, "legacy")
  assert cfg.rewards["pose"].func is rewards.variable_posture
  assert cfg.rewards["action_rate_l2"].weight == -0.1
  assert "vertical_velocity_l2" not in cfg.rewards
  apply_him_reward_profile(cfg, "standing_pose")
  assert cfg.rewards["pose"].func is rewards.standing_posture
  assert cfg.rewards["track_linear_velocity"].func is rewards.track_linear_velocity
  apply_him_reward_profile(cfg, "revised")
  assert cfg.rewards["track_linear_velocity"].func is rewards.track_planar_velocity
  assert cfg.rewards["angular_momentum"].weight == 0
  assert cfg.rewards["action_rate_l2"].weight == -0.01
  apply_him_reward_profile(cfg, "legacy")
  assert cfg.rewards["body_ang_vel"].func is rewards.body_angular_velocity_penalty
  assert cfg.rewards["body_orientation_l2"].weight == -1
  assert "vertical_velocity_l2" not in cfg.rewards
  assert cfg.observations is observations
  assert cfg.commands is commands
  assert cfg.terminations is terminations


def test_baseline_v2_round_trip_restores_old_tracking_parameters() -> None:
  cfg = HIMloco_Go2_env_cfg()
  for old_profile in ("revised", "legacy", "revised_strong_action"):
    apply_him_reward_profile(cfg, "baseline_v2")
    assert cfg.rewards["pose"].func is rewards.commanded_standing_posture
    linear = cfg.rewards["track_linear_velocity"]
    assert linear.func is rewards.track_planar_velocity_anisotropic
    assert linear.params["std_x"] == .5 and linear.params["std_y"] == .15
    assert "std" not in linear.params
    assert cfg.rewards["track_angular_velocity"].params["std"] == .2
    assert "command_threshold" not in cfg.rewards["stand_still"].params
    apply_him_reward_profile(cfg, old_profile)
    assert linear.params["std"] == .5
    assert "std_x" not in linear.params and "std_y" not in linear.params
    assert cfg.rewards["track_angular_velocity"].params["std"] == math.sqrt(.5)
    assert cfg.rewards["stand_still"].func is rewards.stand_still
    assert cfg.rewards["stand_still"].params["command_threshold"] == .1
