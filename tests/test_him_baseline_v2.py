import pytest
import torch
from types import SimpleNamespace

from src.tasks.velocity.config.go2.env_cfgs import HIMloco_Go2_env_cfg
from src.tasks.velocity.mdp.terminations import out_of_terrain_bounds


@pytest.mark.parametrize("play", [False, True])
def test_him_truncates_terrain_exit_without_restoring_nonfoot_termination(play):
  cfg = HIMloco_Go2_env_cfg(play=play)
  assert cfg.terminations["terrain_bounds"].time_out
  assert "nonfoot_ground_2" not in cfg.terminations
  assert "illegal_contact_1" in cfg.terminations


def test_boundary_uses_actual_grid_and_catches_the_captured_freefall_position():
  # The captured MJB has a 10x7 grid, not the nominal cfg.num_cols=20.
  positions = torch.tensor([
    [0., 0., .32], [59., 47., .32], [59.8, 0., .32],
    [0., 47.8, .32], [0., -47.8, .32], [12.37, 66.70, -267.54],
  ])
  terrain = SimpleNamespace(
    cfg=SimpleNamespace(terrain_type="generator", terrain_generator=SimpleNamespace(
      size=(8., 8.), border_width=20., num_cols=20,
    )),
    terrain_origins=torch.zeros(10, 7, 3),
  )
  class Scene(dict):
    pass
  scene = Scene(robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=positions)))
  scene.terrain = terrain
  env = SimpleNamespace(scene=scene, num_envs=len(positions), device="cpu")
  assert torch.equal(out_of_terrain_bounds(env), torch.tensor([False, False, True, True, True, True]))
  terrain.cfg.terrain_type = "plane"
  assert not out_of_terrain_bounds(env).any()


@pytest.mark.parametrize("group", ["actor", "critic", "target_encoder"])
def test_nonzero_low_yaw_and_translation_keep_the_motion_phase(group):
  command = torch.tensor([[0., 0., 0.], [0., 0., .05], [0., 0., -.05], [.05, 0., 0.]])
  env = SimpleNamespace(
    num_envs=4, device="cpu", step_dt=.02, episode_length_buf=torch.full((4,), 5),
    command_manager=SimpleNamespace(get_command=lambda _: command),
  )
  term = HIMloco_Go2_env_cfg().observations[group].terms["phase"]
  value = term.func(env, **term.params)
  assert torch.equal(value[0], torch.zeros(2))
  assert torch.allclose(torch.linalg.vector_norm(value[1:], dim=-1), torch.ones(3))


def make_reward_env(command):
  joint_names = tuple(f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
                      for joint in ("hip", "thigh", "calf"))
  robot = SimpleNamespace(
    data=SimpleNamespace(
      default_joint_pos=torch.zeros(len(command), 12),
      joint_pos=torch.full((len(command), 12), .1),
      root_link_lin_vel_b=torch.zeros(len(command), 3),
      root_link_ang_vel_b=torch.zeros(len(command), 3),
    ),
    find_joints=lambda _: (list(range(12)), joint_names),
  )
  return SimpleNamespace(device="cpu", scene={"robot": robot}, command_manager=SimpleNamespace(
    get_command=lambda _: command,
  ))


@pytest.mark.parametrize("term_name", ["pose", "stand_still"])
def test_standing_objectives_do_not_compete_with_small_nonzero_commands(term_name):
  command = torch.tensor([[0., 0., 0.], [0., 0., .05], [0., 0., -.05], [.05, 0., 0.]])
  env = make_reward_env(command)
  term = HIMloco_Go2_env_cfg().rewards[term_name]
  func = term.func(term, env) if isinstance(term.func, type) else term.func
  value = func(env, **term.params)
  assert value[0] > 0
  assert torch.equal(value[1:], torch.zeros(3))


def test_tracking_precision_has_independent_forward_lateral_and_yaw_scales():
  command = torch.tensor([[.15, 0., 0.], [0., .15, 0.], [0., -.15, 0.], [0., 0., .2], [0., 0., -.2]])
  env = make_reward_env(command)
  cfg = HIMloco_Go2_env_cfg()
  linear, angular = cfg.rewards["track_linear_velocity"], cfg.rewards["track_angular_velocity"]
  xy_reward = linear.func(env, **linear.params)
  yaw_reward = angular.func(env, **angular.params)
  assert torch.allclose(xy_reward, torch.exp(-torch.tensor([.09, 1., 1., 0., 0.])))
  assert torch.allclose(yaw_reward, torch.exp(-torch.tensor([0., 0., 0., 1., 1.])))
  # Neither tracking term includes vertical velocity or roll/pitch stabilization.
  env.scene["robot"].data.root_link_lin_vel_b[:, :2] = command[:, :2]
  env.scene["robot"].data.root_link_lin_vel_b[:, 2] = 1.
  env.scene["robot"].data.root_link_ang_vel_b[:, 2] = command[:, 2]
  env.scene["robot"].data.root_link_ang_vel_b[:, :2] = 1.
  assert torch.equal(linear.func(env, **linear.params), torch.ones(5))
  assert torch.equal(angular.func(env, **angular.params), torch.ones(5))
