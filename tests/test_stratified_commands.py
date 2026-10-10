from types import SimpleNamespace

import pytest
import torch

from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from src.tasks.velocity.config.go2.env_cfgs import (
  HIMloco_Go2_env_cfg, unitree_go2_rough_env_cfg,
)
from src.tasks.velocity.mdp.curriculums import (
  DEFAULT_COMMAND_PROBABILITIES, STRATIFIED_COMMAND_MODES,
  StratifiedVelocityCurriculum, sample_stratified_velocity_commands,
)
from src.tasks.velocity.mdp.stratified_velocity_command import (
  StratifiedVelocityCommand, StratifiedVelocityCommandCfg,
)


def make_term(n=64):
  cfg = StratifiedVelocityCommandCfg(
    entity_name="robot", resampling_time_range=(3., 8.), heading_command=True,
    rel_heading_envs=1.,
    ranges=UniformVelocityCommandCfg.Ranges((-.5, 1.), (-.5, .5), (-.5, .5), (-3.14, 3.14)),
  )
  data = SimpleNamespace(root_link_lin_vel_b=torch.zeros(n, 3),
                         root_link_ang_vel_b=torch.zeros(n, 3), heading_w=torch.zeros(n))
  env = SimpleNamespace(num_envs=n, device="cpu", step_dt=.02,
                        scene={"robot": SimpleNamespace(data=data)},
                        episode_length_buf=torch.full((n,), 500))
  term = StratifiedVelocityCommand(cfg, env)
  env.command_manager = SimpleNamespace(get_term=lambda _: term)
  return term, env


def test_sampling_preserves_task_constraints_and_coverage_at_both_range_limits():
  probabilities = torch.tensor(list(DEFAULT_COMMAND_PROBABILITIES.values()))
  for ranges in [UniformVelocityCommandCfg.Ranges((-.5, 1.), (-.5, .5), (-.5, .5)),
                 UniformVelocityCommandCfg.Ranges((-1.5, 2.5), (-1.5, 1.5), (-1.5, 1.5))]:
    torch.manual_seed(17)
    commands, modes = sample_stratified_velocity_commands(ranges, probabilities, 10000, (.2, .8), (.3, 1.))
    for axis, bounds in enumerate([ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z]):
      assert (commands[:, axis] >= bounds[0]).all() and (commands[:, axis] <= bounds[1]).all()
    fractions = torch.bincount(modes, minlength=7) / len(modes)
    assert torch.allclose(fractions, probabilities, atol=.02)
    for mode, name in enumerate(STRATIFIED_COMMAND_MODES):
      values = commands[modes == mode]
      if name == "standing":
        assert torch.equal(values, torch.zeros_like(values))
      elif name in ("low_forward", "low_backward"):
        assert (values[:, 1:] == 0).all()
        assert (values[:, 0].abs() >= .2).all() and (values[:, 0].abs() <= .8).all()
        assert (values[:, 0] > 0).all() if name == "low_forward" else (values[:, 0] < 0).all()
      elif name in ("lateral", "pure_yaw"):
        axis = 1 if name == "lateral" else 2
        assert (values[:, [j for j in range(3) if j != axis]] == 0).all()
        positive_fraction = (values[:, axis] > 0).float().mean()
        assert .45 < positive_fraction < .55
      elif name == "translation":
        assert (values[:, 2] == 0).all()
        assert ((values[:, :2] != 0).sum(1) == 1).all()


def test_timer_resampling_keeps_pure_yaw_and_low_translation_out_of_heading_control():
  term, _ = make_term(512)
  torch.manual_seed(31)
  ids = torch.arange(512)
  term.reset(ids)
  term.compute(0., ids)
  assert term.task_steps.sum() == 0
  first = term.vel_command_b.clone()
  yaw_ids = term.mode == STRATIFIED_COMMAND_MODES.index("pure_yaw")
  assert yaw_ids.any() and (~term.is_heading_env[yaw_ids]).all()
  term.robot.data.heading_w[:] = 2.
  term.compute(.02)
  assert torch.equal(term.vel_command_b[yaw_ids], first[yaw_ids])
  low = (term.mode == 1) | (term.mode == 2)
  assert (term.vel_command_b[low, 1:] == 0).all()
  term.time_left[:] = 0.
  term.compute(.02)
  assert (term.command_counter == 2).all()
  assert not torch.equal(term.vel_command_b, first)
  assert term.task_steps.sum() == 1024
  saved = term.task_steps[1:].clone()
  term.reset(torch.tensor([0]))
  assert term.task_steps[0].sum() == 0
  assert torch.equal(term.task_steps[1:], saved)


def test_range_promotion_scores_combined_samples_without_standing_or_yaw_dilution():
  term, env = make_term(4)
  params = HIMloco_Go2_env_cfg().curriculum["command_vel"].params.copy()
  params["update_every_episodes"] = 2
  curriculum = StratifiedVelocityCurriculum(CurriculumTermCfg(func=StratifiedVelocityCurriculum, params=params), env)
  term.task_steps[0, 0] = 400
  term.task_steps[1, 4] = 400
  term.task_steps[2:, 6] = 100
  term.task_xy_error_sum[2:, 6] = torch.tensor([80., 90.])
  state = curriculum(env, torch.arange(4))
  assert state["updated"] == 1.
  assert state["last_update_episodes"] == 2.
  assert abs(state["last_mean_error"] - .85) < 1e-6
  assert state["level"] == 0.
  assert state["samples_combined"] == 200.
  assert abs(state["fraction_pure_yaw"] - .4) < 1e-6
  term.task_xy_error_sum[2:, 6] = 20.
  state = curriculum(env, torch.tensor([2, 3]))
  assert state["level"] == .025
  assert state["lin_vel_x_max"] > 1.


def test_configuration_changes_only_him_training_command_route():
  train, play, baseline = HIMloco_Go2_env_cfg(), HIMloco_Go2_env_cfg(play=True), unitree_go2_rough_env_cfg()
  assert isinstance(train.commands["twist"], StratifiedVelocityCommandCfg)
  assert train.curriculum["command_vel"].func is StratifiedVelocityCurriculum
  assert type(play.commands["twist"]) is UniformVelocityCommandCfg
  assert type(baseline.commands["twist"]) is UniformVelocityCommandCfg
  assert train.rewards["action_rate_l2"].weight == -.01
  assert train.commands["twist"].init_velocity_prob == 0.


def test_invalid_mixture_or_unavailable_speed_band_is_rejected():
  term, _ = make_term()
  invalid = DEFAULT_COMMAND_PROBABILITIES.copy()
  invalid["standing"] = .5
  with pytest.raises(ValueError, match="sum to one"):
    term.configure_sampling(invalid, (.2, .8), (.3, 1.))
  with pytest.raises(ValueError, match="minimum speed"):
    term.configure_sampling(DEFAULT_COMMAND_PROBABILITIES, (.6, .8), (.3, 1.))


def test_him_curriculum_covers_signed_low_yaw_without_heading_override():
  term, env = make_term(4096)
  params = HIMloco_Go2_env_cfg().curriculum["command_vel"].params.copy()
  assert params["yaw_speed_band"] == (.05, 1.)
  StratifiedVelocityCurriculum(CurriculumTermCfg(func=StratifiedVelocityCurriculum, params=params), env)
  torch.manual_seed(43)
  ids = torch.arange(env.num_envs)
  term.reset(ids)
  term.compute(0., ids)
  yaw = term.mode == STRATIFIED_COMMAND_MODES.index("pure_yaw")
  values = term.command[yaw].clone()
  assert (values[:, :2] == 0).all()
  assert (values[:, 2].abs() >= .05).all()
  assert (values[:, 2].abs() <= .5).all()  # Initial range still clips the upper band.
  assert ((values[:, 2] >= .05) & (values[:, 2] < .1)).any()
  assert ((values[:, 2] <= -.05) & (values[:, 2] > -.1)).any()
  term.robot.data.heading_w[:] = 2.
  term.compute(.02)
  assert torch.equal(term.command[yaw], values)
  standing = term.mode == STRATIFIED_COMMAND_MODES.index("standing")
  assert standing.any() and (term.command[standing] == 0).all()


@pytest.mark.parametrize("minimum", [0., -.05, float("nan")])
def test_low_yaw_band_still_requires_a_finite_positive_minimum(minimum):
  term, _ = make_term()
  with pytest.raises(ValueError, match="minimum"):
    term.configure_sampling(DEFAULT_COMMAND_PROBABILITIES, (.2, .8), (minimum, 1.))
