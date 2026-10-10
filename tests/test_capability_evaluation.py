from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluate_policy_capabilities import (
  CapabilityConfig, ScheduledVelocityCommand, bootstrap_cluster_interval, make_cases, ranges_at_level,
)


def test_commands_switch_before_next_action_and_scoped_reset_preserves_other_envs():
  term = object.__new__(ScheduledVelocityCommand)
  term.schedule = torch.tensor([[[1., 0., 0.], [2., 0., 0.]]] * 3)
  term.vel_command_b = torch.zeros(3, 3)
  term._env = SimpleNamespace(num_envs=3, device="cpu", episode_length_buf=torch.tensor([24, 25, 275]))
  term.settle_steps, term.stage_steps = 25, 250
  term.use_heading = False
  term.active = torch.ones(3, dtype=torch.bool)
  term._update_command()
  assert torch.equal(term.vel_command_b[:, 0], torch.tensor([0., 1., 2.]))
  term._env.episode_length_buf[1] = 0
  term._update_command(torch.tensor([1]))
  assert torch.equal(term.vel_command_b[:, 0], torch.tensor([0., 0., 2.]))


def test_bootstrap_treats_repeated_resets_as_one_physical_environment():
  original = bootstrap_cluster_interval([0., 1.], [0, 1])
  duplicated = bootstrap_cluster_interval([0., 0., 1., 1.], [0, 0, 1, 1])
  assert original == duplicated


def test_velocity_profiles_match_the_actual_curriculum_interpolation():
  assert np.allclose(ranges_at_level(0.55), [[-1.05, 1.825], [-1.05, 1.05], [-1.05, 1.05]])
  assert np.allclose(ranges_at_level(1.), [[-1.5, 2.5], [-1.5, 1.5], [-1.5, 1.5]])


def test_maneuver_evaluation_includes_both_directions_of_low_yaw():
  commands = {case["command"] for case in make_cases("maneuver", CapabilityConfig())}
  assert all((0., 0., yaw) in commands for yaw in (-.2, -.1, -.05, .05, .1, .2))
