"""Rollout storage extensions required by the HIM auxiliary objective."""

from __future__ import annotations

from collections.abc import Generator

import torch

from rsl_rl.storage import RolloutStorage


class HIMRolloutStorage(RolloutStorage):
  """Standard PPO storage plus aligned target and velocity observations."""

  class Transition(RolloutStorage.Transition):
    def __init__(self) -> None:
      super().__init__()
      self.next_target_observation: torch.Tensor | None = None
      self.velocity_target: torch.Tensor | None = None

    def clear(self) -> None:
      self.__init__()

  def __init__(
    self,
    training_type: str,
    num_envs: int,
    num_transitions_per_env: int,
    obs,
    actions_shape: tuple[int, ...] | list[int],
    target_obs_group: str,
    velocity_obs_group: str,
    device: str = "cpu",
  ) -> None:
    super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

    if target_obs_group not in obs:
      raise KeyError(f"Target observation group '{target_obs_group}' is not present in {list(obs.keys())}")
    if velocity_obs_group not in obs:
      raise KeyError(f"Velocity observation group '{velocity_obs_group}' is not present in {list(obs.keys())}")

    target = obs[target_obs_group]
    velocity = obs[velocity_obs_group]
    self.next_target_observations = torch.zeros(
      num_transitions_per_env,
      *target.shape,
      dtype=target.dtype,
      device=device,
    )
    self.velocity_targets = torch.zeros(
      num_transitions_per_env,
      *velocity.shape,
      dtype=velocity.dtype,
      device=device,
    )

  def add_transition(self, transition: Transition) -> None:
    super().add_transition(transition)
    assert transition.next_target_observation is not None
    assert transition.velocity_target is not None
    self.next_target_observations[self.step - 1].copy_(transition.next_target_observation)
    self.velocity_targets[self.step - 1].copy_(transition.velocity_target)

  def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator[RolloutStorage.Batch, None, None]:
    if self.training_type != "rl":
      raise ValueError("This function is only available for reinforcement learning training.")

    batch_size = self.num_envs * self.num_transitions_per_env
    mini_batch_size = batch_size // num_mini_batches
    indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

    observations = self.observations.flatten(0, 1)
    next_target_observations = self.next_target_observations.flatten(0, 1)
    velocity_targets = self.velocity_targets.flatten(0, 1)
    dones = self.dones.flatten(0, 1)
    actions = self.actions.flatten(0, 1)
    values = self.values.flatten(0, 1)
    returns = self.returns.flatten(0, 1)
    old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
    advantages = self.advantages.flatten(0, 1)
    old_distribution_params = tuple(p.flatten(0, 1) for p in self.distribution_params)  # type: ignore

    for _ in range(num_epochs):
      for i in range(num_mini_batches):
        start = i * mini_batch_size
        stop = (i + 1) * mini_batch_size
        batch_idx = indices[start:stop]
        batch = RolloutStorage.Batch(
          observations=observations[batch_idx],
          actions=actions[batch_idx],
          values=values[batch_idx],
          advantages=advantages[batch_idx],
          returns=returns[batch_idx],
          old_actions_log_prob=old_actions_log_prob[batch_idx],
          old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
          dones=dones[batch_idx],
        )
        batch.next_target_observations = next_target_observations[batch_idx]
        batch.velocity_targets = velocity_targets[batch_idx]
        yield batch
