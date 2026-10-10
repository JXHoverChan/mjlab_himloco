"""Shared semantics for explicitly commanded standing and motion."""

import torch


def is_standing_command(command: torch.Tensor) -> torch.Tensor:
  """Only an all-zero command requests standing; small nonzero commands move.

  The standing sampler and joystick deadzone already produce exact zeros. This
  predicate does not introduce another speed deadzone inside rewards/observations.
  """
  return (command == 0).all(dim=-1)
