import torch

from src.tasks.velocity.mdp.curriculums import _classify_terrain_update


def test_classify_terrain_update_separates_metrics() -> None:
  distance = torch.tensor([5.0, 5.0, 2.0, 5.0, 4.5, 5.0, 10.0])
  expected_distance = torch.full_like(distance, 4.0)
  linear_error = torch.tensor([0.1, 0.1, 0.1, 0.6, 0.4, 0.1, 0.1])
  yaw_error = torch.tensor([0.1, 0.4, 0.1, 0.1, 0.25, 0.1, 0.1])
  active = torch.tensor([True, True, True, True, True, True, False])

  move_up, move_down = _classify_terrain_update(
    distance=distance,
    expected_distance=expected_distance,
    linear_error=linear_error,
    yaw_error=yaw_error,
    active=active,
    terrain_up_distance=4.0,
    linear_error_up_threshold=0.3,
    linear_error_down_threshold=0.55,
    yaw_error_up_threshold=0.2,
    yaw_error_down_threshold=0.35,
  )

  assert torch.equal(move_up, torch.tensor([True, False, False, False, False, True, False]))
  assert torch.equal(move_down, torch.tensor([False, True, True, True, False, False, False]))
