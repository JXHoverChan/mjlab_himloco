from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.sensor import ContactSensor
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def out_of_terrain_bounds(
  env: ManagerBasedRlEnv,
  margin: float = 0.3,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Truncate at the finite generated terrain edge, including its flat border.

  Use the generated grid shape: curriculum mode creates one column per terrain
  type rather than ``cfg.num_cols``. Infinite planes have no boundary.
  """
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_type != "generator":
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
  generator = terrain.cfg.terrain_generator
  assert generator is not None and terrain.terrain_origins is not None
  rows, cols = terrain.terrain_origins.shape[:2]
  half_x = rows * generator.size[0] / 2 + generator.border_width
  half_y = cols * generator.size[1] / 2 + generator.border_width
  if not 0 <= margin < min(half_x, half_y):
    raise ValueError("terrain boundary margin must be nonnegative and inside the footprint")
  position = env.scene[asset_cfg.name].data.root_link_pos_w
  return (position[:, 0].abs() > half_x - margin) | (
    position[:, 1].abs() > half_y - margin
  )


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)
