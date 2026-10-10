"""Explicit physical surfaces; no center platform in the tracking terrain bank."""

from __future__ import annotations

from dataclasses import dataclass
import math

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

from .protocol import PATCH_SIZE, TerrainSpec


GRID_SPACING = .10


def height_grid(terrain: TerrainSpec, size=PATCH_SIZE) -> np.ndarray:
  """Physical elevations on a row-Y/column-X grid, before MuJoCo normalization."""
  coordinates = np.linspace(-size / 2, size / 2, round(size / GRID_SPACING) + 1)
  x, _ = np.meshgrid(coordinates, coordinates, indexing="xy")
  result = math.tan(math.radians(terrain.slope_degrees)) * x
  rng = np.random.default_rng(terrain.seed)
  if terrain.kind == "rough_slope":
    coarse_coordinates = np.linspace(-size / 2, size / 2, round(size / .4) + 1)
    anchors = rng.uniform(-terrain.noise_amplitude, terrain.noise_amplitude,
                          (len(coarse_coordinates), len(coarse_coordinates)))
    rows = np.array([np.interp(coordinates, coarse_coordinates, row) for row in anchors])
    noise = np.array([np.interp(coordinates, coarse_coordinates, col) for col in rows.T]).T
    result += noise
  elif terrain.kind == "obstacles":
    # Fixed 0.6 m square blocks, with bumps and pits. No erased spawn platform.
    cells = round(size / .6) + 2
    blocks = rng.choice([-1., -.5, 0., .5, 1.], size=(cells, cells),
                        p=[.10, .10, .60, .10, .10]) * terrain.height
    index = np.minimum(((coordinates + size / 2) / .6).astype(int), cells - 1)
    result = blocks[index[:, None], index[None, :]]
  elif terrain.kind == "stairs":
    result = np.floor((x + terrain.step_width / 2) / terrain.step_width) * terrain.height
  elif terrain.kind not in ("flat", "slope"):
    raise ValueError(f"Unsupported terrain kind: {terrain.kind}")
  return np.asarray(result, dtype=np.float64)


def surface_sample(grid: np.ndarray, x: float, y: float, size=PATCH_SIZE) -> float:
  """Bilinear height for spawn planning and terrain-exposure diagnostics."""
  px = (x / size + .5) * (grid.shape[1] - 1)
  py = (y / size + .5) * (grid.shape[0] - 1)
  if not (0 <= px <= grid.shape[1] - 1 and 0 <= py <= grid.shape[0] - 1):
    raise ValueError("Requested point lies outside the terrain")
  ix, iy = int(px), int(py)
  jx, jy = min(ix + 1, grid.shape[1] - 1), min(iy + 1, grid.shape[0] - 1)
  a, b = px - ix, py - iy
  return float((1-b) * ((1-a)*grid[iy, ix] + a*grid[iy, jx])
               + b * ((1-a)*grid[jy, ix] + a*grid[jy, jx]))


@dataclass(kw_only=True)
class BenchmarkTerrainCfg(SubTerrainCfg):
  terrain: TerrainSpec

  def function(self, difficulty, spec, rng):
    del difficulty, rng
    body = spec.body("terrain")
    half_x, half_y = self.size[0] / 2, self.size[1] / 2
    grid = height_grid(self.terrain, self.size[0])
    geometries = []
    if self.terrain.kind in ("flat", "stairs"):
      width = self.size[0] if self.terrain.kind == "flat" else self.terrain.step_width
      bottom = float(grid.min()) - .3
      indices = (range(math.floor(-half_x/width)-1, math.ceil(half_x/width)+1)
                 if self.terrain.kind == "stairs" else (0,))
      for index in indices:
        left = max((index-.5)*width, -half_x) if self.terrain.kind == "stairs" else -half_x
        right = min((index+.5)*width, half_x) if self.terrain.kind == "stairs" else half_x
        if right <= left:
          continue
        center = (left + right) / 2
        height = index * self.terrain.height if self.terrain.kind == "stairs" else 0.
        geom = body.add_geom(
          type=mujoco.mjtGeom.mjGEOM_BOX,
          size=[(right-left)/2, half_y, (height-bottom)/2],
          pos=[half_x+center, half_y, (height+bottom)/2],
        )
        geometries.append(TerrainGeometry(geom=geom, color=(.5, .55, .5, 1.)))
    else:
      minimum, maximum = float(grid.min()), float(grid.max())
      span = maximum - minimum
      if span <= 0:
        raise ValueError("Nonflat terrain must have a nonzero elevation range")
      field = spec.add_hfield(
        name=f"blind_benchmark_{self.terrain.name}",
        size=[half_x, half_y, span, .3], nrow=grid.shape[0], ncol=grid.shape[1],
        userdata=((grid-minimum)/span).astype(np.float32).ravel().tolist(),
      )
      geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                           pos=[half_x, half_y, minimum])
      geometries.append(TerrainGeometry(geom=geom, hfield=field, color=(.5, .55, .5, 1.)))
    # Maximum height under the initial stance footprint prevents initial penetration.
    # It is scenario construction, not a runtime policy correction.
    middle = grid.shape[0] // 2
    radius = round(.35 / GRID_SPACING) + 1
    spawn_height = float(grid[middle-radius:middle+radius+1,
                              middle-radius:middle+radius+1].max())
    if self.terrain.kind in ("slope", "rough_slope"):
      slope = math.tan(math.radians(self.terrain.slope_degrees))
      support = grid[middle-radius:middle+radius+1, middle-radius:middle+radius+1]
      local_x = np.linspace(-radius*GRID_SPACING, radius*GRID_SPACING, 2*radius+1)
      spawn_height = float((support - slope*local_x[None, :]).max())
    return TerrainOutput(origin=np.array([half_x, half_y, spawn_height]), geometries=geometries)
