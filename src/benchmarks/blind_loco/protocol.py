"""Immutable task definitions and reference schedules for our internal benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np


PROTOCOL_VERSION = "Go2-BlindLoco-v1.0"
STEP_DT = 0.02
OBS_DIM = 47
PATCH_SIZE = 40.0
SETTLE_SECONDS = 1.0
XY_TOLERANCE = 0.20
YAW_TOLERANCE = 0.10
HOLD_SECONDS = 0.40
TILT_LIMIT_DEGREES = 70.0
ROUTE_LENGTH = 3.0
ROUTE_HALF_WIDTH = 0.75


def digest(value) -> str:
  payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
  return hashlib.sha256(payload.encode()).hexdigest()


def stable_seed(*parts) -> int:
  return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest()[:8], 16)


@dataclass(frozen=True)
class TerrainSpec:
  name: str
  kind: str
  slope_degrees: float = 0.0
  height: float = 0.0
  step_width: float = 0.30
  noise_amplitude: float = 0.0
  seed: int = 0


@dataclass(frozen=True)
class Case:
  name: str
  suite: str
  terrain: str
  mode: str
  duration: float
  range_scale: float = 1.0
  command: tuple[float, float, float] = (0.0, 0.0, 0.0)
  stages: tuple[tuple[float, float, float], ...] = ()
  stage_seconds: float = 0.0


def terrain_library() -> tuple[TerrainSpec, ...]:
  result = [TerrainSpec("flat", "flat")]
  result += [TerrainSpec(f"slope_{angle}deg", "slope", slope_degrees=angle)
             for angle in (10, 20)]
  result += [TerrainSpec(f"stairs_{cm}cm", "stairs", height=cm / 100)
             for cm in (5, 8, 10)]
  # Geometry seeds are part of the protocol, independent of policy/run seed.
  for index in range(3):
    result += [TerrainSpec(f"rough_slope_10deg_s{index}", "rough_slope",
                           slope_degrees=10, noise_amplitude=.02, seed=1400 + index),
               TerrainSpec(f"rough_slope_20deg_s{index}", "rough_slope",
                           slope_degrees=20, noise_amplitude=.04, seed=1500 + index),
               TerrainSpec(f"obstacles_6cm_s{index}", "obstacles", height=.06, seed=1600 + index),
               TerrainSpec(f"obstacles_10cm_s{index}", "obstacles", height=.10, seed=1700 + index)]
  return tuple(result)


def cases_for_suite(suite: str) -> tuple[Case, ...]:
  result = []
  if suite == "tracking":
    for terrain in terrain_library():
      scales = (1.0, 2.0) if terrain.kind in ("flat", "slope", "rough_slope") else (1.0,)
      for scale in scales:
        for mode in ("linear", "angular", "combined"):
          result.append(Case(f"{terrain.name}_{mode}_r{scale:g}", suite,
                             terrain.name, mode, duration=8.0, range_scale=scale))
  elif suite == "maneuver":
    commands = [(0., 0., 0.)]
    commands += [(v, 0., 0.) for v in (-1.5, -1., -.5, -.2, .2, .5, 1., 1.5, 2.)]
    commands += [(0., v, 0.) for v in (-1., -.5, -.2, .2, .5, 1.)]
    commands += [(0., 0., v) for v in (-1., -.5, -.2, -.1, -.05, .05, .1, .2, .5, 1.)]
    for command in commands:
      yaw = command[2] != 0
      name = "cmd_" + "_".join(f"{v:g}" for v in command)
      result.append(Case(name, suite, "flat", "angular" if yaw else "linear",
                         duration=30.0 if yaw or not any(command) else 8.0, command=command))
  elif suite == "transitions":
    for axis in (0, 1, 2):
      values = (.2, .5, 1.) if axis != 1 else (.2, .5)
      for value in values:
        for sign in (-1, 1):
          positive = tuple(sign * value if i == axis else 0. for i in range(3))
          negative = tuple(-v for v in positive)
          stages = ((0., 0., 0.), positive, (0., 0., 0.), negative, (0., 0., 0.))
          result.append(Case(f"axis{axis}_{sign*value:g}", suite, "flat",
                             "angular" if axis == 2 else "linear", duration=15.0,
                             stages=stages, stage_seconds=3.0))
  elif suite == "traversal":
    for terrain in terrain_library():
      for speed in (-.5, -.75, .5, .75):
        result.append(Case(f"{terrain.name}_vx{speed:g}", suite, terrain.name,
                           "linear", duration=10.0, command=(speed, 0., 0.)))
  else:
    raise ValueError(f"Unknown suite: {suite}")
  return tuple(result)


def case_catalog(suites=("tracking", "maneuver", "transitions", "traversal"), profile="standard"):
  if profile not in ("standard", "quick", "smoke"):
    raise ValueError(f"Unknown profile: {profile}")
  result = []
  for suite in suites:
    cases = cases_for_suite(suite)
    if profile == "smoke":
      if suite in ("tracking", "traversal"):
        names = {"flat", "slope_10deg", "stairs_8cm", "rough_slope_10deg_s0", "obstacles_6cm_s0"}
        cases = tuple(c for c in cases if c.terrain in names and c.range_scale == 1.)
      else:
        cases = cases[:2]
    result.extend(cases)
  return tuple(result)


def reference_schedule(case: Case, trials: int, repeat: int, seed: int = 20261009):
  """Return references at action times; a transition never skips observation update."""
  if trials < 1:
    raise ValueError("trials must be positive")
  total_steps = round((SETTLE_SECONDS + case.duration) / STEP_DT)
  settle = round(SETTLE_SECONDS / STEP_DT)
  schedule = np.zeros((trials, total_steps + 1, 3), dtype=np.float32)
  rng = np.random.default_rng(stable_seed(seed, case.name, repeat, "commands"))
  if case.suite == "tracking":
    low = np.array([-case.range_scale, -1., -case.range_scale])
    high = -low
    targets = rng.uniform(low, high, size=(trials, 3)).astype(np.float32)
    if case.mode == "linear":
      targets[:, 2] = 0.
    elif case.mode == "angular":
      targets[:, :2] = 0.
    schedule[:, settle:, :] = targets[:, None, :]
  elif case.stages:
    stage_steps = round(case.stage_seconds / STEP_DT)
    for index, target in enumerate(case.stages):
      begin = settle + index * stage_steps
      schedule[:, begin:begin + stage_steps, :] = target
    schedule[:, -1, :] = case.stages[-1]
  else:
    schedule[:, settle:, :] = case.command
  return schedule


def protocol_contract(profile: str, suites, trials: int, repeats: int, seed: int) -> dict:
  return {
    "version": PROTOCOL_VERSION, "profile": profile,
    "suites": list(suites), "trials_per_case": trials, "repeats": repeats, "seed": seed,
    "step_dt": STEP_DT, "settle_seconds": SETTLE_SECONDS, "patch_size_m": PATCH_SIZE,
    "xy_tolerance_m_s": XY_TOLERANCE, "yaw_tolerance_rad_s": YAW_TOLERANCE,
    "tolerance_rule": "XY=min(0.20,max(0.05,0.25*command_speed)); yaw=min(0.10,max(0.02,0.25*abs(command_yaw)))",
    "response_hold_seconds": HOLD_SECONDS, "tilt_limit_degrees": TILT_LIMIT_DEGREES,
    "route_length_m": ROUTE_LENGTH, "route_half_width_m": ROUTE_HALF_WIDTH,
    "terrains": [asdict(t) for t in terrain_library()],
    "cases": [asdict(c) for c in case_catalog(suites, profile)],
    "inference": "deterministic; frozen weights and normalization; no optimizer",
    "physics": "nominal; foot friction 0.6; no startup randomization, pushes, observation noise",
    "coordinates": "body-frame linear XY and body-frame angular Z, measured from simulation",
    "errors": "per-trial finite mean through the first failure step inclusive, then equal trial weighting; no post-reset samples; failures reported separately",
    "tracking_fraction": "joint XY/yaw tolerance over the full planned horizon; post-failure time is not successful",
    "failure": "first tilt>70deg, thigh contact force>1N, patch exit, physical/observation NaN/Inf",
    "stall": "signed progress along a nonzero translation command <20% of reference; pure-yaw signed rate <20% of reference",
    "uncertainty": "conditional on the fixed terrain bank; trial bootstrap, not training-seed uncertainty",
  }
