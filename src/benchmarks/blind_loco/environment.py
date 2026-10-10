"""Frozen Go2 evaluation environment: no imports of training task configuration."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz

from src.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg

from .protocol import PATCH_SIZE, OBS_DIM, STEP_DT, TerrainSpec
from .terrains import BenchmarkTerrainCfg


class ReplayCommand(CommandTerm):
  """Reference at state k is the command seen by action k, including after reset."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.schedule = torch.zeros((self.num_envs, 1, 3), device=self.device)
    self.value = torch.zeros((self.num_envs, 3), device=self.device)
    self.active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def command(self):
    return self.value

  def _update_metrics(self):
    pass

  def _resample_command(self, env_ids):
    self.value[env_ids] = 0.

  def _update_command(self, env_ids=None):
    ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    k = self._env.episode_length_buf[ids].clamp_max(self.schedule.shape[1]-1)
    self.value[ids] = self.schedule[ids, k] * self.active[ids, None]


@dataclass(kw_only=True)
class ReplayCommandCfg(CommandTermCfg):
  resampling_time_range: tuple[float, float] = (1e9, 1e9)

  def build(self, env):
    return ReplayCommand(self, env)


def phase(env, period=.6):
  angle = (env.episode_length_buf * env.step_dt) % period / period * (2 * torch.pi)
  value = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
  standing = (env.command_manager.get_command("twist") == 0).all(-1)
  return torch.where(standing[:, None], 0., value)


def reset_root(env, env_ids):
  ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
  robot = env.scene["robot"]
  state = robot.data.default_root_state[ids].clone()
  state[:, :3] += env.scene.env_origins[ids] + env.benchmark_initial_offsets[ids]
  pitch = env.benchmark_initial_pitch[ids]
  state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(pitch), pitch, torch.zeros_like(pitch))
  state[:, 7:13] = 0.
  robot.write_root_link_pose_to_sim(state[:, :7], env_ids=ids)
  robot.write_root_link_velocity_to_sim(state[:, 7:13], env_ids=ids)


def make_environment_cfg(terrains: tuple[TerrainSpec, ...], num_envs: int,
                         history_length: int, seed: int):
  terms = {
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
    "phase": ObservationTermCfg(func=phase),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  feet = ("FR", "FL", "RR", "RL")
  sensors = []
  for name, patterns, reduce in (
    ("feet", tuple(f"{leg}_foot_collision" for leg in feet), "netforce"),
    ("thigh", tuple(f"{leg}_thigh_collision" for leg in feet), "none"),
  ):
    sensors.append(ContactSensorCfg(
      name=name, primary=ContactMatch(mode="geom", pattern=patterns, entity="robot"),
      secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"),
      reduce=reduce, num_slots=1,
    ))
  cfg = ManagerBasedRlEnvCfg(
    seed=seed, decimation=4, episode_length_s=1000., auto_reset=False,
    scene=SceneCfg(
      num_envs=num_envs, extent=2., entities={"robot": deepcopy(get_go2_robot_cfg())},
      sensors=tuple(sensors),
      terrain=TerrainEntityCfg(
        terrain_type="generator", max_init_terrain_level=0,
        terrain_generator=TerrainGeneratorCfg(
          seed=20261009, size=(PATCH_SIZE, PATCH_SIZE), curriculum=True, num_rows=1,
          sub_terrains={t.name: BenchmarkTerrainCfg(terrain=t) for t in terrains},
        ),
      ),
    ),
    observations={"actor": ObservationGroupCfg(
      terms=terms, concatenate_terms=True, enable_corruption=False,
      history_length=history_length, flatten_history_dim=False,
    )},
    actions={"joint_pos": JointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=.25, use_default_offset=True,
    )},
    commands={"twist": ReplayCommandCfg()},
    events={
      "reset_root": EventTermCfg(func=reset_root, mode="reset"),
      "reset_joints": EventTermCfg(func=mdp.reset_joints_by_offset, mode="reset", params={
        "position_range": (0., 0.), "velocity_range": (0., 0.),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      }),
    },
    sim=SimulationCfg(
      nconmax=64, njmax=1500, contact_sensor_maxmatch=256,
      mujoco=MujocoCfg(timestep=.005, iterations=10, ls_iterations=20, ccd_iterations=200),
    ),
  )
  if cfg.decimation * cfg.sim.mujoco.timestep != STEP_DT:
    raise ValueError("Control timestep does not match the frozen protocol")
  return cfg


def validate_observation(obs, history_length):
  value = obs["actor"]
  if value.ndim != 3 or tuple(value.shape[1:]) != (history_length, OBS_DIM):
    raise ValueError(f"Expected [N,{history_length},{OBS_DIM}], got {tuple(value.shape)}")
