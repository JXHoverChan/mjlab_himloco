"""Unitree Go2 velocity environment configurations."""

from copy import deepcopy
from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RayCastSensorCfg,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.managers.observation_manager import ObservationTermCfg, ObservationGroupCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from src.tasks.velocity import mdp as local_mdp
from src.tasks.velocity.mdp.reward_profiles import apply_him_reward_profile
from src.tasks.velocity.mdp.stratified_velocity_command import StratifiedVelocityCommandCfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 200
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  thigh_geom_names = tuple(
    f"{leg}_thigh_collision" for leg in foot_names
  )
  thigh_ground_cfg = ContactSensorCfg(
    name="thigh_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=thigh_geom_names,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
    thigh_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  # cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards.pop("foot_gait", None)

  cfg.terminations.pop("fell_over", None)

  cfg.terminations["illegal_contact_1"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": thigh_ground_cfg.name},
  )

  cfg.terminations["nonfoot_ground_2"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg

def HIMloco_Go2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Go2 velocity configuration used by the HIM policy."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.terminations["terrain_bounds"] = TerminationTermCfg(
    func=local_mdp.out_of_terrain_bounds,
    params={"margin": 0.3},
    time_out=True,
  )


  if not play:
    # Sampling runs at every command timer expiry, not just episode reset.
    cfg.commands["twist"] = StratifiedVelocityCommandCfg(**vars(cfg.commands["twist"]))
    cfg.curriculum["terrain_levels"] = CurriculumTermCfg(
      func=local_mdp.AdaptiveTerrainCurriculum,
      params={
        "command_name": "twist",
        "command_activity_threshold": 0.1,
        "terrain_down_distance_ratio": 0.5,
        "linear_error_up_threshold": 0.75,
        "linear_error_down_threshold": 1.10,
        "yaw_error_up_threshold": 0.20,
        "yaw_error_down_threshold": 0.35,
      },
    )
    cfg.curriculum["command_vel"] = CurriculumTermCfg(
      func=local_mdp.StratifiedVelocityCurriculum,
      params={
        "command_name": "twist",
        "mode_probabilities": {
          "standing": 0.05,
          "low_forward": 0.15,
          "low_backward": 0.15,
          "lateral": 0.10,
          "pure_yaw": 0.20,
          "translation": 0.10,
          "combined": 0.25,
        },
        "linear_speed_band": (0.2, 0.8),
        "yaw_speed_band": (0.05, 1.0),
        "initial_ranges": {
          "lin_vel_x": (-0.5, 1.0),
          "lin_vel_y": (-0.5, 0.5),
          "ang_vel_z": (-0.5, 0.5),
        },
        "maximum_ranges": {
          "lin_vel_x": (-1.5, 2.5),
          "lin_vel_y": (-1.5, 1.5),
          "ang_vel_z": (-1.5, 1.5),
        },
        "error_up_threshold": 0.75,
        "error_down_threshold": 1.0,
        "level_step": 0.025,
        "update_every_episodes": 4096,
      },
    )

  cfg.scene.terrain.max_init_terrain_level = 10

  # Measure foot clearance relative to the local terrain, not world z.
  site_names = ("FR", "FL", "RR", "RL")
  foot_height_scan = TerrainHeightSensorCfg(
    name="foot_height_scan",
    frame=tuple(
      ObjRef(type="site", name=site_name, entity="robot")
      for site_name in site_names
    ),
    ray_alignment="yaw",
    pattern=RingPatternCfg.single_ring(radius=0.04, num_samples=4),
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (foot_height_scan,)
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards["terrain_progress"] = RewardTermCfg(
    func=local_mdp.terrain_progress,
    weight=0.1,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
    },
  )
  # cfg.rewards["foot_terrain_clearance"] = RewardTermCfg(
  #   func=local_mdp.feet_terrain_clearance,
  #   weight=-1.0,
  #   params={
  #     "target_height": 0.15,
  #     "height_sensor_name": foot_height_scan.name,
  #     "command_name": "twist",
  #     "command_threshold": 0.1,
  #     "asset_cfg": SceneEntityCfg("robot", site_names=site_names),
  #   },
  # )

  # Allow incidental calf/hip/body terrain contacts in HIM locomotion.
  # Keep illegal_contact_1, which monitors thigh contact, as the contact termination.
  cfg.terminations.pop("nonfoot_ground_2", None)

  # HIM receives proprioception history; terrain height remains privileged to the critic.
  for group in ("actor", "critic"):
    cfg.observations[group].terms["phase"].func = local_mdp.phase_for_commanded_motion
  cfg.observations["actor"].terms.pop("height_scan")
  cfg.observations["actor"].history_length = 5
  cfg.observations["actor"].flatten_history_dim = False

  # Target is the same one-step observation space, evaluated after env.step().
  # It is kept clean so the target branch supplies the contrastive teacher signal.
  target_enc_terms = deepcopy(cfg.observations["actor"].terms)
  for term in target_enc_terms.values():
    term.noise = None
  cfg.observations["target_encoder"] = ObservationGroupCfg(
    terms=target_enc_terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )

  # Clean privileged velocity target for the source encoder's auxiliary head.
  cfg.observations["velocity"] = ObservationGroupCfg(
    terms={
      "base_lin_vel": ObservationTermCfg(
        func=mdp.builtin_sensor,
        params={"sensor_name": "robot/imu_lin_vel"},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
    history_length=0,
  )

  # Separate tracking objectives from stability constraints; reserve pose for rest.
  apply_him_reward_profile(cfg, "baseline_v2")

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

  return cfg
