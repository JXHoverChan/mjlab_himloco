"""Script to play RL agent with RSL-RL."""

import csv
import json
import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago
from scripts.joystick import JoystickCommandEnv, JoystickSettings, LinuxJoystick
from src.tasks.velocity.mdp.reward_profiles import RewardProfile, apply_him_reward_profile


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """Parse a W&B datetime string (or pass through a datetime object)."""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  """Optional checkpoint name within the W&B run to load (e.g. 'model_4000.pt')."""
  checkpoint_file: str | None = None
  reward_profile: RewardProfile | None = None
  """HIM reward variant used for diagnostic reward attribution."""
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  joystick: bool = False
  """Read a physical Linux joystick and map it to the ``twist`` command."""
  joystick_device: str = "/dev/input/js1"
  """Linux joystick device used when ``joystick`` is enabled."""
  joystick_env_id: int = 0
  """Environment receiving the physical joystick command."""
  joystick_forward_axis: int = 1
  joystick_lateral_axis: int = 0
  joystick_yaw_axis: int = 2
  joystick_deadzone: float = 0.08
  joystick_scale: float = 1.0
  joystick_zero_button: int = 0
  """Button held to command zero velocity; use -1 to disable."""
  joystick_invert_forward: bool = True
  joystick_invert_lateral: bool = True
  joystick_invert_yaw: bool = True
  mouse_perturbations: bool = True
  """Enable native MuJoCo Ctrl+mouse external force/torque perturbations."""
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""
  diagnose: bool = False
  """Record per-step state, reward terms, contacts, and termination flags during play."""
  diagnose_steps: int = 0
  """Maximum number of diagnostic steps; zero records until the viewer exits."""
  diagnose_file: str | None = None
  """CSV path for diagnostic output. Defaults next to the loaded checkpoint."""
  diagnose_headless: bool = False
  """Run diagnostic steps without opening a viewer."""
  diagnose_seed: int | None = None
  """Optional environment seed used by diagnostic runs."""
  log_root: str = "logs/rsl_rl"
  """Root directory under which experiment logs are written."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def _max_sensor_force(sensor, num_envs: int, device: torch.device) -> torch.Tensor:
  """Return the maximum contact force per environment for a diagnostic sensor."""
  data = sensor.data
  force = data.force_history if data.force_history is not None else data.force
  if force is None:
    return torch.zeros(num_envs, device=device)
  force_mag = torch.linalg.vector_norm(force, dim=-1)
  return force_mag.amax(dim=tuple(range(1, force_mag.ndim)))


def _max_sensor_found(sensor, num_envs: int, device: torch.device) -> torch.Tensor:
  """Return whether a diagnostic contact sensor found contact per environment."""
  found = sensor.data.found
  if found is None:
    return torch.zeros(num_envs, device=device)
  return found.float().amax(dim=tuple(range(1, found.ndim)))


class _DiagnosticRecorder:
  """Record terminal states before manually resetting diagnostic environments.

  This wrapper is intentionally limited to play. The training environment keeps its
  normal auto-reset path and does not pay the per-step CPU synchronization cost.
  """

  def __init__(self, env: RslRlVecEnvWrapper, output_file: Path, max_steps: int) -> None:
    self.env = env
    self.output_file = output_file
    self.max_steps = max_steps
    self._step = 0
    self._recording_complete = False
    self._closed = False
    self._episode_ids = torch.zeros(
      env.num_envs, dtype=torch.long, device=env.unwrapped.device
    )
    self._rows = 0
    self._done_count = 0
    self._action_rate_sum = 0.0
    self._action_rate_reward_sum = 0.0
    self._termination_counts = {
      name: 0 for name in env.unwrapped.termination_manager.active_terms
    }

    self.output_file.parent.mkdir(parents=True, exist_ok=True)
    self._file = self.output_file.open("w", newline="")

    self._termination_names = env.unwrapped.termination_manager.active_terms
    self._reward_names = env.unwrapped.reward_manager.active_terms
    self._termination_columns = {
      name: f"termination_{name.replace('/', '_')}"
      for name in self._termination_names
    }
    self._reward_columns = {
      name: f"reward_rate_{name.replace('/', '_')}"
      for name in self._reward_names
    }
    self._fieldnames = [
      "step",
      "env_id",
      "episode_id",
      "episode_step",
      "step_dt",
      "done",
      "terminated",
      "time_out",
      "command_x",
      "command_y",
      "command_yaw",
      "velocity_x",
      "velocity_y",
      "velocity_z",
      "yaw_velocity",
      "velocity_error_xy",
      "velocity_error_yaw",
      "action_rate_raw",
      "action_rate_reward_rate",
      "reward_total",
      "base_x",
      "base_y",
      "base_z",
      "gravity_x",
      "gravity_y",
      "gravity_z",
      "terrain_level",
      "nonfoot_force_max",
      "thigh_force_max",
      "nonfoot_contact",
      "thigh_contact",
      *self._termination_columns.values(),
      *self._reward_columns.values(),
    ]
    self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
    self._writer.writeheader()
    self._file.flush()
    print(f"[INFO] Diagnostic CSV: {self.output_file}")

  @property
  def unwrapped(self):
    return self.env.unwrapped

  def __getattr__(self, name):
    return getattr(self.env, name)

  def _sensor_values(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name not in self.unwrapped.scene.sensors:
      zeros = torch.zeros(self.env.num_envs, device=self.unwrapped.device)
      return zeros, zeros
    sensor = self.unwrapped.scene.sensors[name]
    return (
      _max_sensor_force(sensor, self.env.num_envs, self.unwrapped.device),
      _max_sensor_found(sensor, self.env.num_envs, self.unwrapped.device),
    )

  def step(self, actions: torch.Tensor):
    base_env = self.unwrapped
    if self._recording_complete:
      obs, reward, dones, extras = self.env.step(actions)
      done = (dones != 0).detach()
      done_ids = done.nonzero(as_tuple=False).squeeze(-1)
      if len(done_ids) > 0:
        base_env.reset(env_ids=done_ids)
        obs = self.env.get_observations()
      return obs, reward, dones, extras

    command = base_env.command_manager.get_command("twist")
    if command is None:
      raise ValueError("Diagnostic mode requires the 'twist' command term.")
    command_before = command.detach().clone()

    obs, reward, dones, extras = self.env.step(actions)

    terminated = base_env.termination_manager.terminated.detach().clone()
    time_out = base_env.termination_manager.time_outs.detach().clone()
    done = (dones != 0).detach()
    term_values = {
      name: base_env.termination_manager.get_term(name).detach().clone()
      for name in self._termination_names
    }

    robot = base_env.scene["robot"]
    root_pos = robot.data.root_link_pos_w.detach()
    velocity = robot.data.root_link_lin_vel_b.detach()
    angular_velocity = robot.data.root_link_ang_vel_b.detach()
    gravity = robot.data.projected_gravity_b.detach()
    velocity_error_xy = torch.linalg.vector_norm(
      command_before[:, :2] - velocity[:, :2], dim=1
    )
    velocity_error_yaw = torch.abs(command_before[:, 2] - angular_velocity[:, 2])

    action_manager = base_env.action_manager
    action_rate_raw = torch.sum(
      torch.square(action_manager.action - action_manager.prev_action), dim=1
    ).detach()
    reward_rates = base_env.reward_manager._step_reward.detach()  # diagnostics only
    reward_name_to_index = {
      name: index for index, name in enumerate(self._reward_names)
    }
    action_rate_index = reward_name_to_index.get("action_rate_l2")
    if action_rate_index is None:
      action_rate_reward_rate = torch.zeros_like(action_rate_raw)
    else:
      action_rate_reward_rate = reward_rates[:, action_rate_index]

    terrain = getattr(base_env.scene, "terrain", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is None:
      terrain_level = torch.full(
        (self.env.num_envs,), float("nan"), device=base_env.device
      )
    else:
      terrain_level = terrain_levels.float().detach()

    nonfoot_force, nonfoot_contact = self._sensor_values("nonfoot_ground_touch")
    thigh_force, thigh_contact = self._sensor_values("thigh_ground_touch")
    episode_step = base_env.episode_length_buf.detach()
    reward_cpu = reward.detach().cpu()

    command_cpu = command_before.cpu()
    root_pos_cpu = root_pos.cpu()
    velocity_cpu = velocity.cpu()
    angular_velocity_cpu = angular_velocity.cpu()
    gravity_cpu = gravity.cpu()
    velocity_error_xy_cpu = velocity_error_xy.cpu()
    velocity_error_yaw_cpu = velocity_error_yaw.cpu()
    action_rate_raw_cpu = action_rate_raw.cpu()
    action_rate_reward_rate_cpu = action_rate_reward_rate.cpu()
    terrain_level_cpu = terrain_level.cpu()
    nonfoot_force_cpu = nonfoot_force.cpu()
    thigh_force_cpu = thigh_force.cpu()
    nonfoot_contact_cpu = nonfoot_contact.cpu()
    thigh_contact_cpu = thigh_contact.cpu()
    episode_step_cpu = episode_step.cpu()
    term_cpu = {name: values.cpu() for name, values in term_values.items()}
    reward_rate_cpu = reward_rates.cpu()

    for env_id in range(self.env.num_envs):
      row = {
        "step": self._step,
        "env_id": env_id,
        "episode_id": int(self._episode_ids[env_id].item()),
        "episode_step": int(episode_step_cpu[env_id].item()),
        "step_dt": base_env.step_dt,
        "done": int(done[env_id].item()),
        "terminated": int(terminated[env_id].item()),
        "time_out": int(time_out[env_id].item()),
        "command_x": float(command_cpu[env_id, 0].item()),
        "command_y": float(command_cpu[env_id, 1].item()),
        "command_yaw": float(command_cpu[env_id, 2].item()),
        "velocity_x": float(velocity_cpu[env_id, 0].item()),
        "velocity_y": float(velocity_cpu[env_id, 1].item()),
        "velocity_z": float(velocity_cpu[env_id, 2].item()),
        "yaw_velocity": float(angular_velocity_cpu[env_id, 2].item()),
        "velocity_error_xy": float(velocity_error_xy_cpu[env_id].item()),
        "velocity_error_yaw": float(velocity_error_yaw_cpu[env_id].item()),
        "action_rate_raw": float(action_rate_raw_cpu[env_id].item()),
        "action_rate_reward_rate": float(
          action_rate_reward_rate_cpu[env_id].item()
        ),
        "reward_total": float(reward_cpu[env_id].item()),
        "base_x": float(root_pos_cpu[env_id, 0].item()),
        "base_y": float(root_pos_cpu[env_id, 1].item()),
        "base_z": float(root_pos_cpu[env_id, 2].item()),
        "gravity_x": float(gravity_cpu[env_id, 0].item()),
        "gravity_y": float(gravity_cpu[env_id, 1].item()),
        "gravity_z": float(gravity_cpu[env_id, 2].item()),
        "terrain_level": float(terrain_level_cpu[env_id].item()),
        "nonfoot_force_max": float(nonfoot_force_cpu[env_id].item()),
        "thigh_force_max": float(thigh_force_cpu[env_id].item()),
        "nonfoot_contact": float(nonfoot_contact_cpu[env_id].item()),
        "thigh_contact": float(thigh_contact_cpu[env_id].item()),
      }
      for name, column in self._termination_columns.items():
        row[column] = int(term_cpu[name][env_id].item())
        if term_cpu[name][env_id]:
          self._termination_counts[name] += 1
      for name, column in self._reward_columns.items():
        row[column] = float(reward_rate_cpu[env_id, reward_name_to_index[name]].item())
      self._writer.writerow(row)

    self._file.flush()
    self._rows += self.env.num_envs
    self._step += 1
    self._done_count += int(done.sum().item())
    self._action_rate_sum += float(action_rate_raw.sum().item())
    self._action_rate_reward_sum += float(action_rate_reward_rate.sum().item())

    if self.max_steps > 0 and self._step >= self.max_steps:
      self._recording_complete = True
      print("[INFO] Diagnostic step limit reached; recording is paused.")

    done_ids = done.nonzero(as_tuple=False).squeeze(-1)
    if len(done_ids) > 0:
      base_env.reset(env_ids=done_ids)
      self._episode_ids[done_ids] += 1
      obs = self.env.get_observations()

    return obs, reward, dones, extras

  def close(self) -> None:
    if self._closed:
      return
    self._closed = True
    summary_file = self.output_file.with_suffix(".summary.json")
    summary = {
      "steps": self._step,
      "num_envs": self.env.num_envs,
      "rows": self._rows,
      "episodes_completed": self._done_count,
      "termination_counts": self._termination_counts,
      "mean_action_rate_raw": self._action_rate_sum / max(self._rows, 1),
      "mean_action_rate_reward_rate": self._action_rate_reward_sum
      / max(self._rows, 1),
      "csv_file": str(self.output_file),
    }
    with summary_file.open("w") as file:
      json.dump(summary, file, indent=2)
    self._file.close()
    print(f"[INFO] Diagnostic summary: {summary_file}")
    self.env.close()


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  if cfg.reward_profile is not None:
    apply_him_reward_profile(env_cfg, cfg.reward_profile)

  if cfg.diagnose_seed is not None:
    env_cfg.seed = cfg.diagnose_seed

  if cfg.joystick:
    command_cfg = env_cfg.commands.get("twist")
    if command_cfg is None or not hasattr(command_cfg, "ranges"):
      raise ValueError("Joystick mode requires a velocity command named 'twist'.")

    # Manual input owns the command for the whole play episode. Disable the
    # command term's autonomous resampling and state-dependent rewrites so the
    # policy, rewards, and viewer all observe the same joystick command.
    command_cfg.resampling_time_range = (
      float(env_cfg.episode_length_s),
      float(env_cfg.episode_length_s),
    )
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.rel_world_envs = 0.0
    command_cfg.rel_forward_envs = 0.0
    env_cfg.events.pop("push_robot", None)
    print("[INFO] Joystick mode: automatic velocity resampling and push events disabled")

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  if cfg.diagnose:
    # Preserve the terminal state long enough for _DiagnosticRecorder to log it.
    env_cfg.auto_reset = False
    print("[INFO]: Diagnostic mode enabled (manual reset after logging)")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  elif cfg.joystick:
    # A physical controller has one command stream. Avoid creating unused
    # environments that would continue running with random commands.
    env_cfg.scene.num_envs = 1
  elif cfg.diagnose:
    # A single environment makes command/termination traces unambiguous.
    env_cfg.scene.num_envs = 1
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  if cfg.joystick:
    command_cfg = env_cfg.commands.get("twist")
    command_ranges = getattr(command_cfg, "ranges", None)
    if command_ranges is None:
      raise ValueError("Joystick mode requires a UniformVelocityCommandCfg named 'twist'.")
    ranges = (
      tuple(command_ranges.lin_vel_x),
      tuple(command_ranges.lin_vel_y),
      tuple(command_ranges.ang_vel_z),
    )
    joystick = LinuxJoystick(
      cfg.joystick_device,
      JoystickSettings(
        command_ranges=ranges,
        forward_axis=cfg.joystick_forward_axis,
        lateral_axis=cfg.joystick_lateral_axis,
        yaw_axis=cfg.joystick_yaw_axis,
        invert_forward=cfg.joystick_invert_forward,
        invert_lateral=cfg.joystick_invert_lateral,
        invert_yaw=cfg.joystick_invert_yaw,
        deadzone=cfg.joystick_deadzone,
        scale=cfg.joystick_scale,
        zero_button=cfg.joystick_zero_button,
      ),
    )
    if not 0 <= cfg.joystick_env_id < env.num_envs:
      joystick.close()
      raise ValueError(
        f"joystick_env_id={cfg.joystick_env_id} is invalid for {env.num_envs} environments."
      )
    print(
      f"[INFO] Physical joystick controls env {cfg.joystick_env_id}: "
      "forward/lateral/yaw -> twist"
    )
    env = JoystickCommandEnv(env, joystick, cfg.joystick_env_id)

  if cfg.diagnose:
    if cfg.diagnose_file is not None:
      diagnose_file = Path(cfg.diagnose_file)
    elif log_dir is not None:
      checkpoint_stem = resume_path.stem if resume_path is not None else "play"
      diagnose_file = log_dir / "diagnostics" / f"{checkpoint_stem}.csv"
    else:
      diagnose_file = Path(cfg.log_root) / "play_diagnostics" / "trajectory.csv"
    env = _DiagnosticRecorder(env, diagnose_file, cfg.diagnose_steps)

  if cfg.diagnose and cfg.diagnose_headless:
    if cfg.diagnose_steps <= 0:
      raise ValueError("Headless diagnostic mode requires --diagnose-steps > 0.")
    obs = env.get_observations()
    with torch.no_grad():
      for _ in range(cfg.diagnose_steps):
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
    env.close()
    return

  # Build checkpoint manager for hot-swapping checkpoints in the viewer.
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(
      env,
      policy,
      enable_perturbations=cfg.mouse_perturbations,
    ).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  maybe_print_top_level_help("play")

  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab  # noqa: F401
  import src.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
