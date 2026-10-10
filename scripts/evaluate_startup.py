"""Parallel, fixed-command tests of startup from rest, without a viewer.

Commands switch through the command manager at the end of the settling phase,
so the first tested action receives the new command in its observation history.
Metrics exclude post-failure resets and are accumulated on the device.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.tasks.velocity.mdp.reward_profiles import RewardProfile, apply_him_reward_profile


@dataclass
class StartupConfig:
  checkpoint_file: str
  task: str = "Unitree-Go2-HIM"
  reward_profile: RewardProfile = "revised"
  device: str = "cuda:0"
  num_envs: int = 64
  repeats: int = 3
  seed: int = 2026
  terrain: Literal["flat", "rough"] = "flat"
  commands: str = "1,0,0;2,0,0;0,0.75,0;0,0,1"
  """Semicolon-separated body-frame vx,vy,wz command triples."""
  settle_seconds: float = 0.5
  duration_seconds: float = 3.0
  output_file: str = "artifacts/experiment/startup.json"


def parse_commands(value: str) -> list[tuple[float, float, float]]:
  import math

  commands = [tuple(float(x) for x in part.split(",")) for part in value.split(";")]
  if not commands or any(len(c) != 3 or not all(math.isfinite(x) for x in c) for c in commands):
    raise ValueError("Commands must be finite vx,vy,wz triples separated by semicolons.")
  if any(sum(x * x for x in c) == 0 for c in commands):
    raise ValueError("Each test command must be nonzero.")
  return commands


def _fixed_ranges(term, command: tuple[float, float, float]) -> None:
  term.cfg.ranges.lin_vel_x = (command[0], command[0])
  term.cfg.ranges.lin_vel_y = (command[1], command[1])
  term.cfg.ranges.ang_vel_z = (command[2], command[2])


def evaluate(config: StartupConfig) -> dict:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  checkpoint = Path(config.checkpoint_file).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if config.num_envs < 1 or config.repeats < 1:
    raise ValueError("num_envs and repeats must be positive")
  if config.settle_seconds <= 0 or config.duration_seconds < 1.0:
    raise ValueError("Use a positive settling time and at least one second of testing")
  commands = parse_commands(config.commands)
  configure_torch_backends()
  env_cfg = load_env_cfg(config.task, play=True)
  agent_cfg = load_rl_cfg(config.task)
  apply_him_reward_profile(env_cfg, config.reward_profile)
  env_cfg.seed = config.seed
  env_cfg.scene.num_envs = config.num_envs
  env_cfg.auto_reset = False
  env_cfg.curriculum = {}
  env_cfg.events.pop("push_robot", None)
  if config.terrain == "flat":
    env_cfg.scene.terrain.terrain_type = "plane"
    env_cfg.scene.terrain.terrain_generator = None
    env_cfg.events.pop("randomize_terrain", None)
  command_cfg = env_cfg.commands["twist"]
  command_cfg.resampling_time_range = (1e9, 1e9)
  command_cfg.heading_command = False
  command_cfg.ranges.heading = None
  command_cfg.rel_standing_envs = 0.0
  command_cfg.rel_heading_envs = 0.0
  command_cfg.rel_world_envs = 0.0
  command_cfg.rel_forward_envs = 0.0
  command_cfg.init_velocity_prob = 0.0

  base = ManagerBasedRlEnv(cfg=env_cfg, device=config.device)
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  try:
    runner_cls = load_runner_cls(config.task)
    if runner_cls is None:
      raise ValueError("Startup evaluation requires a registered policy runner")
    runner = runner_cls(env, asdict(agent_cfg), device=config.device)
    runner.load(str(checkpoint), load_cfg={"actor": True}, map_location=config.device)
    policy = runner.get_inference_policy(device=config.device)
    term = base.command_manager.get_term("twist")
    dt = base.step_dt
    settle_steps = max(1, round(config.settle_seconds / dt))
    steps = round(config.duration_seconds / dt)
    reward_names = base.reward_manager.active_terms
    termination_names = base.termination_manager.active_terms
    cases = []

    def reset_done(obs, dones):
      ids = dones.nonzero(as_tuple=False).flatten()
      if len(ids):
        values, _ = base.reset(env_ids=ids)
        policy.reset(dones)
        return TensorDict(values, batch_size=[env.num_envs])
      return obs

    with torch.inference_mode():
      for case_id, command in enumerate(commands):
        trial_results = []
        for repeat in range(config.repeats):
          _fixed_ranges(term, (0.0, 0.0, 0.0))
          base.reset(seed=config.seed + case_id * config.repeats + repeat)
          policy.reset()
          obs = env.get_observations()
          alive = torch.ones(env.num_envs, device=config.device, dtype=torch.bool)
          # A failure during settling remains a failed trial, even after reset.
          for step in range(settle_steps):
            if step == settle_steps - 1:
              _fixed_ranges(term, command)
              term.time_left.fill_(0.0)
            obs, _, dones, _ = env.step(policy(obs))
            alive &= dones == 0
            obs = reset_done(obs, dones)

          target = torch.tensor(command, device=config.device)
          speed = torch.linalg.vector_norm(target[:2])
          planar = command[0] != 0 or command[1] != 0
          hit = torch.zeros_like(alive)
          latency = torch.full((env.num_envs,), float("inf"), device=config.device)
          first_second_survival = alive.clone()
          first_progress = torch.zeros(env.num_envs, device=config.device)
          error_xy = torch.zeros(env.num_envs, device=config.device)
          error_yaw = torch.zeros_like(error_xy)
          sample_count = torch.zeros_like(error_xy)
          action_change = torch.zeros_like(error_xy)
          reward_sum = torch.zeros(len(reward_names), device=config.device)
          failures = torch.zeros(len(termination_names), device=config.device)

          for step in range(steps):
            obs, _, dones, _ = env.step(policy(obs))
            valid = alive.clone()
            velocity = base.scene["robot"].data.root_link_lin_vel_b
            yaw = base.scene["robot"].data.root_link_ang_vel_b[:, 2]
            projection = (velocity[:, :2] * target[:2]).sum(dim=1) / speed.clamp_min(1e-6)
            error_xy += torch.linalg.vector_norm(velocity[:, :2] - target[:2], dim=1) * valid
            error_yaw += (yaw - target[2]).abs() * valid
            action_change += ((base.action_manager.action - base.action_manager.prev_action) ** 2).sum(dim=1) * valid
            sample_count += valid
            reward_sum += (base.reward_manager._step_reward * valid[:, None]).sum(dim=0)
            for index, name in enumerate(termination_names):
              failures[index] += (base.termination_manager.get_term(name) & valid).sum()
            alive &= dones == 0
            if (step + 1) * dt <= 1.0 + 1e-6:
              first_progress += projection.clamp_min(0.0) * dt * valid
              reached = projection >= 0.5 * speed if planar else yaw * target[2].sign() >= 0.5 * target[2].abs()
              newly_hit = reached & alive & ~hit
              latency[newly_hit] = (step + 1) * dt
              hit |= newly_hit
              first_second_survival = alive.clone()
            obs = reset_done(obs, dones)

          successful = hit & first_second_survival
          trial_results.append({
            "trials": env.num_envs,
            "successes": int(successful.sum().item()),
            "survivors": int(alive.sum().item()),
            "latency_sum": float(latency[successful].sum().item()),
            "first_second_progress_sum": float(first_progress.sum().item()),
            "xy_error_sum": float(error_xy.sum().item()),
            "yaw_error_sum": float(error_yaw.sum().item()),
            "samples": float(sample_count.sum().item()),
            "action_change_sum": float(action_change.sum().item()),
            "reward_sum": reward_sum.cpu().tolist(),
            "failures": failures.cpu().tolist(),
          })
        total = lambda key: sum(trial[key] for trial in trial_results)
        samples = total("samples")
        successes = total("successes")
        cases.append({
          "command": command,
          "trials": total("trials"),
          "startup_success_rate": successes / total("trials"),
          "survival_rate": total("survivors") / total("trials"),
          "startup_latency_s": total("latency_sum") / successes if successes else None,
          "first_second_progress_m": total("first_second_progress_sum") / total("trials"),
          "mean_velocity_error_xy": total("xy_error_sum") / samples if samples else None,
          "mean_velocity_error_yaw": total("yaw_error_sum") / samples if samples else None,
          "mean_action_rate_raw": total("action_change_sum") / samples if samples else None,
          "reward_rates": {name: sum(t["reward_sum"][i] for t in trial_results) / samples if samples else None for i, name in enumerate(reward_names)},
          "termination_counts": {name: int(sum(t["failures"][i] for t in trial_results)) for i, name in enumerate(termination_names)},
        })
        print(f"[INFO] command={command}: startup={cases[-1]['startup_success_rate']:.3f}, survival={cases[-1]['survival_rate']:.3f}", flush=True)
    result = {
      "config": asdict(config), "step_dt": dt,
      "startup_definition": "Reach 50% of commanded planar speed (or yaw for pure rotation) within one second and survive that second. Settling failures count as failures.",
      "error_definition": "Errors and reward rates include pre-failure states only; compare survival separately.",
      "cases": cases,
    }
    output = Path(config.output_file).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result
  finally:
    env.close()


if __name__ == "__main__":
  evaluate(tyro.cli(StartupConfig))
