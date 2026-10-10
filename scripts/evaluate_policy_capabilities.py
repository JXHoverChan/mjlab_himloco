"""Paired, headless capability evaluation; never updates policy parameters.

Each checkpoint uses the same simulator, randomized physical parameters,
initial states and command schedules. Failed/completed trials are censored.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand, UniformVelocityCommandCfg,
)
from mjlab.utils.torch import configure_torch_backends


RUN = "logs/rsl_rl/go2_him/2026-10-08_11-19-09_Go2-HIM-Rough-2.1-terrain-reward"


@dataclass
class CapabilityConfig:
  checkpoints: tuple[str, ...] = (f"{RUN}/model_1000.pt", f"{RUN}/model_2300.pt")
  output_dir: str = "artifacts/experiment/policy_capabilities"
  device: str = "cuda:0"
  seed: int = 20261008
  repeats: int = 3
  trials_per_case: int = 24
  tracking_trials_per_profile: int = 128
  terrain_levels: tuple[int, ...] = (4, 7, 9)
  settle_seconds: float = 0.5
  tracking_seconds: float = 20.0
  traversal_seconds: float = 8.0
  maneuver_seconds: float = 10.0
  suites: tuple[str, ...] = ("tracking", "heading_tracking", "traversal", "maneuver")


class ScheduledVelocityCommand(UniformVelocityCommand):
  """Replay commands through the normal command/observation update order."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.schedule = torch.zeros((self.num_envs, 1, 3), device=self.device)
    self.heading_schedule = torch.zeros((self.num_envs, 1), device=self.device)
    self.use_heading = False
    self.stage_steps = 1
    self.settle_steps = 1
    self.yaw_limits = torch.ones(self.num_envs, device=self.device)
    self.active = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

  def _resample_command(self, env_ids):
    # All reference samples are generated before the run, not on reset.
    self.vel_command_b[env_ids] = 0.0

  def _update_command(self, env_ids=None):
    ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
    elapsed = self._env.episode_length_buf[ids] - self.settle_steps
    stage = (elapsed.clamp_min(0) // self.stage_steps).clamp_max(self.schedule.shape[1] - 1)
    self.vel_command_b[ids] = self.schedule[ids, stage]
    if self.use_heading:
      error = self.heading_schedule[ids, stage] - self.robot.data.heading_w[ids]
      error = torch.remainder(error + math.pi, 2 * math.pi) - math.pi
      yaw = self.cfg.heading_control_stiffness * error
      self.vel_command_b[ids, 2] = torch.maximum(
        torch.minimum(yaw, self.yaw_limits[ids]), -self.yaw_limits[ids]
      )
    self.vel_command_b[ids] *= ((elapsed >= 0) & self.active[ids]).unsqueeze(1)


@dataclass(kw_only=True)
class ScheduledVelocityCfg(UniformVelocityCommandCfg):
  def build(self, env):
    return ScheduledVelocityCommand(self, env)


def ranges_at_level(level):
  return np.array([[-0.5 - level, 1.0 + 1.5 * level],
                   [-0.5 - level, 0.5 + level], [-0.5 - level, 0.5 + level]])


def make_cases(suite, config):
  if suite in ("tracking", "heading_tracking"):
    return [{"case": f"commands_{level}", "profile": level, "terrain": "flat", "level": 0}
            for level in (0.55, 1.0)]
  if suite == "maneuver":
    commands = [(v, 0., 0.) for v in (-2., -1., -0.5, 0.5, 1., 2.)]
    commands += [(0., v, 0.) for v in (-1., 1.)]
    commands += [(0., 0., v) for v in (-1., -.5, -.2, -.1, -.05, .05, .1, .2, .5, 1.)]
    return [{"case": f"command_{c[0]}_{c[1]}_{c[2]}", "command": c,
             "terrain": "flat", "level": 0} for c in commands]
  terrain_levels = [("flat", 0), ("random_rough", 4)]
  for terrain in ("pyramid_stairs", "pyramid_stairs_inv", "hf_pyramid_slope",
                  "hf_pyramid_slope_inv", "wave_terrain"):
    terrain_levels.extend((terrain, level) for level in config.terrain_levels)
  return [{"case": f"{terrain}_L{level}_vx{speed}", "terrain": terrain,
           "level": level, "command": (speed, 0., 0.)}
          for terrain, level in terrain_levels for speed in (-1.5, -0.75, 0.75, 1.5)]


def build_env(suite, config, cases, n):
  cfg = load_env_cfg("Unitree-Go2-HIM", play=False)
  agent_cfg = load_rl_cfg("Unitree-Go2-HIM")
  cfg.seed = config.seed
  cfg.scene.num_envs = n
  cfg.auto_reset = False
  cfg.curriculum = {}
  cfg.episode_length_s = 60.0
  cfg.events.pop("push_robot", None)
  cfg.events.pop("randomize_terrain", None)
  # Nominal heading enables a physically matched forward/backward route.
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0., 0.), "yaw": (0., 0.)}
  cfg.events["reset_base"].params["velocity_range"] = {}
  for group in cfg.observations.values():
    group.enable_corruption = False
  if suite == "traversal":
    generator = cfg.scene.terrain.terrain_generator
    generator.seed = config.seed
    generator.curriculum = True
    generator.num_rows = 10
    generator.difficulty_range = (0., 1.)
    cfg.scene.terrain.max_init_terrain_level = 0
  else:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
  original = cfg.commands["twist"]
  cfg.commands["twist"] = ScheduledVelocityCfg(
    entity_name=original.entity_name, resampling_time_range=(1e9, 1e9),
    heading_command=False, heading_control_stiffness=0.5,
    ranges=UniformVelocityCommandCfg.Ranges((-3., 3.), (-2., 2.), (-2., 2.)),
  )
  base = ManagerBasedRlEnv(cfg, device=config.device)
  if suite == "traversal":
    terrain = base.scene.terrain
    names = list(cfg.scene.terrain.terrain_generator.sub_terrains)
    types = torch.tensor([names.index(c["terrain"]) for c in cases], device=config.device)
    levels = torch.tensor([c["level"] for c in cases], device=config.device)
    terrain.terrain_types[:] = types.repeat_interleave(config.trials_per_case)
    terrain.terrain_levels[:] = levels.repeat_interleave(config.trials_per_case)
    terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
  env = RslRlVecEnvWrapper(base, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls("Unitree-Go2-HIM")(env, asdict(agent_cfg), device=config.device)
  return base, env, runner, cfg


def sha_tensor(*values):
  digest = hashlib.sha256()
  for value in values:
    digest.update(value.detach().cpu().numpy().tobytes())
  return digest.hexdigest()


def sample_schedule(term, suite, config, cases, repeat, per_case, dt):
  rng = np.random.default_rng(config.seed + repeat)
  n = len(cases) * per_case
  stages = 4 if suite in ("tracking", "heading_tracking") else 1
  schedule = np.zeros((n, stages, 3), dtype=np.float32)
  headings = np.zeros((n, stages), dtype=np.float32)
  # Share command quantiles between profiles as well as between checkpoints.
  quantiles = rng.random((per_case, stages, 3))
  shared_headings = rng.uniform(-math.pi, math.pi, (per_case, stages))
  limits = np.ones(n, dtype=np.float32)
  for i, case in enumerate(cases):
    ids = slice(i * per_case, (i + 1) * per_case)
    if suite in ("tracking", "heading_tracking"):
      ranges = ranges_at_level(case["profile"])
      schedule[ids] = ranges[:, 0] + quantiles * (ranges[:, 1] - ranges[:, 0])
      headings[ids] = shared_headings
      limits[ids] = ranges[2, 1]
    else:
      schedule[ids, 0] = case["command"]
  term.schedule = torch.as_tensor(schedule, device=term.device)
  term.heading_schedule = torch.as_tensor(headings, device=term.device)
  term.yaw_limits = torch.as_tensor(limits, device=term.device)
  term.use_heading = suite == "heading_tracking"
  term.settle_steps = round(config.settle_seconds / dt)
  term.stage_steps = round(config.tracking_seconds / stages / dt)
  term.active.fill_(True)
  return hashlib.sha256(schedule.tobytes() + headings.tobytes()).hexdigest()


@torch.inference_mode()
def run_trials(base, env, policy, term, suite, cases, config, repeat, per_case, traces):
  dt, n = base.step_dt, env.num_envs
  schedule_sha = sample_schedule(term, suite, config, cases, repeat, per_case, dt)
  values, _ = base.reset(seed=config.seed + 1000 + repeat)
  obs = TensorDict(values, batch_size=[n])
  policy.reset()
  initial_sha = sha_tensor(base.sim.data.qpos, base.sim.data.qvel, obs["actor"])
  robot = base.scene["robot"]
  feet = base.scene["feet_ground_contact"]
  foot_ids = robot.find_sites(("FR", "FL", "RR", "RL"))[0]
  device = base.device
  alive = torch.ones(n, dtype=torch.bool, device=device)
  status = torch.zeros(n, dtype=torch.long, device=device)
  end_time = torch.full((n,), float("nan"), device=device)
  goal = torch.zeros(n, dtype=torch.bool, device=device)
  hit = torch.zeros_like(goal)
  latency = torch.full((n,), float("nan"), device=device)
  startup_survived = torch.ones_like(goal)
  fields = ["xy_error", "yaw_error", "samples", "steady_xy_error", "steady_yaw_error",
            "steady_samples", "stall_s", "max_stall_s", "path_m", "slip_sum",
            "action_change_sum", "touchdowns", "foothold_shift_m", "yaw_angle_rad"]
  stats = {name: torch.zeros(n, device=device) for name in fields}
  stall_run = torch.zeros(n, device=device)
  foot_last = robot.data.site_pos_w[:, foot_ids, :2].clone()
  contact_last = feet.data.found > 0
  heading_last = robot.data.heading_w.clone()
  position_last = robot.data.root_link_pos_w[:, :2].clone()
  term_names = base.termination_manager.active_terms
  term_counts = torch.zeros((n, len(term_names)), dtype=torch.long, device=device)
  duration = {"tracking": config.tracking_seconds, "heading_tracking": config.tracking_seconds,
              "traversal": config.traversal_seconds,
              "maneuver": config.maneuver_seconds}[suite]
  settle_steps, test_steps = round(config.settle_seconds / dt), round(duration / dt)
  command_x = torch.tensor([c.get("command", (0., 0., 0.))[0] for c in cases], device=device)
  direction = command_x.sign().repeat_interleave(per_case)
  trace_ids = torch.arange(0, n, per_case, device=device)

  with torch.inference_mode():
    for step in range(settle_steps + test_steps):
      command = term.command.clone()
      actions = policy(obs)
      if not torch.isfinite(actions).all():
        raise RuntimeError("Non-finite policy output; evaluation aborted without modifying actions")
      obs, _, dones, _ = env.step(actions)
      in_test = step >= settle_steps
      elapsed = (step - settle_steps + 1) * dt
      valid = alive.clone()
      pos = robot.data.root_link_pos_w[:, :2]
      vel = robot.data.root_link_lin_vel_b[:, :2]
      yaw = robot.data.root_link_ang_vel_b[:, 2]
      physical_finite = torch.isfinite(base.sim.data.qpos).all(-1) & torch.isfinite(base.sim.data.qvel).all(-1)
      observation_finite = torch.isfinite(obs["actor"]).reshape(n, -1).all(-1)
      finite = physical_finite & observation_finite & torch.isfinite(vel).all(-1) & torch.isfinite(yaw)
      gravity_z = robot.data.projected_gravity_b[:, 2]
      fallen = gravity_z > -math.cos(math.radians(70.))
      offset = pos - base.scene.env_origins[:, :2]
      outside = (offset.abs() > 3.85).any(-1) if suite == "traversal" else torch.zeros_like(alive)
      failed = valid & ((dones != 0) | ~finite | fallen | outside)
      # The terminal step counts. Non-finite samples are excluded, not repaired.
      usable = valid & finite
      if in_test:
        xy_error = torch.linalg.vector_norm(vel - command[:, :2], dim=-1)
        yaw_error = (yaw - command[:, 2]).abs()
        for key, value in (("xy_error", xy_error), ("yaw_error", yaw_error),
                           ("samples", torch.ones(n, device=device))):
          stats[key] += torch.where(usable, value, 0.)
        if elapsed > 1.0:
          stats["steady_xy_error"] += torch.where(usable, xy_error, 0.)
          stats["steady_yaw_error"] += torch.where(usable, yaw_error, 0.)
          stats["steady_samples"] += usable
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        actual_speed = torch.linalg.vector_norm(vel, dim=-1)
        stalled = ((speed >= 0.3) & (actual_speed < 0.1)) | (
          (speed < 0.1) & (command[:, 2].abs() >= 0.3) & (yaw.abs() < 0.1))
        stalled &= usable
        stats["stall_s"] += stalled * dt
        stall_run = torch.where(stalled, stall_run + dt, 0.)
        stats["max_stall_s"] = torch.maximum(stats["max_stall_s"], stall_run)
        stats["path_m"] += torch.where(usable, torch.linalg.vector_norm(pos - position_last, dim=-1), 0.)
        increment = torch.remainder(robot.data.heading_w - heading_last + math.pi, 2 * math.pi) - math.pi
        stats["yaw_angle_rad"] += torch.where(usable, increment, 0.)
        contact = feet.data.found > 0
        foot_pos = robot.data.site_pos_w[:, foot_ids, :2]
        touchdown = contact & ~contact_last
        shift = torch.linalg.vector_norm(foot_pos - foot_last, dim=-1)
        stats["touchdowns"] += torch.where(usable, touchdown.sum(-1), 0.)
        stats["foothold_shift_m"] += torch.where(usable, (shift * touchdown).sum(-1), 0.)
        foot_last = torch.where(touchdown.unsqueeze(-1), foot_pos, foot_last)
        slip = torch.linalg.vector_norm(robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1)
        stats["slip_sum"] += torch.where(usable, (slip * contact).sum(-1) / contact.sum(-1).clamp_min(1), 0.)
        change = ((base.action_manager.action - base.action_manager.prev_action) ** 2).sum(-1)
        stats["action_change_sum"] += torch.where(usable, change, 0.)
        if elapsed <= 1.0 + 1e-6:
          projection = (vel * command[:, :2]).sum(-1) / speed.clamp_min(1e-6)
          reached = torch.where(speed > 0.1, projection >= speed * 0.5,
                                yaw * command[:, 2].sign() >= command[:, 2].abs() * 0.5)
          new = usable & ~failed & ~hit & reached
          latency[new] = elapsed
          hit |= new
          startup_survived &= ~failed
        if suite == "traversal":
          reached = (offset[:, 0] * direction >= 3.0) & (offset[:, 1].abs() <= 0.75)
          new_goal = usable & ~failed & reached
          goal |= new_goal
          end_time[new_goal] = elapsed
        if step % 5 == 0:
          ids = trace_ids.cpu().tolist()
          packed = torch.cat((command, vel, yaw[:, None], offset, gravity_z[:, None]), dim=1).cpu().tolist()
          for i in ids:
            if valid[i]:
              traces.append([suite, cases[i // per_case]["case"], repeat, i, round(elapsed, 4),
                             *packed[i], int(failed[i]), int(goal[i])])
      for j, name in enumerate(term_names):
        term_counts[:, j] += (base.termination_manager.get_term(name) & failed).long()
      status[failed] = 1
      status[failed & outside] = 2
      status[failed & fallen] = 3
      status[failed & ~finite] = 4
      end_time[failed] = max(0., elapsed)
      alive &= ~failed & ~goal
      term.active.copy_(alive)
      contact_last = (feet.data.found > 0).clone()
      position_last = pos.clone()
      heading_last = robot.data.heading_w.clone()
      reset_ids = ((dones != 0) | ~finite).nonzero(as_tuple=False).flatten()
      if reset_ids.numel():
        values, _ = base.reset(env_ids=reset_ids)
        obs = TensorDict(values, batch_size=[n])
      if step == settle_steps - 1:
        # At this point the first nonzero command is already in obs/history.
        if not torch.equal(obs["actor"][:, -1, 6:9], term.command):
          raise RuntimeError("Command/history alignment failed")
      if not alive.any():
        break
  success = goal if suite == "traversal" else status == 0
  end_time[torch.isnan(end_time)] = duration
  columns = {**stats, "status": status, "goal": goal, "success": success,
             "latency_s": latency, "startup_success": hit & startup_survived,
             "end_time_s": end_time}
  arrays = {name: value.cpu().numpy() for name, value in columns.items()}
  failures = term_counts.cpu().numpy()
  rows = []
  for i in range(n):
    samples = arrays["samples"][i]
    steady = arrays["steady_samples"][i]
    row = {"suite": suite, "case": cases[i // per_case]["case"], "repeat": repeat,
           "env_slot": i, "terrain": cases[i // per_case]["terrain"],
           "level": cases[i // per_case]["level"], "initial_sha256": initial_sha,
           "command_schedule_sha256": schedule_sha}
    for name, values in arrays.items():
      value = float(values[i])
      row[name] = value if math.isfinite(value) else None
    for name in ("xy_error", "yaw_error", "slip_sum", "action_change_sum"):
      row[name] = float(arrays[name][i] / samples) if samples else None
    for name in ("steady_xy_error", "steady_yaw_error"):
      row[name] = float(arrays[name][i] / steady) if steady else None
    row["termination_counts"] = dict(zip(term_names, failures[i].tolist()))
    rows.append(row)
  return rows


def bootstrap_cluster_interval(values, clusters, seed=7):
  """Resample physical env slots; repeated resets are not independent robots."""
  groups = {}
  for value, cluster in zip(values, clusters, strict=True):
    groups.setdefault(cluster, []).append(value)
  means = np.array([np.mean(v) for v in groups.values()])
  rng = np.random.default_rng(seed)
  draws = means[rng.integers(0, len(means), (2000, len(means)))].mean(1)
  return float(means.mean()), *np.quantile(draws, [0.025, 0.975]).tolist()


def write_csv(path, rows):
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def summarize(rows, output):
  metrics = ("success", "startup_success", "xy_error", "yaw_error", "steady_xy_error",
             "steady_yaw_error", "stall_s", "max_stall_s", "path_m", "yaw_angle_rad",
             "touchdowns", "foothold_shift_m", "slip_sum", "action_change_sum", "end_time_s")
  groups = {}
  for row in rows:
    groups.setdefault((row["model"], row["suite"], row["case"]), []).append(row)
  summary = []
  for (model, suite, case), group in groups.items():
    result = {"model": model, "suite": suite, "case": case, "trials": len(group),
              "clusters": len({r["env_slot"] for r in group})}
    for metric in metrics:
      valid = [r for r in group if r[metric] is not None]
      if valid:
        mean, lo, hi = bootstrap_cluster_interval([r[metric] for r in valid], [r["env_slot"] for r in valid])
      else:
        mean = lo = hi = None
      result.update({metric: mean, f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
    for status in range(5):
      result[f"status_{status}_count"] = sum(r["status"] == status for r in group)
    result["route_timeout_count"] = sum(r["success"] == 0 and r["status"] == 0 for r in group)
    result["thigh_termination_count"] = sum(r["termination_counts"].get("illegal_contact_1", 0) for r in group)
    summary.append(result)
  write_csv(output / "summary.csv", summary)
  models = list(dict.fromkeys(r["model"] for r in rows))
  comparison = []
  if len(models) == 2:
    for suite, case in dict.fromkeys((r["suite"], r["case"]) for r in rows):
      first = {(r["repeat"], r["env_slot"]): r for r in groups[(models[0], suite, case)]}
      second = {(r["repeat"], r["env_slot"]): r for r in groups[(models[1], suite, case)]}
      assert first.keys() == second.keys()
      for key in first:
        for field in ("initial_sha256", "command_schedule_sha256"):
          if first[key][field] != second[key][field]:
            raise RuntimeError(f"Unpaired {field} for {suite}/{case}/{key}")
      result = {"suite": suite, "case": case, "first_model": models[0], "second_model": models[1]}
      for metric in metrics:
        valid = [k for k in first if first[k][metric] is not None and second[k][metric] is not None]
        if valid:
          delta, lo, hi = bootstrap_cluster_interval(
            [second[k][metric] - first[k][metric] for k in valid], [k[1] for k in valid])
        else:
          delta = lo = hi = None
        result.update({f"delta_{metric}": delta, f"delta_{metric}_ci_low": lo,
                       f"delta_{metric}_ci_high": hi})
      comparison.append(result)
    write_csv(output / "comparison.csv", comparison)
  return summary


def evaluate(config):
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.utils.os import dump_yaml

  configure_torch_backends()
  if config.repeats < 1 or config.trials_per_case < 1 or config.tracking_trials_per_profile < 1:
    raise ValueError("Use positive trial counts")
  if any(s not in ("tracking", "heading_tracking", "traversal", "maneuver") for s in config.suites):
    raise ValueError("Unknown suite")
  paths = [Path(p).resolve() for p in config.checkpoints]
  if len({p.stem for p in paths}) != len(paths):
    raise ValueError("Checkpoint stems must be unique")
  stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
  output = Path(config.output_dir).resolve() / stamp
  output.mkdir(parents=True)
  manifest = {"config": asdict(config), "status": "running", "created": stamp,
              "python": platform.python_version(), "torch": torch.__version__,
              "checkpoints": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                              for p in paths], "initial_states_paired": False,
              "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "definitions": {
                "tracking": "20 s, four pre-generated 5 s reference segments; body XY and yaw sampled uniformly at each profile range; actual command sequence identical for checkpoints.",
                "heading_tracking": "Same XY reference quantiles; heading targets uniform [-pi,pi], yaw=clip(0.5*heading_error), matching training command mode. Heading targets match, actual feedback yaw can differ.",
                "traversal": "Center-to-edge 3 m world-X route within one 8x8 m patch, lateral corridor +/-0.75 m, finish before 8 s without environment termination, >70 deg tilt, numerical failure or patch exit.",
                "success": "Traversal: goal reached before failure. Other suites: survive whole horizon, not a tracking accuracy guarantee.",
                "startup": "Reach half commanded speed/yaw within first second and survive first second; not sustained tracking.",
                "errors": "Per-trial means over finite samples through first failure/goal. Report success/failure alongside errors; no post-reset samples.",
                "stall": "Planar cmd>=0.3 m/s and actual planar speed<0.1 m/s, or pure yaw cmd>=0.3 rad/s and actual yaw<0.1 rad/s.",
                "statuses": "0=no failure (route can still timeout),1=environment termination,2=patch exit,3=tilt>70deg,4=nonfinite.",
                "confidence": "95% percentile bootstrap over physical environment slots, with all repeated resets in each cluster.",
                "touchdowns": "Raw contact false->true at control steps; contact chatter is included, not a validated gait quality score.",
                "pairing": "Same instantiated physics/randomized parameters, exact qpos/qvel/actor-history checksum and normalized reference samples; heading-feedback yaw may differ because trajectories differ.",
              }}
  manifest_file = output / "manifest.json"
  manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"[EVAL] Output: {output}", flush=True)
  all_rows, all_traces = [], []
  try:
    for suite in config.suites:
      cases = make_cases(suite, config)
      per_case = config.tracking_trials_per_profile if suite in ("tracking", "heading_tracking") else config.trials_per_case
      base, env, runner, cfg = build_env(suite, config, cases, len(cases) * per_case)
      dump_yaml(output / f"env_{suite}.yaml", cfg)
      manifest.setdefault("suites", {})[suite] = {"cases": cases, "num_envs": env.num_envs,
        "step_dt": base.step_dt, "physics_dt": base.physics_dt,
        "events": list(cfg.events), "terminations": base.termination_manager.active_terms,
        "parameter_pairing": "same simulator instance for both checkpoints",
        "terrain": repr(cfg.scene.terrain.terrain_generator)}
      try:
        for path in paths:
          runner.load(str(path), load_cfg={"actor": True}, map_location=config.device)
          policy = runner.get_inference_policy(device=config.device)
          parameter_sha = sha_tensor(*policy.state_dict().values())
          for repeat in range(config.repeats):
            print(f"[EVAL] {suite} {path.stem} repeat={repeat+1}/{config.repeats} envs={env.num_envs}", flush=True)
            traces = []
            rows = run_trials(base, env, policy, base.command_manager.get_term("twist"),
                              suite, cases, config, repeat, per_case, traces)
            for row in rows:
              row["model"] = path.stem
            all_rows.extend(rows)
            all_traces.extend([[path.stem, *r] for r in traces])
            (output / "trials.json").write_text(json.dumps(all_rows, indent=2, allow_nan=False) + "\n")
            print(f"[EVAL] success={np.mean([r['success'] for r in rows]):.3f} numerical_failures={sum(r['status']==4 for r in rows)}", flush=True)
          if parameter_sha != sha_tensor(*policy.state_dict().values()):
            raise RuntimeError("Policy weights or normalization statistics changed during evaluation")
      finally:
        env.close()
    summary = summarize(all_rows, output)
    trace_columns = ["model", "suite", "case", "repeat", "env_slot", "time_s", "cmd_x", "cmd_y", "cmd_yaw",
                     "velocity_x", "velocity_y", "yaw_velocity", "offset_x", "offset_y", "gravity_z", "failed", "goal"]
    write_csv(output / "traces.csv", [dict(zip(trace_columns, r)) for r in all_traces])
    manifest.update(status="complete", initial_states_paired=len(paths)==2,
                    trial_count=len(all_rows), summary_rows=len(summary))
  except Exception as error:
    manifest.update(status="failed", error=repr(error))
    raise
  finally:
    manifest_file.write_text(json.dumps(manifest, indent=2) + "\n")
  print(f"[EVAL] Complete: {output}", flush=True)
  return output


if __name__ == "__main__":
  evaluate(tyro.cli(CapabilityConfig))
