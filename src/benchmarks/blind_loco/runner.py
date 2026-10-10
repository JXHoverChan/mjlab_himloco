"""Headless, inference-only execution of the versioned internal benchmark."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends

from .environment import make_environment_cfg, validate_observation
from .metrics import TrialStatistics
from .policy import load_policy, sha_file, state_sha
from .protocol import (
  PATCH_SIZE, ROUTE_HALF_WIDTH, ROUTE_LENGTH, SETTLE_SECONDS, STEP_DT,
  TILT_LIMIT_DEGREES, case_catalog, digest, protocol_contract, reference_schedule,
  stable_seed, terrain_library,
)
from .report import create_report, write_csv, write_json
from .terrains import height_grid


ROOT = Path(__file__).resolve().parents[3]


def now():
  return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def array_sha(*arrays):
  result = hashlib.sha256()
  for value in arrays:
    value = np.ascontiguousarray(value)
    result.update(str(value.shape).encode())
    result.update(str(value.dtype).encode())
    result.update(value.tobytes())
  return result.hexdigest()


def software_snapshot(device):
  packages = {}
  for name in ("mjlab", "rsl-rl-lib", "torch", "mujoco", "mujoco-warp", "warp-lang", "numpy"):
    try:
      packages[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
      packages[name] = "package metadata unavailable"
  source = {str(path.relative_to(ROOT)): sha_file(path)
            for path in Path(__file__).parent.glob("*.py")}
  for relative in ("scripts/benchmark_blind_loco.py", "src/tasks/velocity/rl/himactor.py",
                   "src/tasks/velocity/rl/himencoder.py"):
    source[relative] = sha_file(ROOT/relative)
  assets = [ROOT/"src/assets/robots/unitree_go2/go2_constants.py"]
  assets += sorted((ROOT/"src/assets/robots/unitree_go2").rglob("*.xml"))
  asset_hashes = {str(path.relative_to(ROOT)): sha_file(path) for path in assets}
  snapshot = {"python": platform.python_version(), "packages": packages,
              "torch_cuda": torch.version.cuda, "device": device,
              "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
              "source_sha256": source, "robot_asset_sha256": asset_hashes}
  # Policy implementation/architecture is allowed to change; environment and evaluator are not.
  comparable = {"python": snapshot["python"], "packages": packages,
                "torch_cuda": snapshot["torch_cuda"], "robot_asset_sha256": asset_hashes,
                "evaluation_source_sha256": {key: value for key, value in source.items()
                    if Path(key).name in ("protocol.py", "terrains.py", "environment.py", "metrics.py", "runner.py")}}
  return snapshot, digest(comparable)


def compiled_physics_sha(model):
  fields = ("body_mass", "body_inertia", "body_ipos", "geom_type", "geom_size", "geom_pos",
            "geom_quat", "geom_friction", "geom_condim", "geom_contype", "geom_conaffinity",
            "geom_solref", "geom_solimp", "jnt_range", "dof_armature", "dof_damping",
            "actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "hfield_size", "hfield_data")
  return array_sha(*(np.asarray(getattr(model, name)) for name in fields))


def prepare_trials(env, cases, trials, repeat, seed, terrains):
  n = len(cases)*trials
  lengths = [round((SETTLE_SECONDS+c.duration)/STEP_DT)+1 for c in cases]
  schedule = np.zeros((n, max(lengths), 3), dtype=np.float32)
  offsets = np.zeros((n, 3), dtype=np.float32)
  pitch = np.zeros(n, dtype=np.float32)
  terrain_by_name = {t.name: t for t in terrains}
  terrain_names = [t.name for t in terrains]
  types = []
  command_hashes = []
  for index, case in enumerate(cases):
    ids = slice(index*trials, (index+1)*trials)
    reference = reference_schedule(case, trials, repeat, seed)
    schedule[ids, :reference.shape[1], :] = reference
    rng = np.random.default_rng(stable_seed(seed, case.name, repeat, "initial"))
    offsets[ids, :2] = rng.uniform(-.025, .025, size=(trials, 2))
    spec = terrain_by_name[case.terrain]
    pitch[ids] = -math.radians(spec.slope_degrees)
    offsets[ids, 2] = .32/math.cos(math.radians(spec.slope_degrees))-.32
    types.extend([terrain_names.index(case.terrain)]*trials)
    command_hashes.extend(array_sha(reference[i]) for i in range(trials))
  terrain = env.scene.terrain
  terrain.terrain_types[:] = torch.tensor(types, device=env.device)
  terrain.terrain_levels[:] = 0
  terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
  env.benchmark_initial_offsets = torch.tensor(offsets, device=env.device)
  env.benchmark_initial_pitch = torch.tensor(pitch, device=env.device)
  command = env.command_manager.get_term("twist")
  command.schedule = torch.tensor(schedule, device=env.device)
  command.active.fill_(True)
  values, _ = env.reset(seed=seed+repeat)
  obs = TensorDict(values, batch_size=[n])
  initial_qpos = env.sim.data.qpos.cpu().numpy()
  initial_qvel = env.sim.data.qvel.cpu().numpy()
  initial_hashes = [array_sha(initial_qpos[i], initial_qvel[i]) for i in range(n)]
  return obs, command_hashes, initial_hashes


@torch.inference_mode()
def evaluate_repeat(env, actor, cases, trials, repeat, seed, terrains, output, clip_actions):
  device, n = env.device, env.num_envs
  obs, command_hashes, initial_hashes = prepare_trials(env, cases, trials, repeat, seed, terrains)
  validate_observation(obs, actor.history_length)
  actor.reset()
  command_term = env.command_manager.get_term("twist")
  robot, feet, thighs = env.scene["robot"], env.scene["feet"], env.scene["thigh"]
  foot_ids, foot_names = robot.find_sites(("FR", "FL", "RR", "RL"), preserve_order=True)
  if tuple(foot_names) != ("FR", "FL", "RR", "RL"):
    raise ValueError(f"Unexpected foot-site order: {foot_names}")
  alive = torch.ones(n, dtype=torch.bool, device=device)
  succeeded = torch.zeros_like(alive)
  failed_ever = torch.zeros_like(alive)
  end_time = torch.zeros(n, device=device)
  failure_flags = {name: torch.zeros_like(alive) for name in ("tilt", "thigh_contact", "patch_exit", "numerical")}
  accumulator = TrialStatistics(n, device)
  durations = np.repeat([c.duration for c in cases], trials)
  stage_counts = np.repeat([len(c.stages) or 1 for c in cases], trials)
  duration_tensor = torch.tensor(durations, device=device)
  mode_route = cases[0].suite == "traversal"
  directions = torch.tensor([math.copysign(1., c.command[0]) for c in cases], device=device).repeat_interleave(trials)
  stage_lengths = torch.tensor([round(c.stage_seconds/STEP_DT) if c.stages else 10**9 for c in cases], device=device).repeat_interleave(trials)
  previous_pos = robot.data.root_link_pos_w[:, :2].clone()
  initial_pos = previous_pos.clone()
  previous_heading = robot.data.heading_w.clone()
  contact_names = [name.split("/")[-1] for name in feet.primary_names]
  contact_order = [contact_names.index(f"{leg}_foot_collision") for leg in foot_names]
  previous_contact = (feet.data.found > 0).reshape(n, 4)[:, contact_order].clone()
  last_foothold = robot.data.site_pos_w[:, foot_ids, :2].clone()
  settle_steps = round(SETTLE_SECONDS/STEP_DT)
  max_steps = round((SETTLE_SECONDS+max(durations))/STEP_DT)
  trace_indices = torch.arange(0, n, trials, device=device)
  traces = []
  failure_events = []
  started = time.perf_counter()

  for step in range(max_steps):
    if step == settle_steps:
      initial_pos = robot.data.root_link_pos_w[:, :2].clone()
    valid = alive.clone()
    command = command_term.command.clone()
    if step in (0, settle_steps) or (cases[0].suite == "transitions" and (step-settle_steps) % 150 == 0):
      if not torch.equal(obs["actor"][valid, -1, 6:9], command[valid]):
        raise RuntimeError("Reference/history misalignment: the action did not receive the measured command")
    actions = actor(obs)
    if not torch.isfinite(actions[valid]).all():
      np.savez_compressed(output/f"policy_nonfinite_{cases[0].suite}_r{repeat}_k{step}.npz",
                          obs=obs["actor"].cpu().numpy(), action=actions.cpu().numpy())
      raise FloatingPointError("Active policy produced NaN/Inf; raw state dumped without modifying actions")
    if clip_actions is not None:
      actions = actions.clamp(-clip_actions, clip_actions)
    # Finished/failed trials still occupy simulator slots, but never receive or contribute task scores.
    actions = torch.where(valid[:, None], actions, torch.zeros_like(actions))
    values, _, _, _, _ = env.step(actions)
    obs = TensorDict(values, batch_size=[n])
    elapsed = max(0., (step-settle_steps+1)*STEP_DT)
    position = robot.data.root_link_pos_w[:, :2]
    velocity = robot.data.root_link_lin_vel_b[:, :2]
    yaw = robot.data.root_link_ang_vel_b[:, 2]
    physical_finite = torch.isfinite(env.sim.data.qpos).all(-1) & torch.isfinite(env.sim.data.qvel).all(-1)
    observation_finite = torch.isfinite(obs["actor"]).reshape(n, -1).all(-1)
    finite = physical_finite & observation_finite & torch.isfinite(velocity).all(-1) & torch.isfinite(yaw)
    tilt = robot.data.projected_gravity_b[:, 2] > -math.cos(math.radians(TILT_LIMIT_DEGREES))
    thigh_force = torch.linalg.vector_norm(thighs.data.force, dim=-1).reshape(n, -1).max(-1).values
    contact_failure = thigh_force > 1.
    offset = position - env.scene.env_origins[:, :2]
    patch_exit = (offset.abs() > PATCH_SIZE/2-.5).any(-1)
    fail = valid & (tilt | contact_failure | patch_exit | ~finite)
    for name, event in (("tilt", tilt), ("thigh_contact", contact_failure),
                        ("patch_exit", patch_exit), ("numerical", ~finite)):
      failure_flags[name] |= fail & event
    if fail.any():
      ids = fail.nonzero().flatten().cpu().tolist()
      for i in ids:
        failure_events.append({"case": cases[i//trials].name, "repeat": repeat, "trial": i % trials,
                               "step": step, "time_s": elapsed,
                               "events": [name for name, mask in failure_flags.items() if bool(mask[i])]})
      if (valid & ~finite).any():
        bad = (valid & ~finite).nonzero().flatten()
        np.savez_compressed(output/f"physics_nonfinite_{cases[0].suite}_r{repeat}_k{step}.npz",
                            ids=bad.cpu().numpy(), qpos=env.sim.data.qpos[bad].cpu().numpy(),
                            qvel=env.sim.data.qvel[bad].cpu().numpy(), action=actions[bad].cpu().numpy())
    usable = valid & finite
    goal = torch.zeros_like(alive)
    in_test = step >= settle_steps
    heading_increment = torch.remainder(robot.data.heading_w-previous_heading+math.pi, 2*math.pi)-math.pi
    contact = (feet.data.found > 0).reshape(n, 4)[:, contact_order]
    foot_position = robot.data.site_pos_w[:, foot_ids, :2]
    touchdown = contact & ~previous_contact
    shifts = torch.linalg.vector_norm(foot_position-last_foothold, dim=-1)
    if in_test:
      stage = ((step-settle_steps)//stage_lengths).clamp_max(4)
      stage_elapsed = ((step-settle_steps) % stage_lengths + 1)*STEP_DT
      stage_elapsed = torch.minimum(stage_elapsed, torch.full_like(stage_elapsed, elapsed))
      steady = stage_elapsed > 1.
      torque = robot.data.qfrc_actuator
      power = (torque*robot.data.joint_vel).abs().sum(-1)
      limits = robot.data.soft_joint_pos_limits
      violations = ((robot.data.joint_pos < limits[..., 0]) | (robot.data.joint_pos > limits[..., 1])).float().mean(-1)
      accumulator.update(
        velocity=velocity, yaw=yaw, command=command, usable=usable, failed=fail, steady=steady,
        stage=stage, stage_elapsed=stage_elapsed,
        path_increment=torch.linalg.vector_norm(position-previous_pos, dim=-1),
        yaw_increment=heading_increment, contact=contact,
        slip=torch.linalg.vector_norm(robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1),
        power=power, vertical_velocity=robot.data.root_link_lin_vel_w[:, 2],
        action_change=(env.action_manager.action-env.action_manager.prev_action).square().sum(-1),
        soft_limits=violations, touchdowns=touchdown.sum(-1).float(),
        foothold_shift=(shifts*touchdown).sum(-1),
      )
      if mode_route:
        displacement = position-initial_pos
        goal = usable & ~fail & (displacement[:, 0]*directions >= ROUTE_LENGTH) & (displacement[:, 1].abs() <= ROUTE_HALF_WIDTH)
      if step % 10 == 0:
        packed = torch.cat((command, velocity, yaw[:, None], offset,
                            robot.data.root_link_pos_w[:, 2:3], thigh_force[:, None]), dim=-1)[trace_indices].cpu().tolist()
        trace_alive = valid[trace_indices].cpu().tolist()
        for j, i in enumerate(trace_indices.cpu().tolist()):
          if trace_alive[j]:
            traces.append({"suite": cases[0].suite, "case": cases[i//trials].name,
                           "repeat": repeat, "trial": i % trials, "time_s": elapsed,
                           **dict(zip(("cmd_vx", "cmd_vy", "cmd_yaw", "vx", "vy", "yaw", "x", "y", "z", "thigh_force_n"), packed[j]))})
    deadline = valid & (elapsed >= duration_tensor-1e-7) if in_test else torch.zeros_like(alive)
    finished = fail | goal | deadline
    end_time[finished] = elapsed
    succeeded |= goal if mode_route else deadline & ~fail
    failed_ever |= fail
    alive &= ~finished
    command_term.active.copy_(alive)
    last_foothold = torch.where(touchdown[:, :, None], foot_position, last_foothold)
    previous_contact = contact.clone()
    previous_pos = position.clone()
    previous_heading = robot.data.heading_w.clone()
    # Reset inactive slots only for vector-simulator safety; never count their new samples.
    reset_ids = (finished | (~finite & ~alive)).nonzero().flatten()
    if reset_ids.numel():
      values, _ = env.reset(env_ids=reset_ids)
      obs = TensorDict(values, batch_size=[n])
    if (step+1) % 250 == 0:
      print(f"[BENCH] {cases[0].suite} repeat={repeat+1} step={step+1}/{max_steps} "
            f"active={int(alive.sum())} elapsed_wall={time.perf_counter()-started:.1f}s", flush=True)
    if not alive.any():
      break
  if alive.any():
    raise RuntimeError("Trial horizon accounting left unfinished environments")
  rows = accumulator.export(durations, stage_counts)
  terrain_by_name = {t.name: t for t in terrains}
  geometry_hashes = {t.name: array_sha(height_grid(t)) for t in terrains}
  result_flags = {name: mask.cpu().tolist() for name, mask in failure_flags.items()}
  success = succeeded.cpu().tolist()
  survival = (~failed_ever).cpu().tolist()
  ending = end_time.cpu().tolist()
  for i, row in enumerate(rows):
    case, terrain = cases[i//trials], terrain_by_name[cases[i//trials].terrain]
    row.update({"suite": case.suite, "case": case.name, "terrain": case.terrain,
                "terrain_kind": terrain.kind, "mode": case.mode, "range_scale": case.range_scale,
                "repeat": repeat, "trial": i % trials, "duration_s": case.duration,
                "success": success[i], "survival": survival[i], "end_time_s": ending[i],
                "initial_sha256": initial_hashes[i], "command_sha256": command_hashes[i],
                "geometry_sha256": geometry_hashes[terrain.name],
                **{f"failure_{name}": values[i] for name, values in result_flags.items()}})
  return rows, traces, failure_events


def run_benchmark(checkpoint, output, profile="standard", device="cuda:0",
                  suites=("tracking", "maneuver", "transitions", "traversal"),
                  trials=None, repeats=None, seed=20261009):
  output = Path(output)
  trials = trials if trials is not None else {"standard": 32, "quick": 8, "smoke": 2}[profile]
  repeats = repeats if repeats is not None else (3 if profile == "standard" else 1)
  if trials < 1 or repeats < 1:
    raise ValueError("Trial and repeat counts must be positive")
  configure_torch_backends()
  actor, history_length, checkpoint_info = load_policy(checkpoint, device)
  original_state = state_sha(actor)
  snapshot, execution_sha = software_snapshot(device)
  contract = protocol_contract(profile, suites, trials, repeats, seed)
  manifest = {"status": "running", "started": now(), "checkpoint": checkpoint_info,
              "protocol": contract, "protocol_sha256": digest(contract),
              "execution_contract_sha256": execution_sha, "software": snapshot,
              "command": (output/"command.json").read_text() if (output/"command.json").exists() else None,
              "suites": {}, "policy_unchanged": None}
  write_json(output/"manifest.json", manifest)
  write_json(output/"protocol.json", contract)
  # These files are currently uncommitted. Hashes alone cannot recover a past
  # evaluator, so preserve the exact sources and checkpoint architecture YAML.
  for relative in snapshot["source_sha256"] | snapshot["robot_asset_sha256"]:
    destination = output/"source_snapshot"/relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT/relative, destination)
  shutil.copyfile(checkpoint_info["agent_config"], output/"agent_config.yaml")
  print(f"[BENCH] output={output} checkpoint={checkpoint_info['path']} history={history_length}", flush=True)
  rows, traces, failures = [], [], []
  all_terrains = terrain_library()
  geometry_record = {t.name: {**asdict(t), "height_grid_sha256": array_sha(height_grid(t)),
                              "grid_spacing_m": .1} for t in all_terrains}
  write_json(output/"terrain_geometry.json", geometry_record)
  try:
    for suite in suites:
      cases = tuple(c for c in case_catalog(suites, profile) if c.suite == suite)
      used = {c.terrain for c in cases}
      terrains = tuple(t for t in all_terrains if t.name in used)
      cfg = make_environment_cfg(terrains, len(cases)*trials, history_length, seed)
      dump_yaml(output/f"env_{suite}.yaml", cfg)
      print(f"[BENCH] build {suite}: cases={len(cases)} envs={cfg.scene.num_envs}", flush=True)
      env = ManagerBasedRlEnv(cfg, device=device)
      try:
        if abs(env.step_dt-STEP_DT) > 1e-12:
          raise RuntimeError("Unexpected control timestep")
        manifest["suites"][suite] = {"cases": len(cases), "num_envs": env.num_envs,
                                       "compiled_physics_sha256": compiled_physics_sha(env.sim.mj_model)}
        for repeat in range(repeats):
          results, trajectory, events = evaluate_repeat(
            env, actor, cases, trials, repeat, seed, terrains, output, checkpoint_info["clip_actions"])
          rows.extend(results)
          traces.extend(trajectory)
          failures.extend(events)
          write_json(output/"trials.json", rows)
          write_csv(output/"trials.csv", rows)
          write_json(output/"failure_events.json", failures)
          write_csv(output/"traces.csv", traces)
          print(f"[BENCH] done {suite} repeat={repeat+1}: trials={len(results)} "
                f"success={np.mean([r['success'] for r in results]):.3f} "
                f"numerical={sum(r['failure_numerical'] for r in results)}", flush=True)
        write_json(output/"manifest.json", manifest)
      finally:
        env.close()
    manifest["policy_unchanged"] = state_sha(actor) == original_state
    if not manifest["policy_unchanged"]:
      raise RuntimeError("Policy parameters or normalization statistics changed")
    if sha_file(Path(checkpoint)) != checkpoint_info["sha256"]:
      raise RuntimeError("Checkpoint changed while being evaluated")
    expected = len(case_catalog(suites, profile))*trials*repeats
    if len(rows) != expected:
      raise RuntimeError(f"Incomplete trial catalog: {len(rows)} != {expected}")
    manifest.update(status="complete", finished=now(), trials=len(rows), cases=expected//trials//repeats)
    summaries, overview = create_report(output, rows, manifest)
    manifest["overview"] = overview
    manifest["evaluation_summary"] = {
      "takeaway": "Current policy capability baseline recorded under the fixed internal protocol",
      "claim_update": "neutral", "baseline_relation": "not_comparable", "comparability": "high",
      "failure_mode": "environment" if any(r["failure_numerical"] for r in rows) else "none",
      "next_action": "analysis_campaign",
    }
    write_json(output/"manifest.json", manifest)
    write_json(output/"artifact_manifest.json", {str(p.relative_to(output)): sha_file(p) for p in output.rglob("*")
                                               if p.is_file() and p.name not in ("benchmark.log", "artifact_manifest.json")})
    print(f"[BENCH] COMPLETE cases={len(summaries)} trials={len(rows)} report={output/'report.md'}", flush=True)
    return manifest
  except BaseException as exc:
    manifest.update(status="failed", finished=now(), error=f"{type(exc).__name__}: {exc}", partial_trials=len(rows))
    write_json(output/"manifest.json", manifest)
    raise
