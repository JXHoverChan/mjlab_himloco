"""Run a controlled action-rate reward ablation and summarize diagnostics.

Each variant is trained from the same seed and evaluated with the same separate
diagnostic seed. Only ``rewards/action_rate_l2`` is changed between variants.
The script is intentionally a subprocess orchestrator: the training algorithm
and the play diagnostics remain in their normal entry points.
"""

import csv
import json
import math
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import tyro


@dataclass
class AblationConfig:
  task: str = "Unitree-Go2-HIM"
  """Registered task name passed to train.py and play.py."""
  weights_csv: str = "-0.05,-0.01,-0.005,0.0"
  """Comma-separated action_rate_l2 weights, including the current Go2 baseline."""
  seed: int = 42
  """Training seed shared by all variants."""
  eval_seed: int = 2026
  """Evaluation environment seed shared by all variants."""
  max_iterations: int = 1000
  """Training iterations per variant; use a larger value for the final run."""
  train_num_envs: int = 1024
  """Training environments per variant; conservative for a 15 GiB GPU."""
  eval_steps: int = 1000
  """Number of fixed-seed diagnostic environment steps per variant."""
  eval_num_envs: int = 32
  """Parallel evaluation environments, providing repeated startup trials."""
  output_dir: str = "artifacts/experiment/action_rate_ablation"
  """Parent directory for the durable experiment record."""
  device: str | None = None
  """Optional play device, for example ``cuda:0`` or ``cpu``."""
  dry_run: bool = False
  """Write the manifest and commands without launching train or play."""
  continue_on_error: bool = False
  """Continue other variants after one variant fails."""


_MODEL_PATTERN = re.compile(r"^model_(\d+)\.pt$")

_METRIC_DEFINITIONS = {
  "mean_velocity_error_xy": (
    "Mean planar body-frame velocity command error over all diagnostic rows."
  ),
  "first_second_progress_mean": (
    "Mean positive command-aligned planar speed integrated over each episode's "
    "first one second, only for rows with planar command speed > 0.1 m/s."
  ),
  "startup_success_rate": (
    "Fraction of evaluated episodes with a row in the first second whose "
    "command-aligned speed reaches 50% of the current planar command speed."
  ),
  "startup_latency_mean_s": (
    "Mean first time in each successful episode at which command-aligned speed "
    "reaches 50% of command speed; unsuccessful episodes are excluded."
  ),
  "mean_action_rate_raw": (
    "Mean sum((action - previous_action)^2) reported by play diagnostics."
  ),
  "mean_action_rate_reward_rate": (
    "Mean unscaled weighted action_rate_l2 reward rate, before environment dt "
    "scaling."
  ),
}


def _parse_weights(value: str) -> list[float]:
  weights = [float(part.strip()) for part in value.split(",") if part.strip()]
  if not weights:
    raise ValueError("weights_csv must contain at least one numeric weight")
  if len(set(weights)) != len(weights):
    raise ValueError("weights_csv must not contain duplicate weights")
  return weights


def _weight_slug(weight: float) -> str:
  if weight == 0.0:
    return "zero"
  sign = "minus" if weight < 0 else "plus"
  magnitude = f"{abs(weight):.8g}".replace(".", "p")
  return f"{sign}_{magnitude}"


def _command_string(command: list[str]) -> str:
  return shlex.join(command)


def _subprocess_env(repo_root: Path) -> dict[str, str]:
  environment = os.environ.copy()
  current_pythonpath = environment.get("PYTHONPATH", "")
  environment["PYTHONPATH"] = str(repo_root) + (
    os.pathsep + current_pythonpath if current_pythonpath else ""
  )
  environment["PYTHONUNBUFFERED"] = "1"
  return environment


def _run_logged(
  command: list[str], repo_root: Path, log_file: Path, environment: dict[str, str]
) -> None:
  log_file.parent.mkdir(parents=True, exist_ok=True)
  with log_file.open("w") as output:
    result = subprocess.run(
      command,
      cwd=repo_root,
      env=environment,
      stdout=output,
      stderr=subprocess.STDOUT,
      check=False,
      text=True,
    )
  if result.returncode != 0:
    raise RuntimeError(
      f"Command failed with exit code {result.returncode}; see {log_file}"
    )


def _find_latest_checkpoint(log_root: Path) -> Path:
  candidates: list[tuple[int, float, Path]] = []
  for path in log_root.rglob("model_*.pt"):
    match = _MODEL_PATTERN.match(path.name)
    if match is None:
      continue
    candidates.append((int(match.group(1)), path.stat().st_mtime, path))
  if not candidates:
    raise FileNotFoundError(f"No model_*.pt checkpoint found below {log_root}")
  candidates.sort(key=lambda item: (item[0], item[1]))
  return candidates[-1][2]


def _build_eval_command(
  config: AblationConfig, play_script: Path, checkpoint: str | Path, diagnostics_csv: Path
) -> list[str]:
  command = [
    sys.executable,
    str(play_script),
    config.task,
    "--checkpoint-file",
    str(checkpoint),
    "--num-envs",
    str(config.eval_num_envs),
    "--diagnose",
    "True",
    "--diagnose-headless",
    "True",
    "--diagnose-seed",
    str(config.eval_seed),
    "--diagnose-steps",
    str(config.eval_steps),
    "--diagnose-file",
    str(diagnostics_csv),
    "--viewer",
    "viser",
  ]
  if config.device is not None:
    command.extend(["--device", config.device])
  return command


def _float_or_none(value: str) -> float | None:
  if value == "" or value.lower() in {"nan", "none"}:
    return None
  return float(value)


def _summarize_diagnostic_csv(
  csv_file: Path, diagnostic_summary: dict[str, Any]
) -> dict[str, Any]:
  rows: list[dict[str, float | int]] = []
  with csv_file.open(newline="") as source:
    for raw_row in csv.DictReader(source):
      row: dict[str, float | int] = {}
      for key, value in raw_row.items():
        if key in {"step", "env_id", "episode_id", "episode_step", "done"}:
          row[key] = int(float(value))
        else:
          parsed = _float_or_none(value)
          if parsed is not None:
            row[key] = parsed
      rows.append(row)

  if not rows:
    raise ValueError(f"Diagnostic CSV is empty: {csv_file}")

  velocity_errors = [
    float(row["velocity_error_xy"])
    for row in rows
    if "velocity_error_xy" in row
  ]
  episode_rows: dict[tuple[int, int], list[dict[str, float | int]]] = {}
  for row in rows:
    key = (int(row["env_id"]), int(row["episode_id"]))
    episode_rows.setdefault(key, []).append(row)

  first_second_progress: list[float] = []
  startup_successes = 0
  startup_latencies: list[float] = []
  episodes_with_command = 0
  for episode in episode_rows.values():
    first_second: list[tuple[float, float, float]] = []
    for row in episode:
      command_speed = math.hypot(
        float(row.get("command_x", 0.0)), float(row.get("command_y", 0.0))
      )
      step_dt = float(row.get("step_dt", 0.0))
      episode_time = int(row["episode_step"]) * step_dt
      if command_speed <= 0.1 or episode_time > 1.0:
        continue
      projected_speed = (
        float(row.get("velocity_x", 0.0)) * float(row.get("command_x", 0.0))
        + float(row.get("velocity_y", 0.0)) * float(row.get("command_y", 0.0))
      ) / command_speed
      first_second.append((episode_time, command_speed, max(projected_speed, 0.0)))

    if not first_second:
      continue
    episodes_with_command += 1
    first_second_progress.append(
      sum(projected * float(row.get("step_dt", 0.0)) for _, _, projected in first_second)
    )
    success_times = [
      episode_time
      for episode_time, command_speed, projected_speed in first_second
      if projected_speed >= 0.5 * command_speed
    ]
    if success_times:
      startup_successes += 1
      startup_latencies.append(min(success_times))

  termination_counts = {
    key: int(value) for key, value in diagnostic_summary.get("termination_counts", {}).items()
  }
  termination_events = sum(termination_counts.values())
  return {
    "rows": len(rows),
    "episodes_completed": int(diagnostic_summary.get("episodes_completed", 0)),
    "termination_events": termination_events,
    "termination_counts": termination_counts,
    "mean_velocity_error_xy": (
      sum(velocity_errors) / len(velocity_errors) if velocity_errors else None
    ),
    "first_second_progress_mean": (
      sum(first_second_progress) / len(first_second_progress)
      if first_second_progress
      else None
    ),
    "startup_success_rate": (
      startup_successes / episodes_with_command if episodes_with_command else None
    ),
    "startup_latency_mean_s": (
      sum(startup_latencies) / len(startup_latencies)
      if startup_latencies
      else None
    ),
    "episodes_with_active_command": episodes_with_command,
    "mean_action_rate_raw": diagnostic_summary.get("mean_action_rate_raw"),
    "mean_action_rate_reward_rate": diagnostic_summary.get(
      "mean_action_rate_reward_rate"
    ),
  }


def _write_json(path: Path, value: Any) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_summary_csv(path: Path, records: list[dict[str, Any]]) -> None:
  metric_names = [
    "rows",
    "episodes_completed",
    "termination_events",
    "mean_velocity_error_xy",
    "first_second_progress_mean",
    "startup_success_rate",
    "startup_latency_mean_s",
    "episodes_with_active_command",
    "mean_action_rate_raw",
    "mean_action_rate_reward_rate",
  ]
  fieldnames = [
    "variant",
    "weight",
    "status",
    "checkpoint",
    "diagnostics_csv",
    "error",
    *metric_names,
  ]
  with path.open("w", newline="") as output:
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
      metrics = record.get("metrics", {})
      writer.writerow(
        {
          "variant": record.get("variant"),
          "weight": record.get("weight"),
          "status": record.get("status"),
          "checkpoint": record.get("checkpoint"),
          "diagnostics_csv": record.get("diagnostics_csv"),
          "error": record.get("error"),
          **{name: metrics.get(name) for name in metric_names},
        }
      )


def _persist(run_dir: Path, manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
  manifest["variants"] = records
  _write_json(run_dir / "manifest.json", manifest)
  _write_json(
    run_dir / "summary.json",
    {"metric_definitions": _METRIC_DEFINITIONS, "variants": records},
  )
  _write_summary_csv(run_dir / "summary.csv", records)


def run_ablation(config: AblationConfig) -> Path:
  if config.max_iterations <= 0:
    raise ValueError("max_iterations must be positive")
  if config.train_num_envs <= 0:
    raise ValueError("train_num_envs must be positive")
  if config.eval_steps <= 0:
    raise ValueError("eval_steps must be positive")
  if config.eval_num_envs <= 0:
    raise ValueError("eval_num_envs must be positive")

  weights = _parse_weights(config.weights_csv)
  repo_root = Path(__file__).resolve().parents[1]
  timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
  run_dir = (Path(config.output_dir).resolve() / timestamp)
  run_dir.mkdir(parents=True, exist_ok=False)
  environment = _subprocess_env(repo_root)
  train_script = repo_root / "scripts" / "train.py"
  play_script = repo_root / "scripts" / "play.py"

  manifest: dict[str, Any] = {
    "status": "planned" if config.dry_run else "running",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "question": (
      "Does the action_rate_l2 reward weight change startup behavior and local "
      "stalling under otherwise identical training and evaluation seeds?"
    ),
    "hypothesis": (
      "A larger penalty magnitude may suppress rapid action changes, increasing "
      "startup latency or reducing first-second command-aligned progress."
    ),
    "alternative": (
      "Startup stalls may be unrelated to action_rate_l2 and remain similar "
      "across weights."
    ),
    "task": config.task,
    "training_seed": config.seed,
    "evaluation_seed": config.eval_seed,
    "max_iterations": config.max_iterations,
    "training_num_envs": config.train_num_envs,
    "evaluation_steps": config.eval_steps,
    "evaluation_num_envs": config.eval_num_envs,
    "weights": weights,
    "metric_definitions": _METRIC_DEFINITIONS,
    "environment": {
      "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES"),
      "python": sys.executable,
      "repo_root": str(repo_root),
    },
    "variants": [],
  }
  records: list[dict[str, Any]] = []
  _persist(run_dir, manifest, records)

  for weight in weights:
    variant = _weight_slug(weight)
    variant_dir = run_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=False)
    train_log_root = variant_dir / "train_logs"
    train_log = variant_dir / "train.log"
    eval_log = variant_dir / "eval.log"
    diagnostics_csv = variant_dir / "diagnostics.csv"
    train_command = [
      sys.executable,
      str(train_script),
      config.task,
      "--log-root",
      str(train_log_root),
      "--action-rate-weight",
      repr(weight),
      "--agent.seed",
      str(config.seed),
      "--agent.max-iterations",
      str(config.max_iterations),
      "--env.scene.num-envs",
      str(config.train_num_envs),
      "--agent.run-name",
      variant,
    ]
    eval_command_template = _build_eval_command(
      config, play_script, "<latest-checkpoint>", diagnostics_csv
    )
    record: dict[str, Any] = {
      "variant": variant,
      "weight": weight,
      "status": "planned" if config.dry_run else "running",
      "train_command": train_command,
      "train_command_string": _command_string(train_command),
      "train_log": str(train_log),
      "eval_log": str(eval_log),
      "diagnostics_csv": str(diagnostics_csv),
      "eval_command_template": eval_command_template,
      "eval_command_template_string": _command_string(eval_command_template),
    }
    records.append(record)
    _persist(run_dir, manifest, records)

    if config.dry_run:
      continue

    try:
      _run_logged(train_command, repo_root, train_log, environment)
      checkpoint = _find_latest_checkpoint(train_log_root)
      eval_command = _build_eval_command(
        config, play_script, checkpoint, diagnostics_csv
      )
      record["checkpoint"] = str(checkpoint)
      record["eval_command"] = eval_command
      record["eval_command_string"] = _command_string(eval_command)
      _persist(run_dir, manifest, records)

      _run_logged(eval_command, repo_root, eval_log, environment)
      diagnostic_summary_file = diagnostics_csv.with_suffix(".summary.json")
      diagnostic_summary = json.loads(diagnostic_summary_file.read_text())
      record["metrics"] = _summarize_diagnostic_csv(
        diagnostics_csv, diagnostic_summary
      )
      record["status"] = "complete"
    except Exception as error:
      record["status"] = "failed"
      record["error"] = str(error)
      _persist(run_dir, manifest, records)
      if not config.continue_on_error:
        manifest["status"] = "failed"
        _persist(run_dir, manifest, records)
        raise
    _persist(run_dir, manifest, records)

  manifest["status"] = "planned" if config.dry_run else "complete"
  _persist(run_dir, manifest, records)
  return run_dir


def main() -> None:
  config = tyro.cli(AblationConfig)
  run_dir = run_ablation(config)
  print(f"[INFO] Ablation record: {run_dir}")
  if config.dry_run:
    print("[INFO] Dry run complete; no training or evaluation process was started.")


if __name__ == "__main__":
  main()
