"""Screen HIM reward variants by short continuations from one checkpoint."""

import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import tyro

from scripts.run_action_rate_ablation import (
  _find_latest_checkpoint,
  _run_logged,
  _subprocess_env,
)
from src.tasks.velocity.mdp.reward_profiles import RewardProfile


@dataclass
class RewardAblationConfig:
  checkpoint_file: str
  task: str = "Unitree-Go2-HIM"
  profiles: tuple[RewardProfile, ...] = ("legacy", "standing_pose", "revised")
  train_seeds: tuple[int, ...] = (42,)
  iterations: int = 250
  train_num_envs: int = 1024
  eval_num_envs: int = 64
  eval_repeats: int = 3
  eval_seed: int = 2026
  commands: str = "1,0,0;2,0,0;0,0.75,0;0,0,1"
  terrain: str = "flat"
  """Fixed evaluation surface: flat or rough. Training keeps the task's rough terrain."""
  device: str = "cuda:0"
  gpu_id: int = 0
  freeze_curriculum: bool = True
  output_dir: str = "artifacts/experiment/reward_ablation"
  dry_run: bool = False
  """Write exact commands and budget without running training or evaluation."""


def _persist(path: Path, manifest: dict) -> None:
  (path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
  rows = []
  for record in manifest["runs"]:
    for case in record.get("evaluation", {}).get("cases", []):
      rows.append({
        "variant": record["variant"], "seed": record.get("seed"),
        "command": str(case["command"]), "trials": case["trials"],
        **{name: case[name] for name in (
          "startup_success_rate", "survival_rate", "startup_latency_s",
          "first_second_progress_m", "mean_velocity_error_xy",
          "mean_velocity_error_yaw", "mean_action_rate_raw",
        )},
      })
  if rows:
    with (path / "summary.csv").open("w", newline="") as output:
      writer = csv.DictWriter(output, fieldnames=list(rows[0]))
      writer.writeheader()
      writer.writerows(rows)


def run(config: RewardAblationConfig) -> Path:
  from scripts.evaluate_startup import parse_commands

  checkpoint = Path(config.checkpoint_file).expanduser().resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)
  if min(config.iterations, config.train_num_envs, config.eval_num_envs, config.eval_repeats) < 1:
    raise ValueError("Iteration, environment and repeat counts must be positive")
  if not config.profiles or not config.train_seeds:
    raise ValueError("Provide at least one profile and training seed")
  if len(set(config.profiles)) != len(config.profiles) or len(set(config.train_seeds)) != len(config.train_seeds):
    raise ValueError("Profiles and training seeds must be unique")
  if config.terrain not in ("flat", "rough"):
    raise ValueError("terrain must be flat or rough")
  parse_commands(config.commands)
  root = Path(__file__).resolve().parents[1]
  now = datetime.now(ZoneInfo("Asia/Shanghai"))
  directory = Path(config.output_dir).expanduser().resolve() / now.strftime("%Y-%m-%d_%H-%M-%S_%f")
  directory.mkdir(parents=True, exist_ok=False)
  with checkpoint.open("rb") as source:
    digest = _hash_stream(source)
  # Read the actual runner rollout length when computing the sample budget.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import load_rl_cfg

  rollout = load_rl_cfg(config.task).num_steps_per_env
  manifest = {
    "status": "planned" if config.dry_run else "running",
    "created_at": now.isoformat(), "config": asdict(config),
    "checkpoint": str(checkpoint), "checkpoint_sha256": digest,
    "transitions_per_variant": config.iterations * config.train_num_envs * rollout,
    "comparison": "Short continuation from identical model and optimizer state; measures adaptation of this checkpoint, not learning from scratch.",
    "runs": [],
  }
  environment = _subprocess_env(root)

  def eval_command(model, profile, output):
    return [sys.executable, str(root / "scripts/evaluate_startup.py"),
      "--checkpoint-file", str(model), "--task", config.task,
      "--reward-profile", profile, "--num-envs", str(config.eval_num_envs),
      "--repeats", str(config.eval_repeats), "--seed", str(config.eval_seed),
      "--commands", config.commands, "--terrain", config.terrain,
      "--device", config.device, "--output-file", str(output)]

  baseline = {
    "variant": "checkpoint_before_training", "status": "planned",
    "eval_command": eval_command(checkpoint, "legacy", directory / "before.json"),
  }
  manifest["runs"].append(baseline)
  for seed in config.train_seeds:
    for profile in config.profiles:
      label = f"{profile}_seed{seed}"
      variant_dir = directory / label
      variant_dir.mkdir()
      command = [sys.executable, str(root / "scripts/train.py"), config.task,
        "--checkpoint-file", str(checkpoint), "--reward-profile", profile,
        "--freeze-curriculum", str(config.freeze_curriculum),
        "--log-root", str(variant_dir / "train_logs"),
        "--agent.seed", str(seed), "--agent.max-iterations", str(config.iterations),
        "--agent.save-interval", str(config.iterations + 1),
        "--agent.logger", "tensorboard", "--agent.run-name", label,
        "--env.scene.num-envs", str(config.train_num_envs),
        "--gpu-ids", repr([config.gpu_id])]
      manifest["runs"].append({
        "variant": label, "profile": profile, "seed": seed, "status": "planned",
        "train_command": command, "train_log": str(variant_dir / "train.log"),
        "eval_command_template": eval_command("<trained-checkpoint>", profile, variant_dir / "evaluation.json"),
      })
  _persist(directory, manifest)
  print(f"[INFO] {len(config.profiles) * len(config.train_seeds)} branches, {manifest['transitions_per_variant']:,} transitions per branch", flush=True)
  if config.dry_run:
    return directory
  try:
    for record in manifest["runs"]:
      record["status"] = "running"
      _persist(directory, manifest)
      if "train_command" in record:
        print(f"[INFO] Continuing {record['variant']}", flush=True)
        _run_logged(record["train_command"], root, Path(record["train_log"]), environment)
        variant_dir = Path(record["train_log"]).parent
        model = _find_latest_checkpoint(variant_dir / "train_logs")
        record["checkpoint"] = str(model)
        record["eval_command"] = eval_command(model, record["profile"], variant_dir / "evaluation.json")
      output = Path(record["eval_command"][-1])
      _run_logged(record["eval_command"], root, output.with_suffix(".log"), environment)
      record["evaluation"] = json.loads(output.read_text())
      record["status"] = "complete"
      _persist(directory, manifest)
    manifest["status"] = "complete"
  except Exception as error:
    record["status"] = "failed"
    record["error"] = str(error)
    manifest["status"] = "failed"
    _persist(directory, manifest)
    raise
  _persist(directory, manifest)
  return directory


def _hash_stream(source) -> str:
  digest = hashlib.sha256()
  for block in iter(lambda: source.read(1024 * 1024), b""):
    digest.update(block)
  return digest.hexdigest()


if __name__ == "__main__":
  print(f"[INFO] Experiment record: {run(tyro.cli(RewardAblationConfig))}")
