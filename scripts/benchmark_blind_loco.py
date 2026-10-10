"""Automated run/verify/compare for our versioned blind-locomotion benchmark."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


def main():
  parser = argparse.ArgumentParser(description="固定协议的 Go2 盲运动评测：run / verify / compare")
  commands = parser.add_subparsers(dest="command", required=True)
  run = commands.add_parser("run", help="推理评测；不训练。不指定权重时自动选最新 dated run 的最高 iteration")
  run.add_argument("--checkpoint", type=Path)
  run.add_argument("--checkpoint-root", type=Path, default=Path("logs/rsl_rl/go2_him"))
  run.add_argument("--profile", choices=("standard", "quick", "smoke"), default="standard")
  run.add_argument("--device", default="cuda:0")
  run.add_argument("--suites", nargs="+", choices=("tracking", "maneuver", "transitions", "traversal"),
                   default=["tracking", "maneuver", "transitions", "traversal"])
  run.add_argument("--trials-per-case", type=int)
  run.add_argument("--repeats", type=int)
  run.add_argument("--seed", type=int, default=20261009)
  run.add_argument("--output-dir", type=Path)
  run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
  compare = commands.add_parser("compare", help="严格校验配对和协议后，生成候选减参考的差值表")
  compare.add_argument("reference", type=Path)
  compare.add_argument("candidate", type=Path)
  compare.add_argument("--output-dir", type=Path)
  verify = commands.add_parser("verify", help="只读核验完整产物、回合目录、源快照和权重哈希")
  verify.add_argument("run_dir", type=Path)
  verify.add_argument("--output-file", type=Path, help="可选的核验记录路径；不得覆盖已有文件")
  args = parser.parse_args()
  if args.command == "verify":
    from src.benchmarks.blind_loco.validation import verify_run
    result = json.dumps(verify_run(args.run_dir), ensure_ascii=False, indent=2)+"\n"
    if args.output_file is not None:
      with args.output_file.open("x") as stream:
        stream.write(result)
    print(result, end="")
    return
  from src.benchmarks.blind_loco.policy import select_latest_checkpoint
  stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
  if args.command == "compare":
    from src.benchmarks.blind_loco.report import compare_runs
    output = args.output_dir or Path("artifacts/experiment/blind_loco_benchmark")/f"compare_{stamp}"
    result = compare_runs(args.reference, args.candidate, output)
    print(f"[BENCH] compared {len(result)} cases: {output.resolve()}")
    return
  checkpoint = (args.checkpoint or select_latest_checkpoint(args.checkpoint_root)).resolve()
  output = (args.output_dir or Path("artifacts/experiment/blind_loco_benchmark")/f"{stamp}_{args.profile}").resolve()
  if not args.worker:
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", "scripts.benchmark_blind_loco", "run", "--worker",
               "--checkpoint", str(checkpoint), "--output-dir", str(output), "--profile", args.profile,
               "--device", args.device, "--seed", str(args.seed), "--suites", *args.suites]
    if args.trials_per_case is not None:
      command += ["--trials-per-case", str(args.trials_per_case)]
    if args.repeats is not None:
      command += ["--repeats", str(args.repeats)]
    (output/"command.json").write_text(json.dumps({"argv": command, "cwd": str(Path.cwd()),
                                                 "user_argv": sys.argv}, indent=2)+"\n")
    print(f"[BENCH] selected checkpoint={checkpoint}\n[BENCH] durable output={output}", flush=True)
    with (output/"benchmark.log").open("w") as log:
      process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
      assert process.stdout is not None
      try:
        for line in process.stdout:
          print(line, end="", flush=True)
          log.write(line)
          log.flush()
        return_code = process.wait()
      except BaseException:
        process.terminate()
        process.wait()
        raise
    if return_code:
      raise SystemExit(return_code)
  else:
    from src.benchmarks.blind_loco.runner import run_benchmark
    run_benchmark(checkpoint, output, profile=args.profile, device=args.device, suites=tuple(args.suites),
                  trials=args.trials_per_case, repeats=args.repeats, seed=args.seed)


if __name__ == "__main__":
  main()
