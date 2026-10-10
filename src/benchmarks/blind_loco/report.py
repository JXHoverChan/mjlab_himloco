"""Deterministic tables and strict comparison of our own benchmark runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .protocol import stable_seed
from .validation import FAILURES, verify_run


METRICS = (
  "success", "survival", "tracking_fraction", "xy_error", "yaw_error",
  "steady_xy_error", "steady_yaw_error", "vx_bias", "vy_bias", "wz_bias",
  "stall_s", "max_stall_s", "end_time_s", "path_m", "yaw_angle_rad",
  "yaw_completion_ratio", "response_success_fraction", "mean_response_latency_s",
  "mean_stance_slip_m_s", "mean_mechanical_power_w", "rms_vertical_velocity_m_s",
  "mean_action_change_sq", "soft_joint_limit_fraction", "touchdowns", "foothold_shift_m",
  "FR_stance_fraction", "FL_stance_fraction", "RR_stance_fraction", "RL_stance_fraction",
  "FR_slip_m_s", "FL_slip_m_s", "RR_slip_m_s", "RL_slip_m_s",
) + tuple(f"failure_{name}" for name in FAILURES) + ("timeout",)


def write_json(path, value):
  Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")


def write_csv(path, rows):
  if not rows:
    raise ValueError("Cannot export an empty table")
  fields = list(dict.fromkeys(key for row in rows for key in row))
  with Path(path).open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                       for key, value in row.items()})


def interval(values, key):
  values = np.asarray(values, dtype=float)
  if not np.isfinite(values).all():
    raise ValueError(f"Nonfinite metric in {key}")
  rng = np.random.default_rng(stable_seed(key, "bootstrap"))
  means = values[rng.integers(0, len(values), size=(1500, len(values)))].mean(axis=1)
  lo, hi = np.quantile(means, [.025, .975])
  return float(values.mean()), float(lo), float(hi)


def is_timeout(row):
  return row["suite"] == "traversal" and not row["success"] and row["survival"]


def summarize(rows, *, paired_differences=False):
  groups = {}
  for row in rows:
    groups.setdefault((row["suite"], row["case"]), []).append(row)
  results = []
  for (suite, case), group in groups.items():
    result = {"suite": suite, "case": case, "terrain": group[0]["terrain"],
              "terrain_kind": group[0]["terrain_kind"], "mode": group[0]["mode"],
              "range_scale": group[0]["range_scale"], "trials": len(group)}
    for metric in METRICS:
      if metric == "timeout" and not paired_differences:
        values = [is_timeout(r) for r in group if "success" in r and "survival" in r]
      else:
        values = [r[metric] for r in group if r.get(metric) is not None]
      mean, lo, hi = interval(values, f"{suite}/{case}/{metric}") if values else (None, None, None)
      result[metric] = mean
      result[f"{metric}_ci_low"], result[f"{metric}_ci_high"] = lo, hi
      result[f"{metric}_available_trials"] = len(values)
    # Signed paired differences are rates, not event flags. Never reinterpret
    # e.g. a -1 success difference as a Boolean timeout or failure count.
    if not paired_differences:
      for failure in FAILURES:
        result[f"failure_{failure}_count"] = sum(r.get(f"failure_{failure}", False) for r in group)
      result["timeout_count"] = sum(is_timeout(r) for r in group if "success" in r and "survival" in r)
    results.append(result)
  return results


def fmt(value, digits=3):
  return "—" if value is None else f"{value:.{digits}f}"


def create_report(output, rows, manifest):
  output = Path(output)
  summaries = summarize(rows)
  write_csv(output/"summary.csv", summaries)
  write_json(output/"metrics.json", summaries)
  tracking_table = []
  for result in summaries:
    if result["suite"] == "tracking":
      tracking_table.append({key: result[key] for key in (
        "terrain", "terrain_kind", "mode", "range_scale", "trials", "xy_error", "yaw_error",
        "survival", "tracking_fraction", "steady_xy_error", "steady_yaw_error")})
  if tracking_table:
    write_csv(output/"tracking_table.csv", tracking_table)
  suite_means = {}
  for suite in dict.fromkeys(r["suite"] for r in summaries):
    group = [r for r in summaries if r["suite"] == suite]
    suite_means[suite] = {
      "cases": len(group), "trials": sum(r["trials"] for r in group),
      **{metric: float(np.mean([r[metric] for r in group if r[metric] is not None]))
         if any(r[metric] is not None for r in group) else None
         for metric in ("success", "survival", "tracking_fraction", "xy_error", "yaw_error", "stall_s")},
    }
  write_json(output/"overview.json", suite_means)
  checkpoint = manifest["checkpoint"]
  lines = ["# Go2 盲运动基准评测", "", f"协议：`{manifest['protocol']['version']}`；规模：`{manifest['protocol']['profile']}`。",
           f"权重：`{checkpoint['path']}`，iteration `{checkpoint['iteration']}`。",
           "", "这是自身策略的首轮能力档案，不包含其他论文算法对比；没有单独的训练种子重复，不能据此评价算法跨训练种子的稳定性。",
           "", "## 完整性与可比性", "",
           f"- 场景数：{len(summaries)}；回合数：{len(rows)}。",
           f"- 协议哈希：`{manifest['protocol_sha256']}`。",
           f"- 执行环境哈希：`{manifest['execution_contract_sha256']}`。",
           f"- 参数/归一化统计未改变：`{manifest.get('policy_unchanged')}`。",
           "- 关闭训练课程、奖励、随机推力、观测噪声与启动域随机化；名义物理模型。",
           "- 场景有真实坡面/台阶/粗糙坡/离散障碍，不在中心额外擦除为平地。",
           "- 接触、倾倒、越界、数值异常记录首次失败；失败重置后的样本不计入原回合。",
           "- bootstrap 区间只反映固定地形库下的试验变动，不是对未知地形分布或训练种子的置信区间。",
           "", "## 套件概览", "",
           "| 套件 | 场景 | 回合 | 任务成功率 | 存活率 | 有效跟踪比例 | XY误差 m/s | yaw误差 rad/s | 停滞 s |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for suite, result in suite_means.items():
    lines.append(f"| {suite} | {result['cases']} | {result['trials']} | {fmt(result['success'])} | "
                 f"{fmt(result['survival'])} | {fmt(result['tracking_fraction'])} | "
                 f"{fmt(result['xy_error'])} | {fmt(result['yaw_error'])} | {fmt(result['stall_s'])} |")
  lines += ["", "概览为各场景等权均值；跟踪/操作成功表示存活，穿越成功表示到达目标。误差统计截至首次失败的有限样本（包括有限的失败当步），必须与失败率及完整时段跟踪比例一起看。不同套件任务不同，不合成总分。",
            "", "## 分地形、分工况误差", "",
            "| 固定地形 | 工况 | vx/yaw范围倍率 | XY误差 m/s | yaw误差 rad/s | 存活率 | 有效跟踪比例 |",
            "|---|---|---:|---:|---:|---:|---:|"]
  for row in tracking_table:
    lines.append(f"| {row['terrain']} | {row['mode']} | {row['range_scale']:g} | {fmt(row['xy_error'])} | "
                 f"{fmt(row['yaw_error'])} | {fmt(row['survival'])} | {fmt(row['tracking_fraction'])} |")
  lines += ["", "倍率2时仅 vx 和 yaw 的范围扩大，vy仍为±1 m/s。纯平移也记录非期望 yaw，纯转向也记录非期望平移；这是为自身诊断保留的额外指标。",
            "", "## 低速操作与持续转向", "",
            "| 工况 | 存活率 | 有效跟踪比例 | XY误差 | yaw误差 | 停滞s | 累计yaw/目标 | vy偏差 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
  for row in summaries:
    if row["suite"] == "maneuver":
      lines.append(f"| {row['case']} | {fmt(row['survival'])} | {fmt(row['tracking_fraction'])} | "
                   f"{fmt(row['xy_error'])} | {fmt(row['yaw_error'])} | {fmt(row['stall_s'])} | "
                   f"{fmt(row['yaw_completion_ratio'])} | {fmt(row['vy_bias'])} |")
  lines += ["", "## 启停与换向", "",
            "每个工况按站立、正向、停止、反向、停止执行，每段3秒。响应成功要求XY和yaw误差同时达标并连续保持0.4秒；响应时间只统计成功响应，须与响应成功比例一起看。", "",
            "| 工况 | 存活率 | 有效跟踪比例 | 响应成功比例 | 响应时间 s | XY误差 m/s | yaw误差 rad/s |",
            "|---|---:|---:|---:|---:|---:|---:|"]
  for row in summaries:
    if row["suite"] == "transitions":
      lines.append(f"| {row['case']} | {fmt(row['survival'])} | {fmt(row['tracking_fraction'])} | "
                   f"{fmt(row['response_success_fraction'])} | {fmt(row['mean_response_latency_s'])} | "
                   f"{fmt(row['xy_error'])} | {fmt(row['yaw_error'])} |")
  lines += ["", "## 固定路线通过", "",
            "| 地形/速度 | 到达率 | 存活率 | XY误差 | 停滞s | 平均结束时刻s |",
            "|---|---:|---:|---:|---:|---:|"]
  for row in summaries:
    if row["suite"] == "traversal":
      lines.append(f"| {row['case']} | {fmt(row['success'])} | {fmt(row['survival'])} | {fmt(row['xy_error'])} | "
                   f"{fmt(row['stall_s'])} | {fmt(row['end_time_s'])} |")
  failure_counts = {name: sum(bool(r.get(f"failure_{name}")) for r in rows)
                    for name in ("tilt", "thigh_contact", "patch_exit", "numerical")}
  lines += ["", "## 失败和解释边界", "", f"首次失败事件：`{json.dumps(failure_counts, ensure_ascii=False)}`；多个事件可能在同一步同时发生。",
            "", "- 本轮只建立当前策略基线，不能声称比旧版本改善；后续使用同协议与执行指纹进行配对比较。",
            "- 名义无噪声评测不是随机扰动鲁棒性或实机验证。机械功率为关节广义力乘速度的绝对值之和，不是电池耗电。",
            "- 接触切换/落脚位移仅作诊断，包含接触抖动，不单独当成步态优劣评分。",
            "- 路线到达后立即结束；其有效跟踪比例仍以计划10秒为分母，会因提前到达降低。路线能力以到达率和成功到达时间为主，不用该比例排名。",
            "- 累计yaw/目标的目标积分截至有效测量终止；失败回合的比值不代表完成全部30秒，须同时看存活率。",
            "- 误差区间、逐足支撑/滑移、响应时间和数值异常详见逐回合与汇总表。",
            "", "## 产物", "",
            "`manifest.json`、`protocol.json`、`trials.csv/json`、`summary.csv`、`tracking_table.csv`、`traces.csv`、`env_*.yaml`、`terrain_geometry.json`、`benchmark.log`、`source_snapshot/`、`agent_config.yaml`。",
            "", "下一步：针对低速启停、持续转向和地形通过的薄弱场景提出独立改进；改进后复用本轮协议，不改变任务难度来制造提升。", ""]
  (output/"report.md").write_text("\n".join(lines))
  return summaries, suite_means


def compare_runs(reference: Path, candidate: Path, destination: Path):
  reference, candidate, destination = map(Path, (reference, candidate, destination))
  first, second = [json.loads((path/"manifest.json").read_text()) for path in (reference, candidate)]
  if first["status"] != "complete" or second["status"] != "complete":
    raise ValueError("Only complete benchmark runs can be compared")
  for field in ("protocol_sha256", "execution_contract_sha256"):
    if first[field] != second[field]:
      raise ValueError(f"Non-comparable {field}: rerun both checkpoints under the same protocol/software/physics")
  if first.get("suites", {}).keys() != second.get("suites", {}).keys():
    raise ValueError("Suite catalogs differ")
  for suite in first.get("suites", {}):
    if first["suites"][suite]["compiled_physics_sha256"] != second["suites"][suite]["compiled_physics_sha256"]:
      raise ValueError(f"Compiled physics differs in suite {suite}")
  rows_a, rows_b = [json.loads((path/"trials.json").read_text()) for path in (reference, candidate)]
  key = lambda r: (r["suite"], r["case"], r["repeat"], r["trial"])
  a, b = {key(r): r for r in rows_a}, {key(r): r for r in rows_b}
  if len(a) != len(rows_a) or len(b) != len(rows_b):
    raise ValueError("Duplicate trial identifiers: cannot construct a one-to-one paired comparison")
  if a.keys() != b.keys():
    raise ValueError("Trial catalogs differ")
  deltas = []
  for trial_key in a:
    for field in ("initial_sha256", "command_sha256", "geometry_sha256"):
      if a[trial_key][field] != b[trial_key][field]:
        raise ValueError(f"Unpaired {field}: {trial_key}")
    row = {field: b[trial_key][field] for field in ("suite", "case", "terrain", "terrain_kind", "mode", "range_scale")}
    for metric in METRICS:
      if metric == "timeout":
        left, right = is_timeout(a[trial_key]), is_timeout(b[trial_key])
      else:
        left, right = a[trial_key].get(metric), b[trial_key].get(metric)
      row[metric] = right-left if left is not None and right is not None else None
    deltas.append(row)
  # Matching metadata alone is insufficient: reject corrupted/partial raw
  # results and changed checkpoint bytes before generating any comparison.
  validation = [verify_run(path) for path in (reference, candidate)]
  result = summarize(deltas, paired_differences=True)
  destination.mkdir(parents=True, exist_ok=False)
  write_csv(destination/"comparison.csv", result)
  write_json(destination/"comparison.json", result)
  write_json(destination/"manifest.json", {
    "status": "complete", "kind": "paired_comparison", "reference": str(reference.resolve()),
    "candidate": str(candidate.resolve()), "paired_trials": len(deltas), "cases": len(result),
    "protocol_sha256": first["protocol_sha256"],
    "execution_contract_sha256": first["execution_contract_sha256"],
    "input_verification": validation,
    "comparator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "difference": "candidate minus reference; conditional paired trial bootstrap",
  })
  lines = ["# 自身策略配对比较", "", f"参考：`{reference}`", "", f"候选：`{candidate}`", "",
           f"已核验并配对 {len(deltas)} 回合、{len(result)} 个工况。",
           "差值为候选−参考；成功率/存活率/跟踪比例为正较好，误差/停滞/失败率为负较好。",
           "协议、软件/物理指纹、场景、指令、初始状态全部匹配；两个输入的产物和权重哈希核验均通过。",
           "95%区间与有效配对数量见 comparison.csv。到达时间和响应时间的条件性仍须结合成功率解释。",
           "", "| 工况 | Δ成功率 | Δ存活率 | Δ有效跟踪 | ΔXY误差 m/s | Δyaw误差 rad/s | Δ停滞 s | Δ大腿接地率 | Δ超时率 |",
           "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
  for row in result:
    columns = ("success", "survival", "tracking_fraction", "xy_error", "yaw_error", "stall_s",
               "failure_thigh_contact", "timeout")
    lines.append(f"| {row['suite']}/{row['case']} | " + " | ".join(fmt(row[key]) for key in columns) + " |")
  (destination/"report.md").write_text("\n".join(lines)+"\n")
  return result
