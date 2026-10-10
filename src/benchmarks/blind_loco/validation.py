"""Read-only integrity audit of a complete local benchmark run."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path


FAILURES = ("tilt", "thigh_contact", "patch_exit", "numerical")


def _sha(path):
  value = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      value.update(block)
  return value.hexdigest()


def _require(condition, message):
  if not condition:
    raise ValueError(message)


def _finite(value, location):
  if isinstance(value, dict):
    for name, item in value.items():
      _finite(item, f"{location}/{name}")
  elif isinstance(value, list):
    for index, item in enumerate(value):
      _finite(item, f"{location}/{index}")
  elif isinstance(value, float):
    _require(math.isfinite(value), f"Nonfinite value: {location}")


def verify_run(directory):
  """Check actual artifacts and checkpoint bytes; never repair or rewrite them."""
  directory = Path(directory).resolve()
  read = lambda name: json.loads((directory/name).read_text())
  manifest = read("manifest.json")
  _require(manifest["status"] == "complete", "Only complete benchmark runs can be verified")
  _require(manifest["policy_unchanged"] is True, "Policy/normalization changed during evaluation")
  contract = manifest["protocol"]
  _require(read("protocol.json") == contract, "Protocol artifact differs from manifest")
  protocol_sha = hashlib.sha256(json.dumps(
    contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
  _require(protocol_sha == manifest["protocol_sha256"], "Protocol digest mismatch")
  cases = {(c["suite"], c["name"]): c for c in contract["cases"]}
  _require(len(cases) == len(contract["cases"]), "Duplicate case identifiers in protocol")
  expected = {(suite, case, repeat, trial) for suite, case in cases
              for repeat in range(contract["repeats"])
              for trial in range(contract["trials_per_case"])}
  rows = read("trials.json")
  keys = [(r["suite"], r["case"], r["repeat"], r["trial"]) for r in rows]
  _require(len(set(keys)) == len(keys), "Duplicate trial identifiers")
  _require(set(keys) == expected, "Incomplete or unexpected trial catalog")
  _require(manifest["trials"] == len(rows) and manifest["cases"] == len(cases),
           "Manifest trial/case counts differ from actual data")
  _finite(rows, "trials")
  for row in rows:
    _require(0 <= row["tracking_fraction"] <= 1, f"Tracking fraction outside [0,1]: {row['case']}")
    duration = cases[row["suite"], row["case"]]["duration"]
    _require(row["duration_s"] == duration, f"Trial duration differs from protocol: {row['case']}")
    _require(0 <= row["samples"] <= round(duration / contract["step_dt"]),
             f"Sample count exceeds the planned horizon: {row['case']}")
    _require(not row["success"] or row["survival"], f"Failed trial marked successful: {row['case']}")
    _require(row["survival"] == (not any(row[f"failure_{name}"] for name in FAILURES)),
             f"Survival and failure flags disagree: {row['case']}")
  summaries = read("metrics.json")
  _finite(summaries, "metrics")
  counts = Counter((r["suite"], r["case"]) for r in rows)
  summary_keys = [(r["suite"], r["case"]) for r in summaries]
  _require(len(set(summary_keys)) == len(summary_keys) and set(summary_keys) == set(cases),
           "Summary catalog differs from protocol")
  _require(all(r["trials"] == counts[r["suite"], r["case"]] for r in summaries),
           "Summary trial counts differ from raw trials")
  events = read("failure_events.json")
  failed = {(r["case"], r["repeat"], r["trial"]) for r in rows if not r["survival"]}
  event_keys = [(r["case"], r["repeat"], r["trial"]) for r in events]
  _require(len(event_keys) == len(set(event_keys)) and set(event_keys) == failed,
           "First-failure events do not match failed trials")
  hashes = read("artifact_manifest.json")
  required = {"manifest.json", "protocol.json", "trials.json", "metrics.json", "failure_events.json"}
  _require(required <= set(hashes), "Artifact digest manifest omits required result files")
  for name, sha in hashes.items():
    _require(_sha(directory/name) == sha, f"Artifact digest mismatch: {name}")
  source = manifest["software"]["source_sha256"] | manifest["software"]["robot_asset_sha256"]
  for name, sha in source.items():
    _require(_sha(directory/"source_snapshot"/name) == sha, f"Archived source digest mismatch: {name}")
  checkpoint = manifest["checkpoint"]
  _require(_sha(checkpoint["path"]) == checkpoint["sha256"], "Checkpoint digest mismatch")
  return {
    "status": "verified", "run": str(directory), "cases": len(cases), "trials": len(rows),
    "suite_trials": dict(Counter(r["suite"] for r in rows)), "failed_trials": len(failed),
    "failure_flags": {name: sum(r[f"failure_{name}"] for r in rows) for name in FAILURES},
    "artifact_hashes_checked": len(hashes), "archived_sources_checked": len(source),
    "policy_unchanged": True, "checkpoint_sha256": checkpoint["sha256"],
    "protocol_sha256": protocol_sha,
  }
