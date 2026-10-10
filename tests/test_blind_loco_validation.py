"""Read-only artifact validation; intentionally uses standard-library fixtures."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.benchmarks.blind_loco.validation import verify_run


class ValidationTests(unittest.TestCase):
  def setUp(self):
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.run = Path(self.temp.name)/"run"
    self.run.mkdir()
    checkpoint = Path(self.temp.name)/"model.pt"
    checkpoint.write_bytes(b"fixed test checkpoint")
    protocol = {"cases": [{"suite": "tracking", "name": "flat", "duration": .02}],
                "repeats": 1, "trials_per_case": 1, "step_dt": .02}
    manifest = {"status": "complete", "policy_unchanged": True, "protocol": protocol,
                "protocol_sha256": hashlib.sha256(json.dumps(
                  protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "trials": 1, "cases": 1,
                "checkpoint": {"path": str(checkpoint),
                               "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                "software": {"source_sha256": {}, "robot_asset_sha256": {}}}
    self.row = {"suite": "tracking", "case": "flat", "repeat": 0, "trial": 0,
                "duration_s": .02, "samples": 1, "tracking_fraction": 1.,
                "success": True, "survival": True,
                **{f"failure_{name}": False for name in ("tilt", "thigh_contact", "patch_exit", "numerical")}}
    for name, value in {"manifest.json": manifest, "protocol.json": protocol,
                        "trials.json": [self.row], "failure_events.json": [],
                        "metrics.json": [{"suite": "tracking", "case": "flat", "trials": 1}]}.items():
      self.write(name, value)
    (self.run/"report.md").write_text("original report")
    self.write("artifact_manifest.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                         for p in self.run.iterdir()})

  def write(self, name, value):
    (self.run/name).write_text(json.dumps(value))

  def test_accepts_complete_run_without_rewriting_artifacts(self):
    before = {p.name: p.read_bytes() for p in self.run.iterdir()}
    result = verify_run(self.run)
    self.assertEqual(result["status"], "verified")
    self.assertEqual(result["trials"], 1)
    self.assertEqual(result["failed_trials"], 0)
    self.assertEqual(before, {p.name: p.read_bytes() for p in self.run.iterdir()})

  def test_rejects_corrupted_artifact(self):
    (self.run/"report.md").write_text("changed report")
    with self.assertRaisesRegex(ValueError, "Artifact digest mismatch: report.md"):
      verify_run(self.run)

  def test_rejects_duplicate_trial_identifiers(self):
    self.write("trials.json", [self.row, self.row])
    with self.assertRaisesRegex(ValueError, "Duplicate trial"):
      verify_run(self.run)

  def test_rejects_out_of_range_tracking_fraction(self):
    self.write("trials.json", [{**self.row, "tracking_fraction": 1.01}])
    with self.assertRaisesRegex(ValueError, "Tracking fraction outside"):
      verify_run(self.run)

  def test_rejects_missing_trial(self):
    self.write("trials.json", [])
    with self.assertRaisesRegex(ValueError, "Incomplete or unexpected trial catalog"):
      verify_run(self.run)

  def test_rejects_nonfinite_metric(self):
    self.write("trials.json", [{**self.row, "xy_error": float("nan")}])
    with self.assertRaisesRegex(ValueError, "Nonfinite value"):
      verify_run(self.run)


if __name__ == "__main__":
  unittest.main()
