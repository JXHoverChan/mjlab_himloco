"""Public contract checks; standard library runner needs no pytest installation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
import torch

from src.benchmarks.blind_loco.environment import ReplayCommand, ReplayCommandCfg, make_environment_cfg, phase
from src.benchmarks.blind_loco.metrics import TrialStatistics, tracking_errors
from src.benchmarks.blind_loco.policy import select_latest_checkpoint
from src.benchmarks.blind_loco.protocol import (
  Case, STEP_DT, TerrainSpec, case_catalog, digest, reference_schedule, terrain_library,
)
from src.benchmarks.blind_loco.report import compare_runs, summarize, write_json
from src.benchmarks.blind_loco.terrains import BenchmarkTerrainCfg, height_grid, surface_sample


class ProtocolTests(unittest.TestCase):
  def test_tracking_references_preserve_zero_axes_and_only_expand_vx_yaw(self):
    case = Case("slope_linear_2", "tracking", "slope_10deg", "linear", 8., range_scale=2.)
    schedule = reference_schedule(case, 512, 0)
    self.assertTrue(np.all(schedule[:, :50] == 0))
    self.assertTrue(np.all(schedule[:, :, 2] == 0))
    self.assertLessEqual(np.max(np.abs(schedule[:, :, 1])), 1.)
    self.assertGreater(np.max(np.abs(schedule[:, :, 0])), 1.5)
    self.assertTrue(np.array_equal(schedule, reference_schedule(case, 512, 0)))

  def test_catalog_contains_no_training_level_and_low_yaw_both_directions(self):
    cases = case_catalog()
    commands = {c.command for c in cases if c.suite == "maneuver"}
    self.assertTrue({(0., 0., -.05), (0., 0., .05)} <= commands)
    self.assertEqual(len({c.name for c in cases if c.suite == "tracking"}), 81)
    self.assertTrue(all(c.terrain in {t.name for t in terrain_library()} for c in cases))

  def test_latest_means_latest_run_then_numeric_iteration(self):
    with tempfile.TemporaryDirectory() as temp:
      root = Path(temp)
      old, new = root/"2026-10-08_12-00-00_old", root/"2026-10-09_12-00-00_new"
      old.mkdir(); new.mkdir()
      (old/"model_99999.pt").touch()
      (new/"model_900.pt").touch()
      (new/"model_1000.pt").touch()
      self.assertEqual(select_latest_checkpoint(root), (new/"model_1000.pt").resolve())


class ReplayTests(unittest.TestCase):
  def test_phase_matches_the_training_sensor_formula_and_explicit_standing(self):
    from types import SimpleNamespace
    command = torch.tensor([[0., 0., 0.], [0., 0., .05], [.2, 0., 0.]])
    env = SimpleNamespace(
      num_envs=3, episode_length_buf=torch.tensor([1500, 1501, 1502]), step_dt=STEP_DT,
      command_manager=SimpleNamespace(get_command=lambda name: command),
    )
    angle = (env.episode_length_buf * STEP_DT) % .6 / .6 * (2 * torch.pi)
    expected = torch.stack((angle.sin(), angle.cos()), dim=-1)
    expected[0] = 0.
    self.assertTrue(torch.equal(phase(env), expected))

  def test_command_is_updated_at_observation_time_and_partial_reset_is_scoped(self):
    from types import SimpleNamespace
    env = SimpleNamespace(num_envs=2, device="cpu", episode_length_buf=torch.tensor([49, 50]))
    term = ReplayCommand(ReplayCommandCfg(), env)
    case = Case("startup", "maneuver", "flat", "linear", 8., command=(.5, 0., 0.))
    term.schedule = torch.tensor(reference_schedule(case, 2, 0))
    term.compute(0.)
    self.assertTrue(torch.equal(term.command[:, 0], torch.tensor([0., .5])))
    env.episode_length_buf[0] = 0
    term.reset(torch.tensor([0]))
    term.compute(0., env_ids=torch.tensor([0]))
    self.assertEqual(float(term.command[1, 0]), .5)


class MetricTests(unittest.TestCase):
  def test_perfect_thirty_second_tracking_is_exactly_one(self):
    stats = TrialStatistics(1, "cpu")
    for step in range(1500):
      stats.update(velocity=torch.zeros(1, 2), yaw=torch.zeros(1),
                   command=torch.zeros(1, 3), usable=torch.tensor([True]),
                   failed=torch.tensor([False]), steady=torch.tensor([True]),
                   stage=0, stage_elapsed=(step+1)*STEP_DT)
    row = stats.export([30.], [1])[0]
    self.assertEqual(row["tracking_time_s"], 30.)
    self.assertEqual(row["tracking_fraction"], 1.)

  def test_errors_are_vector_distance_and_absolute_angular_error_not_rmse(self):
    linear, yaw = tracking_errors(torch.tensor([[3., 4.]]), torch.tensor([-.2]), torch.tensor([[0., 0., .1]]))
    self.assertAlmostEqual(float(linear), 5.)
    self.assertAlmostEqual(float(yaw), .3)

  def test_low_yaw_stall_and_post_failure_time_are_not_hidden(self):
    stats = TrialStatistics(1, "cpu")
    command = torch.tensor([[0., 0., .05]])
    for step in range(100):
      stats.update(velocity=torch.zeros(1, 2), yaw=torch.zeros(1), command=command,
                   usable=torch.tensor([step < 50]), failed=torch.tensor([step >= 50]),
                   steady=torch.tensor([False]), stage=0, stage_elapsed=(step+1)*STEP_DT)
    row = stats.export([2.], [1])[0]
    self.assertAlmostEqual(row["stall_s"], 1., places=5)
    self.assertEqual(row["tracking_fraction"], 0.)
    self.assertAlmostEqual(row["yaw_error"], .05, places=5)

  def test_response_requires_sustained_accuracy(self):
    stats = TrialStatistics(1, "cpu")
    for step in range(25):
      accurate = step < 1 or step >= 5
      stats.update(velocity=torch.tensor([[.5 if accurate else 0., 0.]]), yaw=torch.zeros(1),
                   command=torch.tensor([[.5, 0., 0.]]), usable=torch.tensor([True]),
                   failed=torch.tensor([False]), steady=torch.tensor([False]),
                   stage=0, stage_elapsed=(step+1)*STEP_DT)
    row = stats.export([.5], [1])[0]
    self.assertEqual(row["responses"], [True])
    self.assertAlmostEqual(row["response_latencies_s"][0], .12, places=5)


class TerrainTests(unittest.TestCase):
  def test_compiled_surfaces_match_height_grid_in_world_coordinates(self):
    # Check real compiled geometry, not just its configuration. Sampling grid
    # vertices avoids conflating triangular heightfield interpolation with a
    # bilinear planning query between vertices.
    for terrain in terrain_library():
      with self.subTest(terrain=terrain.name):
        spec = mujoco.MjSpec()
        spec.worldbody.add_body(name="terrain")
        BenchmarkTerrainCfg(terrain=terrain, size=(40., 40.)).function(
          0., spec, np.random.default_rng(42))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        grid = height_grid(terrain)
        for x, y in ((-1.1, -.7), (.8, .6), (0., 0.)):
          point = np.array([20.+x, 20.+y, 30.])
          geom_id = np.array([-1], dtype=np.int32)
          distance = mujoco.mj_ray(model, data, point, np.array([0., 0., -1.]),
                                   None, True, -1, geom_id)
          self.assertGreaterEqual(distance, 0.)
          self.assertAlmostEqual(point[2]-distance, surface_sample(grid, x, y), places=5)

  def test_obstacle_routes_encounter_nonflat_geometry(self):
    for terrain in terrain_library():
      if terrain.kind == "obstacles":
        grid = height_grid(terrain)
        elevations = [surface_sample(grid, x, 0.) for x in np.linspace(-3., 3., 61)]
        self.assertGreater(np.ptp(elevations), .01, terrain.name)

  def test_plane_gradient_is_exact_and_random_roughness_is_repeatable(self):
    terrain = TerrainSpec("slope", "slope", slope_degrees=20)
    grid = height_grid(terrain, 4.)
    self.assertAlmostEqual(float(grid[20, -1]-grid[20, 0]), 4*math.tan(math.radians(20)))
    rough = TerrainSpec("rough", "rough_slope", slope_degrees=10, noise_amplitude=.02, seed=42)
    self.assertTrue(np.array_equal(height_grid(rough), height_grid(rough)))

  def test_generated_stairs_have_real_edges_and_compile(self):
    terrain = TerrainSpec("stairs", "stairs", height=.08)
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    cfg = BenchmarkTerrainCfg(terrain=terrain, size=(4., 4.))
    cfg.function(0., spec, np.random.default_rng(42))
    model = spec.compile()
    self.assertGreater(model.ngeom, 10)
    tops = model.geom_pos[:, 2]+model.geom_size[:, 2]
    self.assertAlmostEqual(float(np.median(np.diff(tops))), .08)

  def test_frozen_environment_has_no_training_rewards_curriculum_or_randomization(self):
    cfg = make_environment_cfg(terrain_library()[:1], 2, 5, 42)
    self.assertEqual(cfg.rewards, {})
    self.assertEqual(cfg.curriculum, {})
    self.assertEqual({event.mode for event in cfg.events.values()}, {"reset"})
    self.assertEqual(list(cfg.observations), ["actor"])


class ComparisonTests(unittest.TestCase):
  def complete_run(self, directory, outcomes, errors):
    directory.mkdir()
    checkpoint = directory.parent/"fixture.pt"
    if not checkpoint.exists():
      checkpoint.write_bytes(b"immutable checkpoint fixture")
    protocol = {"cases": [{"suite": "traversal", "name": "flat", "duration": .02}],
                "repeats": 1, "trials_per_case": len(outcomes), "step_dt": .02}
    rows = []
    for trial, ((success, survival), error) in enumerate(zip(outcomes, errors)):
      rows.append({"suite": "traversal", "case": "flat", "terrain": "flat",
                   "terrain_kind": "flat", "mode": "linear", "range_scale": 1.,
                   "repeat": 0, "trial": trial, "duration_s": .02, "samples": 1,
                   "tracking_fraction": 1. if success else 0., "xy_error": error,
                   "success": success, "survival": survival,
                   "initial_sha256": f"initial-{trial}", "command_sha256": f"command-{trial}",
                   "geometry_sha256": "same-geometry", "failure_tilt": not survival,
                   **{f"failure_{name}": False for name in ("thigh_contact", "patch_exit", "numerical")}})
    manifest = {"status": "complete", "policy_unchanged": True, "protocol": protocol,
                "protocol_sha256": digest(protocol), "execution_contract_sha256": "same",
                "suites": {"traversal": {"compiled_physics_sha256": "same"}},
                "trials": len(rows), "cases": 1,
                "checkpoint": {"path": str(checkpoint),
                               "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                "software": {"source_sha256": {}, "robot_asset_sha256": {}}}
    contents = {"manifest.json": manifest, "protocol.json": protocol, "trials.json": rows,
                "metrics.json": [{"suite": "traversal", "case": "flat", "trials": len(rows)}],
                "failure_events.json": [{"case": row["case"], "repeat": 0, "trial": row["trial"]}
                                        for row in rows if not row["survival"]]}
    for name, value in contents.items():
      write_json(directory/name, value)
    write_json(directory/"artifact_manifest.json", {
      path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in directory.iterdir()})

  def test_verified_self_comparison_has_only_zero_available_differences(self):
    with tempfile.TemporaryDirectory() as temp:
      path = Path(temp)/"run"
      self.complete_run(path, [(True, True), (False, True)], [.1, .2])
      result = compare_runs(path, path, Path(temp)/"out")[0]
      for name in ("success", "survival", "xy_error", "tracking_fraction", "failure_tilt", "timeout"):
        self.assertEqual(result[name], 0.)
        self.assertEqual(result[f"{name}_ci_low"], 0.)
        self.assertEqual(result[f"{name}_ci_high"], 0.)
      self.assertNotIn("timeout_count", result)
      manifest = json.loads((Path(temp)/"out"/"manifest.json").read_text())
      self.assertEqual(manifest["paired_trials"], 2)
      self.assertTrue(all(row["status"] == "verified" for row in manifest["input_verification"]))

  def test_failure_and_timeout_differences_are_signed_rates_not_counts(self):
    with tempfile.TemporaryDirectory() as temp:
      reference, candidate = Path(temp)/"a", Path(temp)/"b"
      self.complete_run(reference, [(False, False), (False, True)], [.4, .2])
      self.complete_run(candidate, [(True, True), (True, True)], [.1, .1])
      result = compare_runs(reference, candidate, Path(temp)/"out")[0]
      self.assertEqual(result["success"], 1.)
      self.assertEqual(result["survival"], .5)
      self.assertEqual(result["failure_tilt"], -.5)
      self.assertEqual(result["timeout"], -.5)
      self.assertAlmostEqual(result["xy_error"], -.2)
      self.assertNotIn("failure_tilt_count", result)

  def test_compare_rejects_corrupted_input_before_creating_output(self):
    with tempfile.TemporaryDirectory() as temp:
      path = Path(temp)/"run"
      self.complete_run(path, [(True, True)], [.1])
      write_json(path/"metrics.json", [{"suite": "traversal", "case": "flat", "trials": 1, "xy_error": .5}])
      output = Path(temp)/"out"
      with self.assertRaisesRegex(ValueError, "Artifact digest mismatch: metrics.json"):
        compare_runs(path, path, output)
      self.assertFalse(output.exists())

  def test_compare_rejects_duplicate_trial_identifiers(self):
    with tempfile.TemporaryDirectory() as temp:
      a, b = Path(temp)/"a", Path(temp)/"b"
      a.mkdir(); b.mkdir()
      row = {"suite": "tracking", "case": "flat", "repeat": 0, "trial": 0}
      for path in (a, b):
        write_json(path/"manifest.json", {"status": "complete", "protocol_sha256": "same",
                                          "execution_contract_sha256": "same"})
        write_json(path/"trials.json", [row, row])
      with self.assertRaisesRegex(ValueError, "Duplicate trial"):
        compare_runs(a, b, Path(temp)/"out")

  def test_compare_rejects_mismatched_protocol(self):
    with tempfile.TemporaryDirectory() as temp:
      a, b = Path(temp)/"a", Path(temp)/"b"
      a.mkdir(); b.mkdir()
      for path, value in ((a, "first"), (b, "second")):
        write_json(path/"manifest.json", {"status": "complete", "protocol_sha256": value,
                                          "execution_contract_sha256": "same"})
      with self.assertRaisesRegex(ValueError, "Non-comparable"):
        compare_runs(a, b, Path(temp)/"out")

  def test_conditional_summary_keeps_missing_error_counts(self):
    rows = [{"suite": "maneuver", "case": "one", "terrain": "flat", "terrain_kind": "flat",
             "mode": "linear", "range_scale": 1., "success": False, "survival": False,
             "xy_error": None, "tracking_fraction": 0.},
            {"suite": "maneuver", "case": "one", "terrain": "flat", "terrain_kind": "flat",
             "mode": "linear", "range_scale": 1., "success": True, "survival": True,
             "xy_error": .2, "tracking_fraction": .8}]
    summary = summarize(rows)[0]
    self.assertEqual(summary["trials"], 2)
    self.assertEqual(summary["xy_error_available_trials"], 1)
    self.assertEqual(summary["success"], .5)


if __name__ == "__main__":
  unittest.main()
