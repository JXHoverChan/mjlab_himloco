"""Strict inference-only checkpoint loading; no runner, storage, or optimizer."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re

import torch
from tensordict import TensorDict
import yaml

from src.tasks.velocity.rl.himactor import HIMActor
from src.tasks.velocity.rl.himencoder import HIMEncoder

from .protocol import OBS_DIM


class TupleSafeLoader(yaml.SafeLoader):
  """Training YAML adds tuples, not permission to execute Python tags."""


TupleSafeLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                               lambda loader, node: tuple(loader.construct_sequence(node)))


def sha_file(path: Path) -> str:
  value = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      value.update(block)
  return value.hexdigest()


def select_latest_checkpoint(root: Path) -> Path:
  """Latest dated run containing checkpoints, then highest numeric iteration."""
  root = Path(root)
  directories = sorted((p for p in root.iterdir() if p.is_dir()
                        and re.match(r"\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d", p.name)), reverse=True)
  for directory in directories:
    candidates = [(int(match.group(1)), p) for p in directory.glob("model_*.pt")
                  if (match := re.fullmatch(r"model_(\d+)\.pt", p.name))]
    if candidates:
      return max(candidates, key=lambda item: item[0])[1].resolve()
  raise FileNotFoundError(f"No dated training run with model_*.pt in {root}")


def state_sha(model) -> str:
  value = hashlib.sha256()
  for name, tensor in model.state_dict().items():
    value.update(name.encode())
    value.update(tensor.detach().cpu().contiguous().numpy().tobytes())
  return value.hexdigest()


def load_policy(checkpoint: Path, device: str):
  checkpoint = Path(checkpoint).resolve()
  agent_path = checkpoint.parent / "params" / "agent.yaml"
  if not agent_path.is_file():
    raise FileNotFoundError(f"Missing checkpoint architecture metadata: {agent_path}")
  with agent_path.open() as stream:
    metadata = yaml.load(stream, Loader=TupleSafeLoader)
  if metadata["actor"]["class_name"] != "src.tasks.velocity.rl.himactor:HIMActor":
    raise ValueError("This v1 adapter supports HIMActor only; add an explicit adapter for another architecture")
  package = torch.load(checkpoint, map_location="cpu", weights_only=False)
  state = package["actor_state_dict"]
  if not all(torch.isfinite(value).all() for value in state.values()):
    raise ValueError("Checkpoint contains NaN/Inf parameters or normalization statistics")
  input_width = state["source_encoder.mlp.0.weight"].shape[1]
  if input_width % OBS_DIM:
    raise ValueError(f"Encoder width {input_width} is incompatible with the {OBS_DIM}-D sensor contract")
  history_length = input_width // OBS_DIM
  encoder_config = dict(metadata["algorithm"]["source_encoder"])
  encoder_class = encoder_config.pop("class_name")
  if encoder_class != "src.tasks.velocity.rl.himencoder:HIMEncoder":
    raise ValueError(f"Unsupported encoder adapter: {encoder_class}")
  encoder = HIMEncoder(input_dim=OBS_DIM, history_length=history_length, **encoder_config)
  obs = TensorDict({"actor": torch.zeros(1, history_length, OBS_DIM)}, batch_size=[1])
  model_config = metadata["actor"]
  actor = HIMActor(
    obs=obs, obs_groups={"actor": ["actor"]}, obs_set="actor", output_dim=12,
    source_encoder=encoder, hidden_dims=model_config["hidden_dims"],
    activation=model_config["activation"], obs_normalization=model_config["obs_normalization"],
    distribution_cfg=model_config["distribution_cfg"],
  )
  actor.load_state_dict(state, strict=True)
  actor.to(device).eval().requires_grad_(False)
  return actor, history_length, {
    "path": str(checkpoint), "sha256": sha_file(checkpoint), "iteration": package.get("iter"),
    "agent_config": str(agent_path), "agent_config_sha256": sha_file(agent_path),
    "history_length": history_length, "obs_dim": OBS_DIM,
    "clip_actions": metadata.get("clip_actions"), "actor": model_config,
    "source_encoder": encoder_config,
  }
