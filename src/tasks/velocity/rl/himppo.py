"""PPO with a separate HIM source/target encoder objective."""

from __future__ import annotations

import copy
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer

from .himactor import HIMActor
from .himencoder import HIMEncoder
from .himstorage import HIMRolloutStorage


class HIM_PPO(PPO):
  """PPO plus source/target contrastive and velocity-prediction updates.

  The source encoder is used by the actor under ``torch.no_grad()``. Encoder
  parameters are therefore excluded from the PPO optimizer and are updated
  only by ``encoder_optimizer``.
  """

  actor: HIMActor
  critic: MLPModel
  source_encoder: HIMEncoder
  target_encoder: HIMEncoder

  def __init__(
    self,
    source_encoder: HIMEncoder,
    target_encoder: HIMEncoder,
    actor: HIMActor,
    critic: MLPModel,
    storage: HIMRolloutStorage,
    num_learning_epochs: int = 5,
    num_mini_batches: int = 4,
    clip_param: float = 0.2,
    gamma: float = 0.99,
    lam: float = 0.95,
    value_loss_coef: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 0.001,
    encoder_learning_rate: float | None = None,
    max_grad_norm: float = 1.0,
    encoder_max_grad_norm: float = 10.0,
    optimizer: str = "adam",
    use_clipped_value_loss: bool = True,
    schedule: str = "adaptive",
    desired_kl: float = 0.01,
    normalize_advantage_per_mini_batch: bool = False,
    device: str = "cpu",
    source_obs_group: str = "actor",
    target_obs_group: str = "target_encoder",
    velocity_obs_group: str = "velocity",
    num_prototypes: int = 32,
    temperature: float = 3.0,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iterations: int = 3,
    estimation_loss_coef: float = 1.0,
    swap_loss_coef: float = 1.0,
    rnd_cfg: dict | None = None,
    symmetry_cfg: dict | None = None,
    multi_gpu_cfg: dict | None = None,
  ) -> None:
    self.device = device
    self.is_multi_gpu = multi_gpu_cfg is not None
    if multi_gpu_cfg is not None:
      self.gpu_global_rank = multi_gpu_cfg["global_rank"]
      self.gpu_world_size = multi_gpu_cfg["world_size"]
    else:
      self.gpu_global_rank = 0
      self.gpu_world_size = 1

    self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None
    if symmetry_cfg is not None:
      raise NotImplementedError("HIM_PPO currently requires symmetry_cfg=None because auxiliary fields are not mirrored.")
    self.symmetry: Symmetry | None = None

    self.actor = actor.to(self.device)
    self.critic = critic.to(self.device)
    self.source_encoder = source_encoder.to(self.device)
    self.target_encoder = target_encoder.to(self.device)
    if self.actor.source_encoder is not self.source_encoder:
      raise ValueError("HIMActor and HIM_PPO must share the same source_encoder instance")

    self._raw_actor = self.actor
    self._raw_critic = self.critic
    self._raw_source_encoder = self.source_encoder
    self._raw_target_encoder = self.target_encoder

    self.source_obs_group = source_obs_group
    self.target_obs_group = target_obs_group
    self.velocity_obs_group = velocity_obs_group
    self.temperature = temperature
    self.sinkhorn_epsilon = sinkhorn_epsilon
    self.sinkhorn_iterations = sinkhorn_iterations
    self.estimation_loss_coef = estimation_loss_coef
    self.swap_loss_coef = swap_loss_coef
    self.encoder_max_grad_norm = encoder_max_grad_norm

    if num_prototypes <= 0:
      raise ValueError(f"num_prototypes must be positive, got {num_prototypes}")
    if temperature <= 0 or sinkhorn_epsilon <= 0:
      raise ValueError("temperature and sinkhorn_epsilon must be positive")
    if source_encoder.latent_dim <= 0:
      raise ValueError("source encoder latent_dim must be positive")
    if target_encoder.latent_dim != source_encoder.latent_dim:
      raise ValueError("Source and target encoders must have the same latent_dim")

    self.prototypes = nn.Parameter(
      F.normalize(torch.randn(num_prototypes, source_encoder.latent_dim, device=self.device), dim=-1)
    )

    source_parameter_ids = {id(parameter) for parameter in self.source_encoder.parameters()}
    self._ppo_parameters = [
      parameter
      for parameter in list(self.actor.parameters()) + list(self.critic.parameters())
      if id(parameter) not in source_parameter_ids
    ]
    self._encoder_parameters = list(self.source_encoder.parameters()) + list(self.target_encoder.parameters())
    self._encoder_parameters.append(self.prototypes)

    self.ppo_optimizer = resolve_optimizer(optimizer)(self._ppo_parameters, lr=learning_rate)  # type: ignore
    self.optimizer = self.ppo_optimizer
    encoder_lr = learning_rate if encoder_learning_rate is None else encoder_learning_rate
    self.encoder_optimizer = resolve_optimizer(optimizer)(self._encoder_parameters, lr=encoder_lr)  # type: ignore

    self.storage = storage
    self.transition = HIMRolloutStorage.Transition()
    self.intrinsic_rewards: torch.Tensor | None = None

    self.clip_param = clip_param
    self.num_learning_epochs = num_learning_epochs
    self.num_mini_batches = num_mini_batches
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.gamma = gamma
    self.lam = lam
    self.max_grad_norm = max_grad_norm
    self.use_clipped_value_loss = use_clipped_value_loss
    self.desired_kl = desired_kl
    self.schedule = schedule
    self.learning_rate = learning_rate
    self.encoder_learning_rate = encoder_lr
    self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

  def act(self, obs: TensorDict) -> torch.Tensor:
    """Sample an action and store the current transition fields."""
    self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
    self.transition.actions = self.actor(obs, stochastic_output=True).detach()
    self.transition.values = self.critic(obs).detach()
    self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
    self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
    self.transition.observations = obs
    return self.transition.actions

  def process_env_step(
    self,
    obs: TensorDict,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict[str, torch.Tensor],
  ) -> None:
    """Store PPO data and the next target/velocity observations."""
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    self.source_encoder.update_normalization(obs[self.source_obs_group])
    self.target_encoder.update_normalization(obs[self.target_obs_group])
    if self.rnd:
      self.rnd.update_normalization(obs)

    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    self.transition.next_target_observation = obs[self.target_obs_group].detach().clone()
    self.transition.velocity_target = obs[self.velocity_obs_group].detach().clone()

    if self.rnd:
      self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
      self.transition.rewards += self.intrinsic_rewards

    if "time_outs" in extras:
      timeout = extras["time_outs"].to(self.device).unsqueeze(1)
      self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * timeout, 1)  # type: ignore

    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  def compute_returns(self, obs: TensorDict) -> None:
    """Compute GAE returns from the final critic observation."""
    storage = self.storage
    critic_hidden_state = self.critic.get_hidden_state()
    last_values = self.critic(obs).detach()
    self.critic.reset(hidden_state=critic_hidden_state)

    advantage = 0
    for step in reversed(range(storage.num_transitions_per_env)):
      next_values = last_values if step == storage.num_transitions_per_env - 1 else storage.values[step + 1]
      next_is_not_terminal = 1.0 - storage.dones[step].float()
      delta = storage.rewards[step] + next_is_not_terminal * self.gamma * next_values - storage.values[step]
      advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
      storage.returns[step] = advantage + storage.values[step]

    storage.advantages = storage.returns - storage.values
    if not self.normalize_advantage_per_mini_batch:
      storage.advantages = (storage.advantages - storage.advantages.mean()) / (storage.advantages.std() + 1e-8)

  @staticmethod
  @torch.no_grad()
  def _sinkhorn(scores: torch.Tensor, epsilon: float, iterations: int) -> torch.Tensor:
    """Compute balanced prototype assignments for a score matrix."""
    assignments = torch.exp(scores / epsilon).T
    assignments /= assignments.sum()
    num_prototypes, batch_size = assignments.shape
    for _ in range(iterations):
      assignments /= assignments.sum(dim=1, keepdim=True)
      assignments /= num_prototypes
      assignments /= assignments.sum(dim=0, keepdim=True)
      assignments /= batch_size
    return (assignments * batch_size).T

  def _update_encoders(
    self, batch: RolloutStorage.Batch
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one aligned source/target encoder update."""
    done = batch.dones.to(torch.bool).squeeze(-1)  # type: ignore[union-attr]
    valid = ~done
    if not torch.any(valid):
      zero = self.prototypes.new_zeros(())
      return zero, zero

    source_observations = batch.observations[self.source_obs_group][valid]  # type: ignore[index]
    target_observations = batch.next_target_observations[valid]
    velocity_targets = batch.velocity_targets[valid]

    predicted_velocity, source_latent = self.source_encoder.encode(source_observations)
    _, target_latent = self.target_encoder.encode(target_observations)
    assert predicted_velocity is not None

    with torch.no_grad():
      self.prototypes.copy_(F.normalize(self.prototypes, dim=-1, p=2))

    source_scores = source_latent @ self.prototypes.T
    target_scores = target_latent @ self.prototypes.T
    source_assignments = self._sinkhorn(source_scores, self.sinkhorn_epsilon, self.sinkhorn_iterations)
    target_assignments = self._sinkhorn(target_scores, self.sinkhorn_epsilon, self.sinkhorn_iterations)

    source_log_probs = F.log_softmax(source_scores / self.temperature, dim=-1)
    target_log_probs = F.log_softmax(target_scores / self.temperature, dim=-1)
    swap_loss = -0.5 * (
      source_assignments * target_log_probs + target_assignments * source_log_probs
    ).mean()
    estimation_loss = F.mse_loss(predicted_velocity, velocity_targets)
    encoder_loss = self.estimation_loss_coef * estimation_loss + self.swap_loss_coef * swap_loss

    self.encoder_optimizer.zero_grad()
    encoder_loss.backward()
    if self.is_multi_gpu:
      self.reduce_parameters(self._encoder_parameters)
    nn.utils.clip_grad_norm_(self._encoder_parameters, self.encoder_max_grad_norm)
    self.encoder_optimizer.step()

    # Keep scalar logging values on the training device. Calling ``item()`` here
    # would synchronize the host once for every auxiliary mini-batch.
    return estimation_loss.detach(), swap_loss.detach()

  def update(self) -> dict[str, float]:
    """Run PPO and HIM updates over the collected rollout."""
    mean_value_loss = torch.zeros((), device=self.device)
    mean_surrogate_loss = torch.zeros((), device=self.device)
    mean_entropy = torch.zeros((), device=self.device)
    mean_estimation_loss = torch.zeros((), device=self.device)
    mean_swap_loss = torch.zeros((), device=self.device)
    mean_rnd_loss: torch.Tensor | None = (
      torch.zeros((), device=self.device) if self.rnd else None
    )

    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      original_batch_size = batch.observations.batch_size[0]
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

      estimation_loss, swap_loss = self._update_encoders(batch)

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
      distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
      entropy = self.actor.output_entropy[:original_batch_size]

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
          kl_mean = torch.mean(kl)
          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size
          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          if self.is_multi_gpu:
            learning_rate = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(learning_rate, src=0)
            self.learning_rate = learning_rate.item()
          for parameter_group in self.ppo_optimizer.param_groups:
            parameter_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
      rnd_loss = self.rnd.compute_loss(batch.observations[:original_batch_size]) if self.rnd else None

      self.ppo_optimizer.zero_grad()
      loss.backward()
      if self.rnd:
        self.rnd.optimizer.zero_grad()
        assert rnd_loss is not None
        rnd_loss.backward()

      if self.is_multi_gpu:
        ppo_parameters: list[nn.Parameter] = list(self._ppo_parameters)
        if self.rnd:
          ppo_parameters.extend(self.rnd.parameters())
        self.reduce_parameters(ppo_parameters)
      nn.utils.clip_grad_norm_(self._ppo_parameters, self.max_grad_norm)
      self.ppo_optimizer.step()
      if self.rnd:
        self.rnd.optimizer.step()

      # Accumulate detached scalars on the GPU. Convert the final averages to
      # Python floats only once after all mini-batches have finished.
      mean_value_loss = mean_value_loss + value_loss.detach()
      mean_surrogate_loss = mean_surrogate_loss + surrogate_loss.detach()
      mean_entropy = mean_entropy + entropy.mean().detach()
      mean_estimation_loss = mean_estimation_loss + estimation_loss
      mean_swap_loss = mean_swap_loss + swap_loss
      if mean_rnd_loss is not None:
        assert rnd_loss is not None
        mean_rnd_loss = mean_rnd_loss + rnd_loss.detach()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    loss_names = ["value", "surrogate", "entropy", "estimation", "swap"]
    loss_totals = [
      mean_value_loss,
      mean_surrogate_loss,
      mean_entropy,
      mean_estimation_loss,
      mean_swap_loss,
    ]
    if mean_rnd_loss is not None:
      loss_names.append("rnd")
      loss_totals.append(mean_rnd_loss)
    loss_values = torch.stack(loss_totals).div(num_updates).cpu().tolist()
    loss_dict = dict(zip(loss_names, loss_values, strict=True))

    self.storage.clear()
    return loss_dict

  def train_mode(self) -> None:
    self.actor.train()
    self.critic.train()
    self.source_encoder.train()
    self.target_encoder.train()
    if self.rnd:
      self.rnd.train()

  def eval_mode(self) -> None:
    self.actor.eval()
    self.critic.eval()
    self.source_encoder.eval()
    self.target_encoder.eval()
    if self.rnd:
      self.rnd.eval()

  def save(self) -> dict:
    """Save PPO, target encoder, prototypes, and both optimizer states."""
    saved_dict = {
      "actor_state_dict": self._raw_actor.state_dict(),
      "critic_state_dict": self._raw_critic.state_dict(),
      "target_encoder_state_dict": self._raw_target_encoder.state_dict(),
      "prototypes": self.prototypes.detach().clone(),
      "optimizer_state_dict": self.ppo_optimizer.state_dict(),
      "ppo_optimizer_state_dict": self.ppo_optimizer.state_dict(),
      "encoder_optimizer_state_dict": self.encoder_optimizer.state_dict(),
    }
    if self.rnd:
      saved_dict["rnd_state_dict"] = self.rnd.state_dict()
      saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
    return saved_dict

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if load_cfg is None:
      load_cfg = {
        "actor": True,
        "critic": True,
        "target_encoder": True,
        "prototypes": True,
        "optimizer": True,
        "encoder_optimizer": True,
        "iteration": True,
        "rnd": True,
      }

    if load_cfg.get("actor"):
      self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
    if load_cfg.get("critic"):
      self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
    if load_cfg.get("target_encoder") and "target_encoder_state_dict" in loaded_dict:
      self._raw_target_encoder.load_state_dict(loaded_dict["target_encoder_state_dict"], strict=strict)
    if load_cfg.get("prototypes") and "prototypes" in loaded_dict:
      with torch.no_grad():
        self.prototypes.copy_(loaded_dict["prototypes"].to(self.device))
    if load_cfg.get("optimizer"):
      optimizer_state = loaded_dict.get("ppo_optimizer_state_dict", loaded_dict.get("optimizer_state_dict"))
      if optimizer_state is not None:
        self.ppo_optimizer.load_state_dict(optimizer_state)
        self.learning_rate = self.ppo_optimizer.param_groups[0]["lr"]
    if load_cfg.get("encoder_optimizer") and "encoder_optimizer_state_dict" in loaded_dict:
      self.encoder_optimizer.load_state_dict(loaded_dict["encoder_optimizer_state_dict"])
    if load_cfg.get("rnd") and self.rnd:
      self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
      self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
    return load_cfg.get("iteration", False)

  def get_policy(self) -> HIMActor:
    return self._raw_actor

  def compile(self, mode: str | None = None) -> None:
    # The actor owns the source encoder, so it is compiled as one policy module.
    self.actor = compile_model(self._raw_actor, mode)  # type: ignore
    self.critic = compile_model(self._raw_critic, mode)  # type: ignore

  @staticmethod
  def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> HIM_PPO:
    """Construct source/target encoders, HIM actor, critic, and storage."""
    algorithm_cfg = copy.deepcopy(cfg["algorithm"])
    actor_cfg = copy.deepcopy(cfg["actor"])
    critic_cfg = copy.deepcopy(cfg["critic"])

    alg_class = resolve_callable(algorithm_cfg.pop("class_name"))
    actor_class = resolve_callable(actor_cfg.pop("class_name"))
    critic_class = resolve_callable(critic_cfg.pop("class_name"))

    # MjlabOnPolicyRunner normally removes these optional model fields. Keep
    # construction correct when this method is called directly as well.
    for model_cfg in (actor_cfg, critic_cfg):
      if model_cfg.get("cnn_cfg") is None:
        model_cfg.pop("cnn_cfg", None)
      if model_cfg.get("distribution_cfg") is None:
        model_cfg.pop("distribution_cfg", None)
      if model_cfg.get("rnn_type") is None:
        for key in ("rnn_type", "rnn_hidden_dim", "rnn_num_layers"):
          model_cfg.pop(key, None)

    source_encoder_cfg = algorithm_cfg.pop("source_encoder")
    target_encoder_cfg = algorithm_cfg.pop("target_encoder")
    source_obs_group = algorithm_cfg.pop("source_obs_group", "actor")
    target_obs_group = algorithm_cfg.pop("target_obs_group", "target_encoder")
    velocity_obs_group = algorithm_cfg.pop("velocity_obs_group", "velocity")

    obs_groups = resolve_obs_groups(obs, copy.deepcopy(cfg["obs_groups"]), ["actor", "critic"])
    cfg["obs_groups"] = obs_groups
    algorithm_cfg = resolve_rnd_config(algorithm_cfg, obs, obs_groups, env)
    algorithm_cfg = resolve_symmetry_config(algorithm_cfg, env)

    if source_obs_group not in obs or target_obs_group not in obs or velocity_obs_group not in obs:
      raise KeyError(
        "HIM observation groups are missing. "
        f"Required: {source_obs_group}, {target_obs_group}, {velocity_obs_group}; "
        f"available: {list(obs.keys())}"
      )

    source_observation = obs[source_obs_group]
    target_observation = obs[target_obs_group]
    velocity_observation = obs[velocity_obs_group]
    if source_observation.ndim != 3:
      raise ValueError(
        f"Source group '{source_obs_group}' must have shape [N, H, D], got {tuple(source_observation.shape)}"
      )
    if target_observation.ndim != 2:
      raise ValueError(
        f"Target group '{target_obs_group}' must have shape [N, D], got {tuple(target_observation.shape)}"
      )
    if velocity_observation.ndim != 2 or velocity_observation.shape[-1] != 3:
      raise ValueError(
        f"Velocity group '{velocity_obs_group}' must have shape [N, 3], got {tuple(velocity_observation.shape)}"
      )
    if source_observation.shape[-1] != target_observation.shape[-1]:
      raise ValueError(
        "Source and target one-step observations must have the same dimension for this HIM configuration: "
        f"{source_observation.shape[-1]} != {target_observation.shape[-1]}"
      )

    source_encoder_class = resolve_callable(source_encoder_cfg.pop("class_name"))
    target_encoder_class = resolve_callable(target_encoder_cfg.pop("class_name"))
    source_encoder = source_encoder_class(
      input_dim=int(source_observation.shape[-1]),
      history_length=int(source_observation.shape[1]),
      **source_encoder_cfg,
    ).to(device)
    target_encoder = target_encoder_class(
      input_dim=int(target_observation.shape[-1]),
      history_length=1,
      **target_encoder_cfg,
    ).to(device)

    actor = actor_class(
      obs,
      obs_groups,
      "actor",
      env.num_actions,
      source_encoder=source_encoder,
      source_obs_group=source_obs_group,
      **actor_cfg,
    ).to(device)
    critic = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)

    storage = HIMRolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      target_obs_group=target_obs_group,
      velocity_obs_group=velocity_obs_group,
      device=device,
    )

    # These fields are consumed by HIM_PPO.__init__. ``share_cnn_encoders`` is
    # a standard PPO option and has no meaning for this actor implementation.
    algorithm_cfg.pop("share_cnn_encoders", None)
    cfg["algorithm"] = algorithm_cfg
    algorithm = alg_class(
      source_encoder=source_encoder,
      target_encoder=target_encoder,
      actor=actor,
      critic=critic,
      storage=storage,
      source_obs_group=source_obs_group,
      target_obs_group=target_obs_group,
      velocity_obs_group=velocity_obs_group,
      device=device,
      multi_gpu_cfg=cfg.get("multi_gpu"),
      **algorithm_cfg,
    )
    algorithm.compile(cfg.get("torch_compile_mode"))
    return algorithm

  def broadcast_parameters(self) -> None:
    """Broadcast all HIM and PPO parameters in distributed training."""
    model_params = [
      self._raw_actor.state_dict(),
      self._raw_critic.state_dict(),
      self._raw_target_encoder.state_dict(),
      self.prototypes.detach().clone(),
    ]
    if self.rnd:
      model_params.append(self.rnd.predictor.state_dict())
    torch.distributed.broadcast_object_list(model_params, src=0)
    self._raw_actor.load_state_dict(model_params[0])
    self._raw_critic.load_state_dict(model_params[1])
    self._raw_target_encoder.load_state_dict(model_params[2])
    with torch.no_grad():
      self.prototypes.copy_(model_params[3].to(self.device))
    if self.rnd:
      self.rnd.predictor.load_state_dict(model_params[4])

  def reduce_parameters(self, parameters: Iterable[nn.Parameter]) -> None:
    """Average non-null gradients across distributed workers."""
    parameters = list(parameters)
    gradients = [parameter.grad.reshape(-1) for parameter in parameters if parameter.grad is not None]
    if not gradients:
      return
    all_gradients = torch.cat(gradients)
    torch.distributed.all_reduce(all_gradients, op=torch.distributed.ReduceOp.SUM)
    all_gradients /= self.gpu_world_size
    offset = 0
    for parameter in parameters:
      if parameter.grad is not None:
        numel = parameter.numel()
        parameter.grad.data.copy_(all_gradients[offset : offset + numel].view_as(parameter.grad.data))
        offset += numel
