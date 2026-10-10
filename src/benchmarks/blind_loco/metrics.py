"""Device-side trial statistics. Failure samples cannot disappear from success rates."""

from __future__ import annotations

import torch

from .protocol import HOLD_SECONDS, STEP_DT, XY_TOLERANCE, YAW_TOLERANCE


def tracking_errors(velocity, yaw, command):
  return torch.linalg.vector_norm(velocity-command[:, :2], dim=-1), (yaw-command[:, 2]).abs()


def tracking_tolerances(command):
  speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
  linear = (.25*speed).clamp(min=.05, max=XY_TOLERANCE)
  angular = (.25*command[:, 2].abs()).clamp(min=.02, max=YAW_TOLERANCE)
  return linear, angular


class TrialStatistics:
  """Public accumulator; all masks are explicit and no observations are repaired."""

  def __init__(self, num_envs, device):
    self.n = num_envs
    self.device = device
    names = ("samples", "xy_error", "yaw_error", "vx_bias", "vy_bias", "wz_bias",
             "steady_samples", "steady_xy_error", "steady_yaw_error",
             "stall_s", "max_stall_s", "path_m", "yaw_angle_rad", "target_yaw_angle_rad",
             "contact_samples", "slip_sum", "power_sum", "energy_j", "vertical_velocity_sq",
             "action_change_sum", "soft_limit_sum", "touchdowns", "foothold_shift_m")
    self.values = {name: torch.zeros(num_envs, device=device) for name in names}
    self.stall_run = torch.zeros(num_envs, device=device)
    self.hold = torch.zeros(num_envs, dtype=torch.long, device=device)
    self.previous_command = torch.full((num_envs, 3), float("nan"), device=device)
    self.responses = torch.zeros((num_envs, 5), dtype=torch.bool, device=device)
    self.latencies = torch.full((num_envs, 5), float("nan"), device=device)
    self.foot_contacts = torch.zeros((num_envs, 4), device=device)
    self.foot_slip_sum = torch.zeros_like(self.foot_contacts)
    self.tracking_steps = torch.zeros(num_envs, dtype=torch.long, device=device)

  def update(self, *, velocity, yaw, command, usable, failed, steady,
             stage, stage_elapsed, path_increment=None, yaw_increment=None,
             contact=None, slip=None, power=None, vertical_velocity=None,
             action_change=None, soft_limits=None, touchdowns=None, foothold_shift=None):
    xy_error, yaw_error = tracking_errors(velocity, yaw, command)
    linear_tolerance, angular_tolerance = tracking_tolerances(command)
    accurate = (xy_error <= linear_tolerance) & (yaw_error <= angular_tolerance)
    successful = usable & ~failed & accurate
    self.tracking_steps += successful.long()
    stage = torch.as_tensor(stage, dtype=torch.long, device=self.device).expand(self.n)
    stage_elapsed = torch.as_tensor(stage_elapsed, device=self.device).expand(self.n)
    changed = (command != self.previous_command).any(-1)
    self.hold = torch.where(changed | ~successful, 0, self.hold)
    self.hold += successful.long()
    ids = torch.arange(self.n, device=self.device)
    new_response = successful & (self.hold >= round(HOLD_SECONDS/STEP_DT)) & ~self.responses[ids, stage]
    latency = stage_elapsed - (round(HOLD_SECONDS/STEP_DT)-1)*STEP_DT
    self.latencies[ids, stage] = torch.where(new_response, latency, self.latencies[ids, stage])
    self.responses[ids, stage] |= new_response
    self.previous_command.copy_(command)

    speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
    # Avoid division for a zero command without assigning meaning to 0/0.
    divisor = torch.where(speed > 0, speed, torch.ones_like(speed))
    progress = (velocity * command[:, :2]).sum(-1) / divisor
    stalled = ((speed > 0) & (progress < .2*speed)) | (
      (speed == 0) & (command[:, 2] != 0) &
      (yaw*command[:, 2].sign() < .2*command[:, 2].abs()))
    stalled &= usable
    self.stall_run = torch.where(stalled, self.stall_run + STEP_DT, 0.)
    self.values["max_stall_s"] = torch.maximum(self.values["max_stall_s"], self.stall_run)
    updates = {
      "samples": torch.ones_like(xy_error), "xy_error": xy_error, "yaw_error": yaw_error,
      "vx_bias": velocity[:, 0]-command[:, 0], "vy_bias": velocity[:, 1]-command[:, 1],
      "wz_bias": yaw-command[:, 2],
      "steady_samples": steady.float(), "steady_xy_error": xy_error*steady,
      "steady_yaw_error": yaw_error*steady,
      "stall_s": stalled.float()*STEP_DT, "target_yaw_angle_rad": command[:, 2]*STEP_DT,
    }
    for name, value in (("path_m", path_increment), ("yaw_angle_rad", yaw_increment),
                        ("power_sum", power), ("energy_j", None if power is None else power*STEP_DT),
                        ("action_change_sum", action_change), ("soft_limit_sum", soft_limits),
                        ("touchdowns", touchdowns), ("foothold_shift_m", foothold_shift)):
      if value is not None:
        updates[name] = value
    if vertical_velocity is not None:
      updates["vertical_velocity_sq"] = vertical_velocity.square()
    if contact is not None and slip is not None:
      updates["contact_samples"] = contact.sum(-1).float()
      updates["slip_sum"] = (slip*contact).sum(-1)
      self.foot_contacts += torch.where(usable[:, None], contact.float(), 0.)
      self.foot_slip_sum += torch.where(usable[:, None], slip*contact, 0.)
    for name, value in updates.items():
      self.values[name] += torch.where(usable, value, 0.)

  def export(self, durations, stage_counts):
    values = {name: value.cpu().numpy() for name, value in self.values.items()}
    responses = self.responses.cpu().numpy()
    latencies = self.latencies.cpu().numpy()
    contacts = self.foot_contacts.cpu().numpy()
    slips = self.foot_slip_sum.cpu().numpy()
    tracking_steps = self.tracking_steps.cpu().numpy()
    rows = []
    for i in range(self.n):
      count, steady = float(values["samples"][i]), float(values["steady_samples"][i])
      row = {name: float(value[i]) for name, value in values.items()}
      for name in ("xy_error", "yaw_error", "vx_bias", "vy_bias", "wz_bias"):
        row[name] = row[name]/count if count else None
      for name in ("steady_xy_error", "steady_yaw_error"):
        row[name] = row[name]/steady if steady else None
      # Count discrete control intervals exactly instead of repeatedly adding a
      # float32 timestep (which can make perfect tracking exceed 100%).
      row["tracking_time_s"] = int(tracking_steps[i]) * STEP_DT
      row["tracking_fraction"] = int(tracking_steps[i]) / round(durations[i] / STEP_DT)
      row["mean_mechanical_power_w"] = row["power_sum"]/count if count else None
      row["rms_vertical_velocity_m_s"] = (row["vertical_velocity_sq"]/count)**.5 if count else None
      row["mean_action_change_sq"] = row["action_change_sum"]/count if count else None
      row["soft_joint_limit_fraction"] = row["soft_limit_sum"]/count if count else None
      row["mean_stance_slip_m_s"] = row["slip_sum"]/row["contact_samples"] if row["contact_samples"] else None
      row["yaw_completion_ratio"] = (row["yaw_angle_rad"]/row["target_yaw_angle_rad"]
                                     if abs(row["target_yaw_angle_rad"]) > 1e-6 else None)
      row["responses"] = responses[i, :stage_counts[i]].tolist()
      row["response_latencies_s"] = [float(x) if flag else None
                                     for x, flag in zip(latencies[i], responses[i])][:stage_counts[i]]
      row["response_success_fraction"] = sum(row["responses"])/stage_counts[i]
      good_latencies = [x for x in row["response_latencies_s"] if x is not None]
      row["mean_response_latency_s"] = sum(good_latencies)/len(good_latencies) if good_latencies else None
      for leg, name in enumerate(("FR", "FL", "RR", "RL")):
        row[f"{name}_stance_fraction"] = float(contacts[i, leg]/count) if count else None
        row[f"{name}_slip_m_s"] = float(slips[i, leg]/contacts[i, leg]) if contacts[i, leg] else None
      rows.append(row)
    return rows
