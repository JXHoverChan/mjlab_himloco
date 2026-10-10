"""Linux joystick input for interactive policy playback.

The adapter uses the kernel's legacy ``/dev/input/js*`` interface directly,
so play mode does not require an additional Python joystick package. It maps
three normalized axes to a velocity command and leaves the trained policy in
charge of producing joint actions.
"""

from __future__ import annotations

import errno
import fcntl
import os
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
_EVENT = struct.Struct("<IhBB")


def _ioc(direction: int, type_: int, number: int, size: int) -> int:
  """Build a Linux ioctl request number."""
  return (
    (direction << 30)
    | (size << 16)
    | (type_ << 8)
    | number
  )


def _js_ioctl(number: int, size: int) -> int:
  return _ioc(2, ord("j"), number, size)  # _IOC_READ


def _read_u8(fd: int, request: int) -> int:
  value = bytearray(1)
  fcntl.ioctl(fd, request, value)
  return value[0]


def _apply_deadzone(value: float, deadzone: float) -> float:
  """Apply a rescaled symmetric deadzone while preserving full stick range."""
  magnitude = abs(value)
  if magnitude <= deadzone:
    return 0.0
  return (1.0 if value >= 0.0 else -1.0) * (magnitude - deadzone) / (
    1.0 - deadzone
  )


def _scale_axis(value: float, lower: float, upper: float) -> float:
  """Map a normalized signed axis into an asymmetric command range."""
  if not lower <= 0.0 <= upper:
    raise ValueError(
      f"Joystick command ranges must include zero, got ({lower}, {upper})."
    )
  return value * (upper if value >= 0.0 else abs(lower))


def map_axes_to_command(
  axes: Mapping[int, float],
  command_ranges: tuple[tuple[float, float], ...],
  *,
  forward_axis: int,
  lateral_axis: int,
  yaw_axis: int,
  invert_forward: bool,
  invert_lateral: bool,
  invert_yaw: bool,
  deadzone: float,
  scale: float,
) -> tuple[float, float, float]:
  """Convert three joystick axes into ``(vx, vy, wz)`` command values."""
  if len(command_ranges) != 3:
    raise ValueError(f"Expected three command ranges, got {len(command_ranges)}.")
  if not 0.0 <= deadzone < 1.0:
    raise ValueError(f"deadzone must be in [0, 1), got {deadzone}.")
  if scale < 0.0:
    raise ValueError(f"scale must be non-negative, got {scale}.")

  axis_indices = (forward_axis, lateral_axis, yaw_axis)
  inversions = (invert_forward, invert_lateral, invert_yaw)
  values: list[float] = []
  for axis, invert, (lower, upper) in zip(
    axis_indices, inversions, command_ranges, strict=True
  ):
    value = float(axes.get(axis, 0.0))
    value = _apply_deadzone(value, deadzone) * scale
    value = max(-1.0, min(1.0, value))
    if invert:
      value = -value
    values.append(_scale_axis(value, lower, upper))
  return values[0], values[1], values[2]


@dataclass(frozen=True)
class JoystickSettings:
  """Physical-axis mapping and command scaling for play mode."""

  command_ranges: tuple[tuple[float, float], ...]
  forward_axis: int = 1
  lateral_axis: int = 0
  yaw_axis: int = 2
  invert_forward: bool = True
  invert_lateral: bool = True
  invert_yaw: bool = True
  deadzone: float = 0.08
  scale: float = 1.0
  zero_button: int = 0


class LinuxJoystick:
  """Read a Linux ``/dev/input/js*`` device without third-party packages."""

  def __init__(self, device: str, settings: JoystickSettings) -> None:
    self.device = device
    self.settings = settings
    self._fd: int | None = None
    self._pending = bytearray()
    self._axes: dict[int, float] = {}
    self._buttons: dict[int, bool] = {}

    try:
      self._fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
    except PermissionError as exc:
      raise PermissionError(
        f"Cannot read {device}. Add the current user to the 'input' group "
        "or grant read permission to this device."
      ) from exc
    except FileNotFoundError as exc:
      raise FileNotFoundError(
        f"Joystick device not found: {device}. Check /dev/input/js*."
      ) from exc

    try:
      self.name = self._read_name()
      self.axis_count = _read_u8(self._fd, _js_ioctl(0x11, 1))
      self.button_count = _read_u8(self._fd, _js_ioctl(0x12, 1))
      self._validate_axis_settings()
      self._poll_events()
    except Exception:
      self.close()
      raise

    print(
      f"[INFO] Joystick: {self.name} ({self.device}), "
      f"axes={self.axis_count}, buttons={self.button_count}"
    )

  def _read_name(self) -> str:
    assert self._fd is not None
    buffer = bytearray(128)
    fcntl.ioctl(self._fd, _js_ioctl(0x13, len(buffer)), buffer)
    return bytes(buffer).split(b"\0", 1)[0].decode("utf-8", errors="replace")

  def _validate_axis_settings(self) -> None:
    for label, axis in (
      ("forward", self.settings.forward_axis),
      ("lateral", self.settings.lateral_axis),
      ("yaw", self.settings.yaw_axis),
    ):
      if not 0 <= axis < self.axis_count:
        raise ValueError(
          f"Joystick {label} axis {axis} is unavailable; device has "
          f"{self.axis_count} axes."
        )
    if self.settings.zero_button >= self.button_count:
      raise ValueError(
        f"Joystick zero button {self.settings.zero_button} is unavailable; "
        f"device has {self.button_count} buttons. Use -1 to disable it."
      )

  def _poll_events(self) -> None:
    assert self._fd is not None
    while True:
      try:
        data = os.read(self._fd, 4096)
      except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
          break
        raise RuntimeError(f"Failed to read joystick {self.device}: {exc}") from exc
      if not data:
        raise RuntimeError(f"Joystick disconnected: {self.device}")
      self._pending.extend(data)

    event_size = _EVENT.size
    complete_size = len(self._pending) - len(self._pending) % event_size
    for offset in range(0, complete_size, event_size):
      _, raw_value, event_type, number = _EVENT.unpack_from(self._pending, offset)
      event_type &= ~JS_EVENT_INIT
      if event_type == JS_EVENT_AXIS:
        normalized = max(-1.0, min(1.0, raw_value / 32767.0))
        self._axes[number] = normalized
      elif event_type == JS_EVENT_BUTTON:
        self._buttons[number] = raw_value != 0
    del self._pending[:complete_size]

  def update_command(self, command: Any, env_id: int) -> None:
    """Poll the device and update one environment's 3-D velocity command."""
    if command.ndim != 2 or command.shape[1] != 3:
      raise ValueError(
        f"Joystick requires a command tensor with shape (N, 3), got {command.shape}."
      )
    if not 0 <= env_id < command.shape[0]:
      raise IndexError(f"Joystick environment index {env_id} is out of range.")

    self._poll_events()
    if self.settings.zero_button >= 0 and self._buttons.get(
      self.settings.zero_button, False
    ):
      values = (0.0, 0.0, 0.0)
    else:
      values = map_axes_to_command(
        self._axes,
        self.settings.command_ranges,
        forward_axis=self.settings.forward_axis,
        lateral_axis=self.settings.lateral_axis,
        yaw_axis=self.settings.yaw_axis,
        invert_forward=self.settings.invert_forward,
        invert_lateral=self.settings.invert_lateral,
        invert_yaw=self.settings.invert_yaw,
        deadzone=self.settings.deadzone,
        scale=self.settings.scale,
      )

    command[env_id, 0] = values[0]
    command[env_id, 1] = values[1]
    command[env_id, 2] = values[2]

  def close(self) -> None:
    if self._fd is not None:
      os.close(self._fd)
      self._fd = None


class JoystickCommandEnv:
  """Proxy an RL environment and inject joystick commands before observation."""

  def __init__(self, env: Any, joystick: LinuxJoystick, env_id: int = 0) -> None:
    self.env = env
    self.joystick = joystick
    self.env_id = env_id

  @property
  def unwrapped(self):
    return self.env.unwrapped

  def __getattr__(self, name: str) -> Any:
    return getattr(self.env, name)

  def _update_command(self) -> None:
    command = self.unwrapped.command_manager.get_command("twist")
    if command is None:
      raise ValueError("Joystick mode requires the 'twist' command term.")
    self.joystick.update_command(command, self.env_id)

  def get_observations(self):
    self._update_command()
    return self.env.get_observations()

  def step(self, actions):
    # Keep the command correct both for callers that use get_observations()
    # and for callers that invoke step() directly.
    self._update_command()
    result = self.env.step(actions)
    self._update_command()
    return result

  def close(self) -> None:
    try:
      self.joystick.close()
    finally:
      self.env.close()
