from scripts.joystick import map_axes_to_command


def test_map_axes_to_command_uses_axis_inversion_and_asymmetric_ranges() -> None:
  command = map_axes_to_command(
    {0: 0.5, 1: -1.0, 2: -0.5},
    ((-1.0, 2.0), (-1.0, 1.0), (-1.0, 1.0)),
    forward_axis=1,
    lateral_axis=0,
    yaw_axis=2,
    invert_forward=True,
    invert_lateral=False,
    invert_yaw=False,
    deadzone=0.0,
    scale=1.0,
  )

  assert command == (2.0, 0.5, -0.5)


def test_map_axes_to_command_rescales_deadzone() -> None:
  command = map_axes_to_command(
    {0: 0.05, 1: 0.5, 2: 0.08},
    ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    forward_axis=1,
    lateral_axis=0,
    yaw_axis=2,
    invert_forward=False,
    invert_lateral=False,
    invert_yaw=False,
    deadzone=0.1,
    scale=1.0,
  )

  assert command[1] == 0.0
  assert command[2] == 0.0
  assert abs(command[0] - (0.5 - 0.1) / 0.9) < 1e-6


def test_lateral_axis_can_be_inverted_for_body_frame_commands() -> None:
  command = map_axes_to_command(
    {0: 1.0, 1: 0.0, 2: 0.0},
    ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)),
    forward_axis=1,
    lateral_axis=0,
    yaw_axis=2,
    invert_forward=True,
    invert_lateral=True,
    invert_yaw=False,
    deadzone=0.0,
    scale=1.0,
  )

  assert command == (0.0, -1.0, 0.0)
