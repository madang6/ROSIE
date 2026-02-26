"""
Learned-policy controller — wraps a user-provided model (ONNX or callable).

The model receives a flat observation vector built from the VehicleState
and returns an action vector that maps to a ControlCommand.

Supports:
  - ONNX Runtime models (``*.onnx``)
  - Any Python callable with signature  ``obs → action``
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rosie.controllers.base import Controller
from rosie.state import VehicleState, ControlObjective, ControlCommand

log = logging.getLogger("rosie.controllers.learned")


class LearnedController(Controller):
    """Run a learned policy as the control loop.

    Parameters
    ----------
    model :
        Either a path to an ``.onnx`` file *or* a Python callable
        ``(np.ndarray,) -> np.ndarray`` that maps observation → action.
    obs_builder :
        Optional function ``(VehicleState, ControlObjective) -> np.ndarray``
        that constructs the observation vector.  Defaults to
        ``default_obs_builder``.
    action_interpreter :
        Optional function ``(np.ndarray,) -> ControlCommand`` that converts
        the raw action output into a ControlCommand.  Defaults to
        ``default_action_interpreter``.
    """

    def __init__(
        self,
        model: str | Path | Callable[[np.ndarray], np.ndarray],
        obs_builder: Callable[[VehicleState, ControlObjective], np.ndarray] | None = None,
        action_interpreter: Callable[[np.ndarray], ControlCommand] | None = None,
    ) -> None:
        self._infer: Callable[[np.ndarray], np.ndarray]

        if callable(model) and not isinstance(model, (str, Path)):
            self._infer = model
        else:
            self._infer = _load_onnx(Path(model))

        self._obs_builder = obs_builder or default_obs_builder
        self._action_interpreter = action_interpreter or default_action_interpreter

    @property
    def name(self) -> str:
        return "learned"

    def compute(
        self,
        state: VehicleState,
        objective: ControlObjective,
    ) -> ControlCommand:
        obs = self._obs_builder(state, objective)
        action = self._infer(obs)
        return self._action_interpreter(action)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def default_obs_builder(
    state: VehicleState,
    objective: ControlObjective,
) -> np.ndarray:
    """Build a flat observation from state + objective.

    Layout (19-dim):
      [0:3]   position  (or zeros)
      [3:7]   orientation quaternion  (or identity)
      [7:10]  linear velocity  (or zeros)
      [10:13] angular velocity  (or zeros)
      [13:16] target position  (or zeros)
      [16:19] target velocity  (or zeros)
    """
    pos = state.position if state.position is not None else np.zeros(3)
    ori = state.orientation if state.orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])
    lv = state.linear_velocity if state.linear_velocity is not None else np.zeros(3)
    av = state.angular_velocity if state.angular_velocity is not None else np.zeros(3)
    tp = objective.target_position if objective.target_position is not None else np.zeros(3)
    tv = objective.target_velocity if objective.target_velocity is not None else np.zeros(3)
    return np.concatenate([pos, ori, lv, av, tp, tv]).astype(np.float32)


def default_action_interpreter(action: np.ndarray) -> ControlCommand:
    """Interpret a 6-dim action as [vx, vy, vz, wx, wy, wz]."""
    cmd = ControlCommand()
    if len(action) >= 3:
        cmd.linear_velocity = action[:3].astype(float)
    if len(action) >= 6:
        cmd.angular_velocity = action[3:6].astype(float)
    return cmd


# ---------------------------------------------------------------------------
# ONNX loading
# ---------------------------------------------------------------------------


def _load_onnx(path: Path) -> Callable[[np.ndarray], np.ndarray]:
    """Return an inference callable backed by ONNX Runtime."""
    try:
        import onnxruntime as ort  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX models: pip install onnxruntime"
        ) from exc

    session = ort.InferenceSession(str(path))
    input_name = session.get_inputs()[0].name

    def _run(obs: np.ndarray) -> np.ndarray:
        result = session.run(None, {input_name: obs.reshape(1, -1).astype(np.float32)})
        return result[0].flatten()

    log.info("Loaded ONNX model from %s", path)
    return _run
