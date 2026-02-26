"""
Passthrough controller — directly forwards the objective as a command.

Used for velocity-mode teleop and for cases where the agent or an external
planner already provides fully-resolved commands.
"""

from __future__ import annotations

import numpy as np

from rosie.controllers.base import Controller, ControlContext
from rosie.state import VehicleState, ControlObjective, ControlCommand


class PassthroughController(Controller):
    """Zero-intelligence relay: objective → command with no feedback."""

    @property
    def name(self) -> str:
        return "passthrough"

    def compute(
        self,
        state: VehicleState,
        objective: ControlObjective,
        ctx: ControlContext | None = None,
    ) -> ControlCommand:
        cmd = ControlCommand()

        if objective.mode == "velocity":
            if objective.target_velocity is not None:
                cmd.linear_velocity = objective.target_velocity.copy()
            if objective.target_yaw_rate is not None:
                cmd.angular_velocity = np.array([0.0, 0.0, objective.target_yaw_rate])

        elif objective.mode == "position":
            # For passthrough, just forward the target as a position command
            if objective.target_position is not None:
                cmd.position = objective.target_position.copy()
            cmd.yaw = objective.target_yaw

        return cmd
