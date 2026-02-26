"""
Abstract controller interface.

Every controller takes (VehicleState, ControlObjective, ControlContext) and
returns a ControlCommand.  The Heart calls ``compute()`` at the control
loop rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from rosie.state import VehicleState, ControlObjective, ControlCommand


@dataclass
class ControlContext:
    """Extra context the Heart passes to controllers each tick.

    Simple controllers (PID, passthrough) can ignore this.
    Trajectory-tracking controllers (MPC, learned policies) need it.
    """

    # Seconds since the current objective was set (trajectory time)
    t_trajectory: float = 0.0

    # Previous control command (for derivative / rate-limiting)
    prev_command: ControlCommand | None = None

    # Previous body-rate input [thrust, p, q, r] (from VehicleRatesSetpoint feedback)
    prev_input: np.ndarray | None = None

    # Image from state (convenience — avoids controllers reaching into VehicleState.images)
    image: np.ndarray | None = None


class Controller(ABC):
    """Base class for all ROSIE controllers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier shown to the agent."""

    @abstractmethod
    def compute(
        self,
        state: VehicleState,
        objective: ControlObjective,
        ctx: ControlContext | None = None,
    ) -> ControlCommand:
        """Produce a control command from current state + objective.

        Called at the Heart's control rate (e.g. 50 Hz).  Implementations
        should be fast — no blocking I/O.

        Parameters
        ----------
        state : VehicleState
            Current fused state estimate.
        objective : ControlObjective
            What the agent/user wants.
        ctx : ControlContext | None
            Extra context (trajectory time, previous input, image).
            Simple controllers may ignore this.
        """

    def reset(self) -> None:
        """Reset internal state (e.g. integrator terms).

        Called whenever the control objective changes mode.
        """
