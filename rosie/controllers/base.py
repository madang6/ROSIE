"""
Abstract controller interface.

Every controller takes (VehicleState, ControlObjective) and returns a
ControlCommand.  The Heart calls ``compute()`` at the control loop rate.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rosie.state import VehicleState, ControlObjective, ControlCommand


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
    ) -> ControlCommand:
        """Produce a control command from current state + objective.

        Called at the Heart's control rate (e.g. 50 Hz).  Implementations
        should be fast — no blocking I/O.
        """

    def reset(self) -> None:
        """Reset internal state (e.g. integrator terms).

        Called whenever the control objective changes mode.
        """
