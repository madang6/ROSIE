"""
Tests for rosie.flight — FlightPhase, ReadyGate, FlightRecorder.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.testing as npt

from rosie.state import VehicleState, ControlCommand
from rosie.flight import FlightPhase, ReadyGate, FlightRecorder


# ── FlightPhase ──────────────────────────────────────────────────────────────


class TestFlightPhase:
    def test_enum_values(self):
        assert FlightPhase.INIT.value == 0
        assert FlightPhase.READY.value == 1
        assert FlightPhase.ACTIVE.value == 2
        assert FlightPhase.HOLD.value == 3
        assert FlightPhase.LAND.value == 4

    def test_all_phases_exist(self):
        names = {p.name for p in FlightPhase}
        assert names == {"INIT", "READY", "ACTIVE", "HOLD", "LAND"}


# ── ReadyGate ────────────────────────────────────────────────────────────────


class TestReadyGate:
    def test_default_passes_with_enough_updates(self):
        gate = ReadyGate()
        state = VehicleState(
            stamp=time.monotonic(),
            position=np.array([0.0, 0.0, 0.0]),
        )
        passed, reason = gate.check(state, update_count=10)
        assert passed
        assert reason == "ready"

    def test_fails_without_enough_updates(self):
        gate = ReadyGate(min_state_updates=10)
        state = VehicleState(position=np.array([0.0, 0.0, 0.0]))
        passed, reason = gate.check(state, update_count=5)
        assert not passed
        assert "waiting for state updates" in reason

    def test_fails_without_position(self):
        gate = ReadyGate(min_state_updates=0)
        state = VehicleState()
        passed, reason = gate.check(state, update_count=100)
        assert not passed
        assert "no position" in reason

    def test_altitude_check_passes(self):
        gate = ReadyGate(
            target_altitude=10.0,
            altitude_tolerance=0.5,
            min_state_updates=0,
        )
        state = VehicleState(position=np.array([0.0, 0.0, 10.2]))
        passed, _ = gate.check(state, update_count=100)
        assert passed

    def test_altitude_check_fails(self):
        gate = ReadyGate(
            target_altitude=10.0,
            altitude_tolerance=0.5,
            min_state_updates=0,
        )
        state = VehicleState(position=np.array([0.0, 0.0, 5.0]))
        passed, reason = gate.check(state, update_count=100)
        assert not passed
        assert "altitude error" in reason

    def test_orientation_check_passes(self):
        gate = ReadyGate(
            target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            orientation_tolerance=0.1,
            min_state_updates=0,
        )
        state = VehicleState(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        passed, _ = gate.check(state, update_count=100)
        assert passed

    def test_orientation_check_fails(self):
        gate = ReadyGate(
            target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            orientation_tolerance=0.01,
            min_state_updates=0,
        )
        state = VehicleState(
            position=np.array([0.0, 0.0, 0.0]),
            orientation=np.array([0.0, 0.0, 0.707, 0.707]),  # 90 deg off
        )
        passed, reason = gate.check(state, update_count=100)
        assert not passed
        assert "orientation error" in reason


# ── FlightRecorder ───────────────────────────────────────────────────────────


class TestFlightRecorder:
    def test_empty_recorder(self):
        rec = FlightRecorder()
        assert rec.tick_count == 0
        d = rec.as_dict()
        assert d["tick_count"] == 0

    def test_record_single_tick(self):
        rec = FlightRecorder()
        state = VehicleState(
            stamp=time.monotonic(),
            position=np.array([1.0, 2.0, 3.0]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            linear_velocity=np.array([0.5, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        cmd = ControlCommand(
            linear_velocity=np.array([1.0, 0.0, 0.0]),
            angular_velocity=np.zeros(3),
        )
        rec.record(state, cmd, FlightPhase.ACTIVE, loop_time=0.001)
        assert rec.tick_count == 1
        npt.assert_array_equal(rec.positions[0], [1.0, 2.0, 3.0])
        npt.assert_array_equal(rec.cmd_linear_velocities[0], [1.0, 0.0, 0.0])
        assert rec.phases[0] == FlightPhase.ACTIVE.value
        assert rec.loop_times[0] == 0.001

    def test_record_body_rates(self):
        rec = FlightRecorder()
        cmd = ControlCommand(
            thrust=0.5,
            body_rates=np.array([0.1, 0.2, 0.3]),
        )
        rec.record(VehicleState(stamp=1.0), cmd, FlightPhase.ACTIVE)
        assert rec.cmd_thrusts[0] == 0.5
        npt.assert_array_equal(rec.cmd_body_rates[0], [0.1, 0.2, 0.3])

    def test_record_with_none_cmd(self):
        rec = FlightRecorder()
        rec.record(VehicleState(stamp=1.0), None, FlightPhase.INIT)
        assert rec.tick_count == 1
        npt.assert_array_equal(rec.cmd_linear_velocities[0], [0, 0, 0])

    def test_as_dict_has_correct_length(self):
        rec = FlightRecorder(max_ticks=100)
        for i in range(10):
            rec.record(
                VehicleState(stamp=float(i), position=np.array([float(i), 0, 0])),
                None,
                FlightPhase.ACTIVE,
            )
        d = rec.as_dict()
        assert d["tick_count"] == 10
        assert len(d["positions"]) == 10
        assert len(d["timestamps"]) == 10

    def test_max_ticks_respected(self):
        rec = FlightRecorder(max_ticks=5)
        for i in range(10):
            rec.record(VehicleState(stamp=float(i)), None, FlightPhase.ACTIVE)
        assert rec.tick_count == 5

    def test_summary(self):
        rec = FlightRecorder()
        for i in range(100):
            rec.record(
                VehicleState(stamp=float(i) * 0.02),
                None,
                FlightPhase.ACTIVE,
                loop_time=0.001,
            )
        s = rec.summary()
        assert s["tick_count"] == 100
        assert s["mean_loop_ms"] > 0
        assert s["mean_hz"] > 0

    def test_summary_empty(self):
        rec = FlightRecorder()
        s = rec.summary()
        assert s["tick_count"] == 0
