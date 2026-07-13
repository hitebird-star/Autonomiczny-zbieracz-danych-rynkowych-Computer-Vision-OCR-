from __future__ import annotations

import unittest
from types import SimpleNamespace

from scanner.analysis.coverage_map import CoverageMap
from scanner.analysis.movement_memory import MovementMemory
from scanner.pipeline import AutonomousMarketLoop


class FakeCoverageMap:
    envelope = (348, 501, 672, 794)

    def blocked_ahead(self, pos, target, *, lookahead=None) -> bool:
        return False

    def is_blocked(self, cell) -> bool:
        return False

    def cell_of(self, pos):
        return (0, 0)

    def record_block(self, pos, reason) -> None:
        pass


class EdgeBlockedCoverageMap(FakeCoverageMap):
    def blocked_ahead(self, pos, target, *, lookahead=None) -> bool:
        # Symuluje stan "twarzą do ściany": krok W idzie ku mniejszemu Y
        # i jest blokowany, ale cofnięcie S w +Y jest wolne.
        return target[1] < pos[1]


class RecordingMovement:
    def __init__(self) -> None:
        self.calls: list[tuple[str, float, float]] = []

    def execute(self, key, duration, settle) -> None:
        self.calls.append((key, duration, settle))


class RecordingDiagnostics:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def event(self, kind: str, **data) -> None:
        self.events.append((kind, data))


class PositionPipeline:
    def __init__(self, pos, vectors) -> None:
        self.current_position = pos
        self._odometry_vectors = vectors
        self.units_per_step = 3.2
        self.position_source = "dead_reckoning"

    def read_current_position(self, key):
        vec = self._odometry_vectors.get(key)
        if vec is not None and self.current_position is not None:
            self.current_position = (
                self.current_position[0] + vec[0],
                self.current_position[1] + vec[1],
            )
        return self.current_position


class MovementBoundaryGuardTests(unittest.TestCase):
    def _loop(self, *, pos, vec=(2.0, 0.0)) -> AutonomousMarketLoop:
        loop = object.__new__(AutonomousMarketLoop)
        loop.capture_pipeline = SimpleNamespace(
            current_position=pos,
            _odometry_vectors={"w": vec},
            units_per_step=3.2,
        )
        loop._covmap = FakeCoverageMap()
        loop._farm_boundary = None
        return loop

    def test_blocks_next_step_outside_coverage_map(self) -> None:
        loop = self._loop(pos=(500.0, 700.0), vec=(2.0, 0.0))

        blocked, reason, next_pos = loop._movement_blocked_by_coverage("w")

        self.assertTrue(blocked)
        self.assertEqual(reason, "next_outside_coverage_envelope")
        self.assertEqual(next_pos, (502.0, 700.0))

    def test_allows_step_inside_coverage_map(self) -> None:
        loop = self._loop(pos=(450.0, 700.0), vec=(2.0, 0.0))

        blocked, reason, next_pos = loop._movement_blocked_by_coverage("w")

        self.assertFalse(blocked)
        self.assertEqual(reason, "")
        self.assertEqual(next_pos, (452.0, 700.0))

    def test_learned_bad_move_blocks_before_step(self) -> None:
        loop = self._loop(pos=(450.0, 700.0), vec=(2.0, 0.0))
        memory = MovementMemory(min_attempts=4, avoid_failure_rate=0.75)
        cell = loop._covmap.cell_of((450.0, 700.0))
        for _ in range(4):
            memory.record_blocked(cell, "w")
        loop._movement_memory = memory

        blocked, reason, next_pos = loop._movement_blocked_by_coverage("w")

        self.assertTrue(blocked)
        self.assertEqual(reason, "learned_bad_move")
        self.assertEqual(next_pos, (452.0, 700.0))

    def test_outside_envelope_return_step_overrides_learned_bad_move(self) -> None:
        loop = self._loop(pos=(400.0, 650.0), vec=(0.0, 10.0))
        loop.capture_pipeline._odometry_vectors["s"] = (0.0, 10.0)
        memory = MovementMemory(min_attempts=4, avoid_failure_rate=0.75)
        cell = loop._covmap.cell_of((400.0, 650.0))
        for _ in range(4):
            memory.record_blocked(cell, "s")
        loop._movement_memory = memory

        blocked, reason, next_pos = loop._movement_blocked_by_coverage("s")

        self.assertFalse(blocked)
        self.assertEqual(reason, "")
        self.assertEqual(next_pos, (400.0, 660.0))

    def test_outside_envelope_step_farther_out_is_blocked(self) -> None:
        loop = self._loop(pos=(400.0, 650.0), vec=(0.0, -10.0))

        blocked, reason, next_pos = loop._movement_blocked_by_coverage("w")

        self.assertTrue(blocked)
        self.assertEqual(reason, "outside_coverage_envelope")
        self.assertEqual(next_pos, (400.0, 640.0))

    def test_repeated_learned_bad_drive_escapes_instead_of_stopping(self) -> None:
        coverage = CoverageMap((0, 100, 0, 80), cell_size=20.0)
        loop = object.__new__(AutonomousMarketLoop)
        loop._covmap = coverage
        loop._last_drive_block_reason = "learned_bad_move"
        loop._coverage_component_exhausted = False
        loop._coverage_stop_reason = None
        loop._last_learned_bad_move_cell = None
        loop._learned_bad_move_repeats = 0
        loop._g4_recovery_attempts = 0
        loop.diagnostics = RecordingDiagnostics()
        loop.capture_pipeline = PositionPipeline(
            (10.0, 10.0),
            {
                "w": (2.0, 0.0),
                "s": (-2.0, 0.0),
            },
        )
        loop.movement = RecordingMovement()
        loop._movement_memory = None

        pos = (10.0, 10.0)
        target = (50.0, 10.0)

        self.assertFalse(loop._handle_failed_drive(pos=pos, target=target))
        self.assertFalse(loop._handle_failed_drive(pos=pos, target=target))
        self.assertFalse(loop._handle_failed_drive(pos=pos, target=target))

        self.assertFalse(loop._coverage_component_exhausted)
        self.assertIsNone(loop._coverage_stop_reason)
        self.assertEqual([call[0] for call in loop.movement.calls], ["d", "w"])
        self.assertIn(
            "learned_bad_move_escape",
            [kind for kind, _ in loop.diagnostics.events],
        )

    def test_goto_escapes_with_backstep_when_target_is_behind_blocked_forward(self) -> None:
        loop = object.__new__(AutonomousMarketLoop)
        loop.capture_pipeline = PositionPipeline(
            (400.0, 677.0),
            {
                "w": (-2.59, -1.87),
                "s": (2.59, 1.87),
            },
        )
        loop._covmap = EdgeBlockedCoverageMap()
        loop._farm_boundary = None
        loop._goto_step_budget = 1
        loop._cell_size = 20
        loop._step_hold = 0.6
        loop._settle = 0.9
        loop.movement = RecordingMovement()
        loop.diagnostics = RecordingDiagnostics()

        reached = loop._drive_toward_target((418.0, 722.0))

        self.assertFalse(reached)
        self.assertEqual([call[0] for call in loop.movement.calls], ["s"])
        self.assertEqual(
            [kind for kind, _ in loop.diagnostics.events],
            ["goto_escape_backstep"],
        )
        self.assertGreater(loop.capture_pipeline.current_position[1], 677.0)

    def test_goto_turns_in_place_when_forward_and_backstep_are_boundary_blocked(self) -> None:
        loop = object.__new__(AutonomousMarketLoop)
        loop.capture_pipeline = PositionPipeline(
            (400.0, 720.0),
            {
                "w": (1.0, 0.0),
                "s": (-1.0, 0.0),
            },
        )
        loop._covmap = FakeCoverageMap()
        loop._farm_boundary = None
        loop._goto_step_budget = 1
        loop._cell_size = 20
        loop._step_hold = 0.6
        loop._settle = 0.9
        loop._turn_nudge_steps = 2
        loop._boundary_escape_turn_limit = 3
        loop._boundary_escape_turns = 0
        loop.movement = RecordingMovement()
        loop.diagnostics = RecordingDiagnostics()

        def blocked(step_key):
            return True, "farm_boundary", (399.0, 720.0)

        loop._movement_blocked_by_coverage = blocked

        reached = loop._drive_toward_target((390.0, 720.0))

        self.assertFalse(reached)
        self.assertEqual([call[0] for call in loop.movement.calls], ["d"])
        self.assertEqual(loop.movement.calls[0][1:], (0.18, 0.12))
        self.assertIn(
            "goto_escape_turn",
            [kind for kind, _ in loop.diagnostics.events],
        )

    def test_recovery_stops_before_blocked_backstep(self) -> None:
        loop = object.__new__(AutonomousMarketLoop)
        loop.capture_pipeline = PositionPipeline(
            (400.0, 720.0),
            {
                "w": (1.0, 0.0),
                "s": (-1.0, 0.0),
            },
        )
        loop._recovery_attempt = 0
        loop._stuck_count = 1
        loop._last_positions_for_stuck = [(400.0, 720.0)]
        loop._steps_since_fix = 4
        loop.movement = RecordingMovement()
        loop.diagnostics = RecordingDiagnostics()

        def blocked(step_key):
            if step_key == "s":
                return True, "coverage_blocked_ahead", (399.0, 720.0)
            return False, "", None

        loop._movement_blocked_by_coverage = blocked

        loop._movement_recovery()

        self.assertEqual(loop.movement.calls, [])
        self.assertIn(
            "recovery_step_blocked",
            [kind for kind, _ in loop.diagnostics.events],
        )
        self.assertEqual(loop._stuck_count, 0)
        self.assertEqual(loop._last_positions_for_stuck, [])
        self.assertEqual(loop._steps_since_fix, 0)

    def test_recovery_uses_single_backstep_not_three(self) -> None:
        loop = object.__new__(AutonomousMarketLoop)
        loop.capture_pipeline = PositionPipeline(
            (400.0, 720.0),
            {
                "w": (1.0, 0.0),
                "s": (-1.0, 0.0),
            },
        )
        loop._recovery_attempt = 0
        loop._stuck_count = 1
        loop._last_positions_for_stuck = [(400.0, 720.0)]
        loop._steps_since_fix = 4
        loop.movement = RecordingMovement()
        loop.diagnostics = RecordingDiagnostics()
        loop._movement_blocked_by_coverage = lambda step_key: (False, "", None)

        loop._movement_recovery()

        self.assertEqual([call[0] for call in loop.movement.calls].count("s"), 1)
        self.assertEqual([call[0] for call in loop.movement.calls][0], "s")

    def test_guard_drives_to_first_bfs_hop_not_distant_target(self) -> None:
        coverage = CoverageMap((0, 100, 0, 80), cell_size=20.0)
        for cell in coverage.all_cells():
            if cell != (2, 0):
                center = coverage.cell_center(cell)
                for _ in range(4):
                    coverage.record_scan(center, duplicate=True)
        coverage.mark_no_go((1, 0))
        coverage.mark_no_go((1, 1))

        loop = object.__new__(AutonomousMarketLoop)
        loop.capture_pipeline = SimpleNamespace(
            current_position=(10.0, 10.0),
            _odometry_vectors={"w": (0.0, 0.0)},
            position_source="ocr",
        )
        loop._covmap = coverage
        loop._farm_boundary = None
        loop._stuck_count = 0
        loop._steps_since_fix = 0
        loop._coverage_dup_floor = 4
        loop._last_positions_for_stuck = []
        loop.diagnostics = None
        loop.movement = RecordingMovement()
        driven_to = []
        loop._drive_toward_target = lambda target: driven_to.append(target) or True

        loop._guard_movement("w")

        self.assertEqual(driven_to, [coverage.cell_center((0, 1))])


if __name__ == "__main__":
    unittest.main()
