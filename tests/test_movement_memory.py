from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scanner.analysis.movement_memory import MovementMemory


class MovementMemoryTests(unittest.TestCase):
    def test_avoids_only_after_repeated_high_failure_rate(self) -> None:
        memory = MovementMemory(min_attempts=4, avoid_failure_rate=0.75)
        cell = (2, 3)

        for _ in range(3):
            memory.record_blocked(cell, "w")

        self.assertFalse(memory.should_avoid(cell, "w"))

        memory.record_blocked(cell, "w")

        self.assertTrue(memory.should_avoid(cell, "w"))

    def test_successes_keep_edge_available(self) -> None:
        memory = MovementMemory(min_attempts=4, avoid_failure_rate=0.75)
        cell = (2, 3)

        memory.record_blocked(cell, "w")
        memory.record_blocked(cell, "w")
        memory.record_success(cell, "w")
        memory.record_success(cell, "w")

        self.assertFalse(memory.should_avoid(cell, "w"))

    def test_learned_bad_move_allows_periodic_probe(self) -> None:
        memory = MovementMemory(
            min_attempts=4,
            avoid_failure_rate=0.75,
            probe_every=3,
        )
        cell = (2, 3)

        for _ in range(4):
            memory.record_blocked(cell, "w")

        self.assertTrue(memory.should_avoid(cell, "w"))
        self.assertTrue(memory.should_avoid(cell, "w"))
        self.assertFalse(memory.should_avoid(cell, "w"))  # kontrolny probe
        self.assertTrue(memory.should_avoid(cell, "w"))

    def test_success_resets_probe_counter(self) -> None:
        memory = MovementMemory(
            min_attempts=4,
            avoid_failure_rate=0.75,
            probe_every=3,
        )
        cell = (2, 3)

        for _ in range(4):
            memory.record_blocked(cell, "w")

        self.assertTrue(memory.should_avoid(cell, "w"))
        self.assertTrue(memory.should_avoid(cell, "w"))
        memory.record_success(cell, "w")

        self.assertTrue(memory.should_avoid(cell, "w"))
        self.assertTrue(memory.should_avoid(cell, "w"))
        self.assertFalse(memory.should_avoid(cell, "w"))

    def test_roundtrip(self) -> None:
        memory = MovementMemory(min_attempts=5, avoid_failure_rate=0.8, probe_every=7)
        memory.record_blocked((1, 2), "s")
        memory.record_stuck((1, 2), "s")
        memory.record_success((2, 2), "d")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movement_memory.json"
            memory.save(path)

            loaded = MovementMemory.load(path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.min_attempts, 5)
        self.assertEqual(loaded.probe_every, 7)
        self.assertEqual(loaded.stats((1, 2), "s").blocked, 1)
        self.assertEqual(loaded.stats((1, 2), "s").stuck, 1)
        self.assertEqual(loaded.stats((2, 2), "d").successes, 1)

    # ---- tolerancyjny odczyt (pusty/uszkodzony plik) ----

    def test_load_empty_file_returns_none(self) -> None:
        """Pusty plik (0 bajtów) po nieatomowym Ctrl+C -> fresh start."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movement_memory.json"
            path.write_text("", encoding="utf-8")
            loaded = MovementMemory.load(path)
            self.assertIsNone(loaded)

    def test_load_whitespace_only_returns_none(self) -> None:
        """Plik zawierajacy tylko biale znaki -> fresh start."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movement_memory.json"
            path.write_text("   \n\t  \n", encoding="utf-8")
            loaded = MovementMemory.load(path)
            self.assertIsNone(loaded)

    def test_load_corrupted_json_returns_none(self) -> None:
        """Uszkodzony JSON -> fresh start."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movement_memory.json"
            path.write_text('{"version": 1, "entries": [', encoding="utf-8")
            loaded = MovementMemory.load(path)
            self.assertIsNone(loaded)

    def test_load_corrupted_json_saves_copy(self) -> None:
        """Uszkodzony JSON -> plik .corrupted do diagnostyki."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movement_memory.json"
            corrupted = Path(temp) / "movement_memory.json.corrupted"
            broken = '{"version": 1, "broken": true...'
            path.write_text(broken, encoding="utf-8")
            loaded = MovementMemory.load(path)
            self.assertIsNone(loaded)
            self.assertTrue(corrupted.exists(), ".corrupted missing")

    def test_load_nonexistent_file_returns_none(self) -> None:
        """Brak pliku -> None (bez zmian)."""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "nonexistent.json"
            loaded = MovementMemory.load(path)
            self.assertIsNone(loaded)

    # ---- atomowy zapis ----

    def test_save_atomic_no_tmp_leftover(self) -> None:
        """Po zapisie plik .tmp NIE istnieje, docelowy istnieje."""
        memory = MovementMemory()
        memory.record_success((0, 0), "w")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movement_memory.json"
            tmp = Path(temp) / "movement_memory.json.tmp"
            memory.save(path)
            self.assertTrue(path.exists(), "target file missing")
            self.assertFalse(tmp.exists(), ".tmp leftover after os.replace")

    def test_save_then_load_via_atomic_roundtrip(self) -> None:
        """Round-trip: atomowy save -> tolerancyjny load."""
        memory = MovementMemory(min_attempts=3, avoid_failure_rate=0.5)
        memory.record_blocked((5, 5), "a")
        memory.record_stuck((5, 5), "a")
        memory.record_blocked((5, 5), "a")
        memory.record_blocked((5, 5), "a")
        memory.record_success((6, 6), "d")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "movement_memory.json"
            memory.save(path)
            loaded = MovementMemory.load(path)

        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded.min_attempts, 3)
        self.assertEqual(loaded.avoid_failure_rate, 0.5)
        self.assertTrue(loaded.should_avoid((5, 5), "a"))
        self.assertFalse(loaded.should_avoid((6, 6), "d"))


if __name__ == "__main__":
    unittest.main()
