"""Samoucząca pamięć ruchu po komórkach mapy.

CoverageMap mówi *gdzie* chcemy dojść. MovementMemory uczy się, które wyjścia
z danej komórki są w praktyce złe: stoją w miejscu, wpadają w guard, prowadzą
w teksturę albo często kończą się blokadą. To jest czysta, offline-testowalna
warstwa danych; live-loop tylko zapisuje obserwacje i pyta, czy dany ruch ma już
wystarczająco zły bilans.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

Cell = tuple[int, int]


@dataclass(slots=True)
class MoveStats:
    attempts: int = 0
    successes: int = 0
    blocked: int = 0
    stuck: int = 0

    @property
    def failures(self) -> int:
        return self.blocked + self.stuck

    @property
    def failure_rate(self) -> float:
        if self.attempts <= 0:
            return 0.0
        return self.failures / self.attempts

    def to_dict(self) -> dict[str, int]:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "blocked": self.blocked,
            "stuck": self.stuck,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MoveStats":
        return cls(
            attempts=int(data.get("attempts", 0) or 0),
            successes=int(data.get("successes", 0) or 0),
            blocked=int(data.get("blocked", 0) or 0),
            stuck=int(data.get("stuck", 0) or 0),
        )


class MovementMemory:
    """Trwała pamięć skuteczności ruchu `(cell, key)`.

    `should_avoid` jest konserwatywne: wymaga kilku prób i wysokiego odsetka
    porażek. Pojedynczy błąd OCR nie zamyka przejścia.
    """

    def __init__(
        self,
        *,
        min_attempts: int = 4,
        avoid_failure_rate: float = 0.75,
        probe_every: int = 5,
    ) -> None:
        if min_attempts < 1:
            raise ValueError("min_attempts musi być >= 1")
        if not 0.0 <= avoid_failure_rate <= 1.0:
            raise ValueError("avoid_failure_rate musi być w zakresie 0..1")
        if probe_every < 0:
            raise ValueError("probe_every musi byc >= 0")
        self.min_attempts = min_attempts
        self.avoid_failure_rate = avoid_failure_rate
        self.probe_every = probe_every
        self._stats: dict[tuple[Cell, str], MoveStats] = {}
        self._avoid_checks: dict[tuple[Cell, str], int] = {}

    def stats(self, cell: Cell, key: str) -> MoveStats:
        return self._stats.get((tuple(cell), key), MoveStats())

    def record_success(self, cell: Cell, key: str) -> None:
        stat = self._stats.setdefault((tuple(cell), key), MoveStats())
        stat.attempts += 1
        stat.successes += 1
        self._avoid_checks.pop((tuple(cell), key), None)

    def record_blocked(self, cell: Cell, key: str) -> None:
        stat = self._stats.setdefault((tuple(cell), key), MoveStats())
        stat.attempts += 1
        stat.blocked += 1

    def record_stuck(self, cell: Cell, key: str) -> None:
        stat = self._stats.setdefault((tuple(cell), key), MoveStats())
        stat.attempts += 1
        stat.stuck += 1

    def should_avoid(self, cell: Cell, key: str) -> bool:
        stat = self.stats(cell, key)
        avoid = (
            stat.attempts >= self.min_attempts
            and stat.failure_rate >= self.avoid_failure_rate
        )
        if not avoid:
            self._avoid_checks.pop((tuple(cell), key), None)
            return False
        if self.probe_every <= 0:
            return True
        guard_key = (tuple(cell), key)
        checks = self._avoid_checks.get(guard_key, 0) + 1
        self._avoid_checks[guard_key] = checks
        return checks % self.probe_every != 0

    def to_dict(self) -> dict[str, Any]:
        entries = []
        for (cell, key), stat in sorted(self._stats.items()):
            entries.append(
                {
                    "cell": [cell[0], cell[1]],
                    "key": key,
                    **stat.to_dict(),
                }
            )
        return {
            "version": 1,
            "min_attempts": self.min_attempts,
            "avoid_failure_rate": self.avoid_failure_rate,
            "probe_every": self.probe_every,
            "entries": entries,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MovementMemory":
        memory = cls(
            min_attempts=int(data.get("min_attempts", 4) or 4),
            avoid_failure_rate=float(data.get("avoid_failure_rate", 0.75) or 0.75),
            probe_every=int(data.get("probe_every", 5) or 0),
        )
        for entry in data.get("entries", []) or []:
            cell_data = entry.get("cell")
            key = entry.get("key")
            if (
                not isinstance(cell_data, list)
                or len(cell_data) != 2
                or not isinstance(key, str)
            ):
                continue
            cell = (int(cell_data[0]), int(cell_data[1]))
            memory._stats[(cell, key)] = MoveStats.from_dict(entry)
        return memory

    def save(self, path: str | Path) -> None:
        """Zapis atomowy: najpierw plik tymczasowy, potem `os.replace`.

        `os.replace` jest atomowe na tym samym systemie plików (Windows NTFS
        też).  Ctrl+C w trakcie `write_text` zostawi tylko `.tmp` — właściwy
        plik nigdy nie będzie pusty ani częściowy.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, out)

    @classmethod
    def load(cls, path: str | Path) -> "MovementMemory | None":
        """Tolerancyjny odczyt: pusty/uszkodzony plik → None (fresh start).

        Nie crashuje na pliku 0-bajtowym (np. po Ctrl+C na starym,
        nieatomowym zapisie) ani na uszkodzonym JSON-ie.  Uszkodzony plik
        jest kopiowany do `.corrupted` dla diagnostyki.
        """
        src = Path(path)
        if not src.exists():
            return None
        try:
            raw = src.read_text(encoding="utf-8")
        except OSError:
            _logger.warning("MovementMemory: nie można odczytać %s – start od zera", src)
            return None
        if not raw.strip():
            _logger.warning("MovementMemory: pusty plik %s – start od zera", src)
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            _logger.warning("MovementMemory: uszkodzony JSON w %s – start od zera", src)
            # Zachowaj uszkodzony plik dla diagnostyki
            corrupted = src.with_suffix(src.suffix + ".corrupted")
            try:
                corrupted.write_text(raw, encoding="utf-8")
            except OSError:
                pass
            return None
        return cls.from_dict(data)
