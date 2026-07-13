"""Czysty kontrakt między live-game Codexa a matematyką/UI Atlasa.

Te struktury są celowo małe i serializowalne. Moduły live produkują obserwacje,
a rdzeń offline może je konsumować bez importów z capture/runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


Point2 = tuple[float, float]
WindowRectTuple = tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class ShopScreenObservation:
    """Pojedynczy kandydat sklepu wykryty w klatce klienta.

    `local_position` jest w układzie obrazu klienta Glevia2. `screen_position`
    jest absolutne na pulpicie i służy tylko do debug/live-kliku.
    """

    local_position: Point2
    screen_position: Point2
    area: int
    distance: float
    hybrid_score: float | None = None
    likely_false: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FrameSnapshot:
    """Jedna klatka dla Atlasa: pozycja gracza + widoczne sklepy."""

    timestamp: str
    window_rect: WindowRectTuple
    player_game: Point2 | None
    shops_screen: tuple[Point2, ...]
    shop_observations: tuple[ShopScreenObservation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MoveObservation:
    """Efekt jednego kontrolowanego kroku kalibracyjnego."""

    key: str
    started_at: str
    duration_s: float
    player_before: Point2 | None
    player_after: Point2 | None
    delta_game: Point2
    delta_screen: tuple[Point2, ...]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
