"""Śledzenie kandydatów i eliminacja ponownych wizyt."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from PIL import Image

from scanner.detection.shop_detector import ShopCandidate


@dataclass(slots=True)
class TrackedShop:
    track_id: str
    position: tuple[int, int]
    fingerprint: str | None = None
    attempts: int = 0
    visited: bool = False
    failed: bool = False
    last_seen_round: int = 0
    distance: float = float('inf')  # z ramki detektora (ShopCandidate.distance)


def visual_fingerprint(image: Image.Image, *, size: int = 16) -> str:
    """Stabilny, tani fingerprint wyglądu sklepu/sceny."""

    gray = np.asarray(image.convert("L").resize((size, size)), dtype=np.uint8)
    average = int(gray.mean())
    bits = np.packbits((gray >= average).reshape(-1))
    return hashlib.blake2b(bits.tobytes(), digest_size=12).hexdigest()


class ShopTracker:
    def __init__(
        self,
        *,
        match_radius: int = 90,
        stale_rounds: int = 3,
        failed_zone_radius: int = 35,
        max_click_dist: int = 300,
    ) -> None:
        # Kliknięcie sklepu powoduje podejście postaci i przesunięcie całej
        # sceny. Z pomiarów live ten sam stragan zmieniał pozycję o 30–82 px.
        # Poprzednie 28 px tworzyło za każdym razem nowy track i prowadziło do
        # wielokrotnego otwierania tego samego sprzedawcy.
        #
        # Faza 1 (RESCAN_SCATTER_PLAN): failed_zone 80→35 (nie blankietuj pół okolicy).
        # max_click_dist=300: klik tylko „wokół siebie" – dalekie cele = krok trasy.
        self.match_radius = match_radius
        self.stale_rounds = stale_rounds
        self.failed_zone_radius = failed_zone_radius
        self.max_click_dist = max_click_dist
        self.round = 0
        self._sequence = 0
        self._tracks: dict[str, TrackedShop] = {}
        self._visited_fingerprints: set[str] = set()
        self._failed_zones: list[tuple[int, int]] = []
        self._fp_attempts: dict[str, int] = {}
        self.max_open_attempts: int = 1  # WARSTWA A: ile razy wolno otworzyć ten sam fingerprint

    @property
    def tracks(self) -> tuple[TrackedShop, ...]:
        return tuple(self._tracks.values())

    def update(self, candidates: list[ShopCandidate]) -> list[TrackedShop]:
        self.round += 1
        # Kandydaci są posortowani przez detektor (odległość / hybryda), nie
        # przez ich faktyczne położenie. Dawne, zachłanne ``_nearest``
        # przypisywało pierwszy kandydat do najbliższego tracka, nawet jeśli
        # następny kandydat był jego idealnym odpowiednikiem. Po przesunięciu
        # kamery przenosiło to status ``visited`` na sąsiedni, jeszcze
        # nieskanowany sklep i pętla ruszała dalej z ``selected=None``.
        #
        # Najpierw rezerwujemy globalnie najkrótsze pary kandydat↔track. To
        # nadal jest lekki, deterministyczny matcher (bez nowych zależności),
        # ale nie pozwala dalekiej parze ukraść identity parze bliższej.
        unmatched = set(self._tracks)
        matched: dict[int, TrackedShop] = {}
        radius_squared = self.match_radius**2
        pairs: list[tuple[int, int, str]] = []
        for candidate_index, candidate in enumerate(candidates):
            cx, cy = candidate.screen_position
            for track_id, track in self._tracks.items():
                tx, ty = track.position
                distance_squared = (tx - cx) ** 2 + (ty - cy) ** 2
                if distance_squared <= radius_squared:
                    pairs.append((distance_squared, candidate_index, track_id))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        matched_candidates: set[int] = set()
        for _, candidate_index, track_id in pairs:
            if candidate_index in matched_candidates or track_id not in unmatched:
                continue
            matched[candidate_index] = self._tracks[track_id]
            matched_candidates.add(candidate_index)
            unmatched.remove(track_id)

        visible = []
        for candidate_index, candidate in enumerate(candidates):
            track = matched.get(candidate_index)
            if track is None:
                self._sequence += 1
                track = TrackedShop(
                    track_id=f"shop-{self._sequence:05d}",
                    position=candidate.screen_position,
                )
                self._tracks[track.track_id] = track
            else:
                track.position = candidate.screen_position
            track.distance = candidate.distance
            track.last_seen_round = self.round
            visible.append(track)

        for track_id in tuple(unmatched):
            if self.round - self._tracks[track_id].last_seen_round > self.stale_rounds:
                del self._tracks[track_id]
        return visible

    def next_unvisited(self, visible: list[TrackedShop]) -> TrackedShop | None:
        return self.peek_unvisited(visible)

    def peek_unvisited(self, visible: list[TrackedShop]) -> TrackedShop | None:
        """Spróbuj KAŻDY odrębny sklep raz, zanim wrócisz do któregokolwiek (SCAN_DRAIN_FIX).

        Przebieg 1: attempts==0 (świeże). Przebieg 2: reszta dozwolona.
        FIX A: failed-zone wyklucza TYLKO próbowane-ponownie (untried -> nie pytaj strefy).
        FIX B: dwuprzebiegowy — najpierw nigdy-nie-próbowane, dopiero potem retry.
        """
        for prefer_fresh in (True, False):
            for track in visible:
                if track.visited or track.failed:
                    continue
                if track.fingerprint is not None and track.fingerprint in self._visited_fingerprints:
                    continue
                if track.distance > self.max_click_dist:
                    continue                     # za daleko: pomiń, szukaj bliższego (nie kończ!)
                untried = track.attempts == 0
                if prefer_fresh and not untried:
                    continue                     # przebieg 1: tylko świeże
                if not untried and self._in_failed_zone(track.position):
                    continue                     # FIX A: failed-zone tylko dla RE-prób
                return track
        return None

    def attach_fingerprint(self, track: TrackedShop, fingerprint: str) -> bool:
        # WARSTWA A: licznik otwarć PER FINGERPRINT (odporny na jitter)
        track.fingerprint = fingerprint
        if fingerprint in self._visited_fingerprints:
            track.visited = True
            return True
        n = self._fp_attempts.get(fingerprint, 0) + 1
        self._fp_attempts[fingerprint] = n
        if n > self.max_open_attempts:
            self._visited_fingerprints.add(fingerprint)
            track.visited = True
            return True
        return False

    def mark_visited(self, track: TrackedShop) -> None:
        track.visited = True
        if track.fingerprint:
            self._visited_fingerprints.add(track.fingerprint)

    def mark_failed(self, track: TrackedShop, *, terminal: bool = False) -> None:
        track.attempts += 1
        track.failed = terminal
        if terminal:
            self._failed_zones.append(track.position)

    def has_untried(self, visible: list[TrackedShop]) -> bool:
        """SCAN_DRAIN_FIX C: czy są jeszcze nietknięte sklepy w pierścieniu."""
        return any(
            not t.visited and not t.failed and t.attempts == 0
            and (t.fingerprint is None or t.fingerprint not in self._visited_fingerprints)
            and t.distance <= self.max_click_dist
            for t in visible
        )

    def _in_failed_zone(self, position: tuple[int, int]) -> bool:
        radius_squared = self.failed_zone_radius**2
        return any(
            (position[0] - x) ** 2 + (position[1] - y) ** 2
            <= radius_squared
            for x, y in self._failed_zones
        )

    def _nearest(
        self, position: tuple[int, int], allowed: set[str]
    ) -> TrackedShop | None:
        matches = []
        radius_squared = self.match_radius**2
        for track_id in allowed:
            track = self._tracks[track_id]
            distance = (
                (track.position[0] - position[0]) ** 2
                + (track.position[1] - position[1]) ** 2
            )
            if distance <= radius_squared:
                matches.append((distance, track))
        return min(matches, key=lambda item: item[0])[1] if matches else None
