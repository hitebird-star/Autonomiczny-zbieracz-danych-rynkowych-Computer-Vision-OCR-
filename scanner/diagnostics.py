"""Czytelna diagnostyka autonomicznej pętli na żywym obrazie gry."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


class AutoDiagnostics:
    """Zapisuje scenę, maskę koloru, overlay kandydatów i zdarzenia JSONL."""

    def __init__(
        self,
        root: str | Path = "dbg/auto",
        *,
        max_image_rounds: int = 0,
        images_every: int = 1,
    ) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.directory = Path(root) / stamp
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events_path = self.directory / "events.jsonl"
        self.round = 0
        self.max_image_rounds = max(0, int(max_image_rounds))
        self.images_every = max(1, int(images_every))
        self._saved_image_rounds = 0

    def event(self, kind: str, **data: Any) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "event": kind,
            **data,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def record_detection(
        self,
        image: Image.Image,
        mask: Image.Image,
        candidates: list,
        tracks: list,
        selected,
        *,
        screen_offset: tuple[int, int],
        legacy_pick=None,
    ) -> None:
        self.round += 1
        stem = f"round_{self.round:03d}"
        save_images = (
            self.round % self.images_every == 0
            and (
                self.max_image_rounds == 0
                or self._saved_image_rounds < self.max_image_rounds
            )
        )
        files: dict[str, str] = {}
        if save_images:
            image.save(self.directory / f"{stem}_scene.png")
            mask.save(self.directory / f"{stem}_mask.png")
            files["scene"] = f"{stem}_scene.png"
            files["mask"] = f"{stem}_mask.png"
            self._saved_image_rounds += 1

        overlay = image.convert("RGB")
        draw = ImageDraw.Draw(overlay)
        track_by_position = {track.position: track for track in tracks}
        selected_id = selected.track_id if selected is not None else None
        legacy_pick_id = (
            legacy_pick.track_id if legacy_pick is not None else None
        )
        entries = []
        for index, candidate in enumerate(candidates, start=1):
            x, y = candidate.local_position
            track = track_by_position.get(candidate.screen_position)
            chosen = track is not None and track.track_id == selected_id
            color = (
                (255, 230, 0)
                if chosen
                else (255, 90, 40)
                if candidate.likely_false
                else (0, 255, 80)
            )
            radius = 13 if chosen else 9
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                outline=color,
                width=3,
            )
            label = f"{index}"
            if track is not None:
                label += f"/{track.track_id}"
                if track.visited:
                    label += "/V"
                if track.failed:
                    label += "/F"
            draw.text((x + 12, y - 14), label, fill=color)
            entries.append(
                {
                    "index": index,
                    "local": [x, y],
                    "screen": list(candidate.screen_position),
                    "area": candidate.area,
                    "distance": round(candidate.distance, 2),
                    "hybrid_score": (
                        round(candidate.hybrid_score, 4)
                        if candidate.hybrid_score is not None
                        else None
                    ),
                    "likely_false": candidate.likely_false,
                    "track_id": track.track_id if track else None,
                    "track_visited": track.visited if track else None,
                    "track_failed": track.failed if track else None,
                    "track_attempts": track.attempts if track else None,
                    "track_fingerprinted": bool(track.fingerprint) if track else None,
                    "selected": chosen,
                    "legacy_selected": (
                        track is not None
                        and track.track_id == legacy_pick_id
                    ),
                }
            )
        if save_images:
            overlay.save(self.directory / f"{stem}_overlay.png")
            files["overlay"] = f"{stem}_overlay.png"
        self.event(
            "detection",
            round=self.round,
            screen_offset=list(screen_offset),
            candidates=entries,
            selected=selected_id,
            legacy_pick=legacy_pick_id,
            ranking_changed=(
                selected_id is not None
                and legacy_pick_id is not None
                and selected_id != legacy_pick_id
            ),
            images_saved=save_images,
            files=files,
        )

    def record_interaction(self, event: dict[str, Any]) -> None:
        self.event("interaction", **event)

    def record_capture(self, track, outcome) -> None:
        self.event(
            "capture_outcome",
            track_id=track.track_id,
            position=list(track.position),
            scan_id=outcome.scan.scan_id,
            status=outcome.scan.status.value,
            duplicate=outcome.duplicate,
            seller=outcome.scan.seller,
            reason=outcome.scan.error.reason if outcome.scan.error else None,
        )

    def record_movement(self, *, index: int, total: int, step) -> None:
        self.event(
            "movement",
            index=index,
            total=total,
            key=step.key,
            duration=step.duration,
            settle=step.settle,
            step_kind=step.kind,
            lane=step.lane,
        )
