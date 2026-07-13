"""Trwały, audytowalny zapis skanów odporny na przerwanie procesu."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image

from scanner.models import ShopScan
from scanner.models.shop_scan import utc_now_iso


_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
_REPLACE_RETRIES = 6
_REPLACE_BACKOFF_S = 0.15


def _atomic_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Atomowo podmień plik z krótkim retry na chwilowy lock Windowsa."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if not isinstance(exc, PermissionError) and winerror not in (5, 32):
                raise
            time.sleep(_REPLACE_BACKOFF_S * (attempt + 1))
    os.replace(src, dst)


class ScanRepository:
    def __init__(self, root: str | Path = "scans") -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def scan_dir(self, scan_id: str) -> Path:
        if not _SAFE_ID.fullmatch(scan_id) or scan_id in {".", ".."}:
            raise ValueError("scan_id zawiera niedozwolone znaki")
        path = (self.root / scan_id).resolve()
        if path.parent != self.root:
            raise ValueError("scan_id wychodzi poza katalog repozytorium")
        return path

    def create(self, scan: ShopScan) -> Path:
        directory = self.scan_dir(scan.scan_id)
        (directory / "tooltips").mkdir(parents=True, exist_ok=True)
        self.save_manifest(scan)
        self.append_event(scan.scan_id, "scan_created", status=scan.status.value)
        return directory

    def save_manifest(self, scan: ShopScan) -> Path:
        directory = self.scan_dir(scan.scan_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "manifest.json"
        temporary = directory / ".manifest.json.tmp"
        payload = json.dumps(
            scan.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        )
        with self._lock:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(payload)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            _atomic_replace(temporary, target)
            self._fsync_directory(directory)
        return target

    def load(self, scan_id: str) -> ShopScan:
        path = self.scan_dir(scan_id) / "manifest.json"
        return ShopScan.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def append_event(
        self, scan_id: str, event: str, **fields: Any
    ) -> Path:
        if not event.strip():
            raise ValueError("event nie może być pusty")
        directory = self.scan_dir(scan_id)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "raw_events.jsonl"
        record = {
            "timestamp": utc_now_iso(),
            "event": event,
            "scan_id": scan_id,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        return path

    def save_shop_image(self, scan_id: str, image: Image.Image) -> str:
        return self._save_image(scan_id, "shop.png", image)

    def save_tooltip_image(
        self, scan_id: str, slot: int, frame: int, image: Image.Image
    ) -> str:
        if not 0 <= slot <= 99:
            raise ValueError("slot musi mieścić się w zakresie 0..99")
        if frame < 1:
            raise ValueError("frame musi być dodatni")
        relative = f"tooltips/slot_{slot:03d}_{frame}.png"
        return self._save_image(scan_id, relative, image)

    def save_raw_frame(self, scan_id: str, name: str, image: Image.Image) -> str:
        return self._save_image(scan_id, f"frames/{name}.png", image)

    def _save_image(self, scan_id: str, relative: str, image: Image.Image) -> str:
        directory = self.scan_dir(scan_id)
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp")
        with self._lock:
            image.save(temporary, format="PNG")
            with temporary.open("r+b") as handle:
                os.fsync(handle.fileno())
            _atomic_replace(temporary, target)
            self._fsync_directory(target.parent)
        return relative.replace("\\", "/")

    def iter_scans(self) -> list[ShopScan]:
        scans = []
        for manifest in sorted(self.root.glob("*/manifest.json")):
            try:
                scans.append(
                    ShopScan.from_dict(
                        json.loads(manifest.read_text(encoding="utf-8"))
                    )
                )
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
        return scans

    def pending(self) -> list[ShopScan]:
        terminal = {"verified", "review", "failed"}
        return [scan for scan in self.iter_scans() if scan.status.value not in terminal]

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        # Windows nie pozwala standardowo fsyncować katalogu. Sam plik został
        # już zsynchronizowany; na POSIX domykamy również wpis katalogowy.
        if os.name == "nt":
            return
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
