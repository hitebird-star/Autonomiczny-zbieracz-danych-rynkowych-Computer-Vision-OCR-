"""Konfiguracja Atlasu — wszystko ustawialne, osobny plik `atlas_config.json`.

Nie dotyka `scanner_config.json` (obecny system nietknięty). UI (etap 2) edytuje ten
plik przez panel ustawień. Ścieżki domyślne wskazują na istniejące wyjścia bota
(read-only), plus własny store Atlasu.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

ATLAS_CONFIG_PATH = "atlas_config.json"


@dataclass(slots=True)
class AtlasConfig:
    # --- źródła read-only (wyjścia obecnego bota) ---
    market_dir: str = "market_map/glevia_market"
    scans_dir: str = "scans"
    shops_registry: str = "market_map/glevia_market/shops.jsonl"
    prices_csv: str = "ceny.csv"
    boundary_file: str = "market_map/glevia_market/farm_map.json"

    # --- własny store Atlasu ---
    atlas_store: str = "market_map/glevia_market/atlas.json"
    calibration_file: str = "market_map/glevia_market/atlas_calibration.json"

    # --- fotogrametria / kalibracja ---
    anchor: tuple[float, float] = (960.0, 540.0)   # piksel renderu postaci
    calib_keys: tuple[str, ...] = ("w", "a", "d")   # ≥2 nierównoległe kierunki
    calib_steps_per_key: int = 8                     # kroków na kierunek
    calib_step_hold_s: float = 0.12                  # ile trzymać klawisz na krok
    max_calib_residual_px: float = 6.0               # próg akceptacji kalibracji

    # --- model mapy ---
    dedup_radius_units: float = 1.5                  # promień scalania anonimowych sklepów (jedn. gry)
    position_ema: float = 0.35                        # waga nowej próbki przy uśrednianiu pozycji

    # --- serwer web (MVP: stdlib http.server + SSE) ---
    server_host: str = "127.0.0.1"
    server_port: int = 8770
    sse_interval_s: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # tuple -> list dla czystego JSON
        d["anchor"] = list(self.anchor)
        d["calib_keys"] = list(self.calib_keys)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtlasConfig":
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in (data or {}).items() if k in known}
        if "anchor" in kwargs and kwargs["anchor"] is not None:
            a = kwargs["anchor"]
            kwargs["anchor"] = (float(a[0]), float(a[1]))
        if "calib_keys" in kwargs and kwargs["calib_keys"] is not None:
            kwargs["calib_keys"] = tuple(str(k) for k in kwargs["calib_keys"])
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path = ATLAS_CONFIG_PATH) -> "AtlasConfig":
        """Wczytaj config; brak/uszkodzony → wartości domyślne (Atlas ma zawsze ruszyć)."""
        src = Path(path)
        if not src.exists():
            return cls()
        try:
            raw = src.read_text(encoding="utf-8")
            if not raw.strip():
                return cls()
            return cls.from_dict(json.loads(raw))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            return cls()

    def save(self, path: str | Path = ATLAS_CONFIG_PATH) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, out)
