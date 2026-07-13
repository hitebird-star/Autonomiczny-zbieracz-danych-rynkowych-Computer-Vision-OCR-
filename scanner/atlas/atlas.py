"""Model mapy Atlasu: sklepy z metryczną pozycją, granica, własny store.

Schemat sklepu wg kontraktu Codexa (pod live-mapę, powrót do sklepu, najtańsze oferty,
UI/HTML, bazę itemów):

    shop_id, fingerprint, seller, scan_ids,
    game_position   — zgrubna (bot, coord POSTACI; wiele sklepów dzieli jeden punkt),
    atlas_position  — METRYCZNA (rzut fotogrametryczny, osobna per sklep),
    screen_position — ostatni piksel,
    position_confidence, transform_version.

`MarketAtlas` jest offline-czysty: konsumuje dane rejestru (read-only, przez `ingest`)
oraz — gdy dostępny `GroundProjection` + piksel sklepu — liczy `atlas_position`. Dedup:
fingerprint (klucz mocny) + promień lokalizacji (dla anonimowych trafień live).
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from scanner.atlas.calibration import GroundProjection
from scanner.atlas.contracts import Point2 as Vec


@dataclass(slots=True)
class AtlasShop:
    shop_id: str
    fingerprint: str
    seller: str = ""
    scan_ids: list[str] = field(default_factory=list)
    game_position: Vec | None = None
    atlas_position: Vec | None = None
    screen_position: Vec | None = None
    position_confidence: float = 0.0
    transform_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        def pt(v: Vec | None) -> list[float] | None:
            return None if v is None else [float(v[0]), float(v[1])]

        return {
            "shop_id": self.shop_id,
            "fingerprint": self.fingerprint,
            "seller": self.seller,
            "scan_ids": list(self.scan_ids),
            "game_position": pt(self.game_position),
            "atlas_position": pt(self.atlas_position),
            "screen_position": pt(self.screen_position),
            "position_confidence": float(self.position_confidence),
            "transform_version": self.transform_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AtlasShop":
        def pt(v: Any) -> Vec | None:
            if not v:
                return None
            return (float(v[0]), float(v[1]))

        return cls(
            shop_id=str(d.get("shop_id", "")),
            fingerprint=str(d.get("fingerprint", "")),
            seller=str(d.get("seller", "") or ""),
            scan_ids=list(d.get("scan_ids", []) or []),
            game_position=pt(d.get("game_position")),
            atlas_position=pt(d.get("atlas_position")),
            screen_position=pt(d.get("screen_position")),
            position_confidence=float(d.get("position_confidence", 0.0) or 0.0),
            transform_version=str(d.get("transform_version", "") or ""),
        )


def _dist(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MarketAtlas:
    """Trwała, metryczna mapa rynku z własnym układem współrzędnych."""

    def __init__(
        self,
        *,
        dedup_radius_units: float = 1.5,
        position_ema: float = 0.35,
        boundary: list[Vec] | None = None,
    ) -> None:
        self.dedup_radius_units = float(dedup_radius_units)
        self.position_ema = float(position_ema)
        self.boundary: list[Vec] = list(boundary or [])
        self._by_fp: dict[str, AtlasShop] = {}

    # ---- ingest rejestru (read-only dane bota) -------------------------------
    def upsert_from_registry(self, records: Iterable[dict[str, Any]]) -> int:
        """Nanieś statyczne dane sklepu (fingerprint/seller/scan_ids/game_position).

        Nie liczy jeszcze `atlas_position` — to robi `place_shop` gdy mamy piksel + rzut.
        Zwraca liczbę nowych sklepów.
        """
        added = 0
        for r in records:
            fp = str(r.get("fingerprint") or "").strip()
            if not fp:
                continue
            gx, gy = r.get("x"), r.get("y")
            gpos = (float(gx), float(gy)) if gx is not None and gy is not None else None
            existing = self._by_fp.get(fp)
            if existing is None:
                self._by_fp[fp] = AtlasShop(
                    shop_id=str(r.get("shop_id", fp)),
                    fingerprint=fp,
                    seller=str(r.get("seller", "") or ""),
                    scan_ids=list(r.get("scan_ids", []) or []),
                    game_position=gpos,
                )
                added += 1
            else:
                if r.get("seller"):
                    existing.seller = str(r["seller"])
                for sid in r.get("scan_ids", []) or []:
                    if sid not in existing.scan_ids:
                        existing.scan_ids.append(sid)
                if gpos is not None:
                    existing.game_position = gpos
        return added

    # ---- rzut metryczny per sklep --------------------------------------------
    def place_shop(
        self,
        fingerprint: str,
        player_game: Vec,
        screen_px: Vec,
        projection: GroundProjection,
        *,
        confidence: float = 1.0,
    ) -> AtlasShop:
        """Policz `atlas_position` z piksela sklepu + pozycji postaci + rzutu.

        Uśrednia z poprzednią pozycją (EMA ważone confidence) — kolejne obserwacje
        tego samego sklepu stabilizują metryczną lokalizację.
        """
        pos = projection.screen_to_game(player_game, screen_px)
        shop = self._by_fp.get(fingerprint)
        if shop is None:
            shop = AtlasShop(shop_id=fingerprint, fingerprint=fingerprint)
            self._by_fp[fingerprint] = shop
        if shop.atlas_position is None:
            shop.atlas_position = pos
            shop.position_confidence = confidence
        else:
            a = self.position_ema * confidence
            shop.atlas_position = (
                (1 - a) * shop.atlas_position[0] + a * pos[0],
                (1 - a) * shop.atlas_position[1] + a * pos[1],
            )
            shop.position_confidence = min(1.0, shop.position_confidence + a * (1 - shop.position_confidence))
        shop.screen_position = (float(screen_px[0]), float(screen_px[1]))
        shop.transform_version = projection.version
        return shop

    # ---- zapytania -----------------------------------------------------------
    def shops(self) -> list[AtlasShop]:
        return list(self._by_fp.values())

    def located_shops(self) -> list[AtlasShop]:
        """Sklepy z policzoną metryczną pozycją (gotowe na mapę/nawigację)."""
        return [s for s in self._by_fp.values() if s.atlas_position is not None]

    def nearest(self, game_pos: Vec, k: int = 1) -> list[AtlasShop]:
        located = [s for s in self.located_shops()]
        located.sort(key=lambda s: _dist(game_pos, s.atlas_position))  # type: ignore[arg-type]
        return located[:k]

    def by_seller(self, seller: str) -> list[AtlasShop]:
        s = seller.strip().lower()
        return [x for x in self._by_fp.values() if x.seller.strip().lower() == s]

    # ---- serializacja --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dedup_radius_units": self.dedup_radius_units,
            "position_ema": self.position_ema,
            "boundary": [[p[0], p[1]] for p in self.boundary],
            "shops": [s.to_dict() for s in sorted(self._by_fp.values(), key=lambda x: x.shop_id)],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MarketAtlas":
        atlas = cls(
            dedup_radius_units=float(d.get("dedup_radius_units", 1.5)),
            position_ema=float(d.get("position_ema", 0.35)),
            boundary=[(float(p[0]), float(p[1])) for p in d.get("boundary", []) or []],
        )
        for sd in d.get("shops", []) or []:
            shop = AtlasShop.from_dict(sd)
            if shop.fingerprint:
                atlas._by_fp[shop.fingerprint] = shop
        return atlas

    def save(self, path: str | Path) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, out)

    @classmethod
    def load(cls, path: str | Path) -> "MarketAtlas | None":
        src = Path(path)
        if not src.exists():
            return None
        try:
            raw = src.read_text(encoding="utf-8")
            if not raw.strip():
                return None
            return cls.from_dict(json.loads(raw))
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            return None
