"""Warstwa danych pod UI (czysta, testowalna): MarketAtlas → payload JSON dla frontendu.

Oddzielona od HTTP, żeby dało się ją testować bez serwera i użyć też do eksportu
statycznego (SVG/HTML). Sklep renderujemy na `atlas_position` (metryczna) jeśli jest,
inaczej fallback na `game_position` (zgrubna, coord postaci) — oznaczone `source`, żeby
UI mogło pokazać, co jest już skalibrowane, a co dopiero czeka na rzut.
"""

from __future__ import annotations

from typing import Any

from scanner.atlas.atlas import MarketAtlas


def build_render_payload(atlas: MarketAtlas) -> dict[str, Any]:
    shops: list[dict[str, Any]] = []
    for s in atlas.shops():
        pos = s.atlas_position or s.game_position
        if pos is None:
            continue
        shops.append(
            {
                "id": s.shop_id,
                "fingerprint": s.fingerprint,
                "seller": s.seller,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "source": "atlas" if s.atlas_position else "game",
                "confidence": round(float(s.position_confidence), 3),
                "transform_version": s.transform_version,
            }
        )

    xs = [sh["x"] for sh in shops] + [p[0] for p in atlas.boundary]
    ys = [sh["y"] for sh in shops] + [p[1] for p in atlas.boundary]
    bounds = None
    if xs and ys:
        bounds = {
            "minx": min(xs),
            "miny": min(ys),
            "maxx": max(xs),
            "maxy": max(ys),
        }

    located = sum(1 for sh in shops if sh["source"] == "atlas")
    sellers = sorted({s.seller for s in atlas.shops() if s.seller})
    return {
        "boundary": [[float(p[0]), float(p[1])] for p in atlas.boundary],
        "bounds": bounds,
        "shops": shops,
        "stats": {
            "total": len(atlas.shops()),
            "rendered": len(shops),
            "located": located,          # metryczne (po kalibracji)
            "on_stand": len(shops) - located,  # jeszcze na coord postoju
            "sellers": len(sellers),
        },
    }
