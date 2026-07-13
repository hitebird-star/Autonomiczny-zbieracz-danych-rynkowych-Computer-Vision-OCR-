"""Ingest read-only: wyjścia obecnego bota → dane do Atlasu (nic nie zapisuje do nich).

Czyta:
  - `shops.jsonl`  → rejestr sklepów (fingerprint/seller/scan_ids/x,y),
  - `farm_map.json`→ polygon granicy rynku (jednostki gry),
  - `ceny.csv`     → oferty per sprzedawca (do „najtańszych ofert" i naniesienia na sklep).

Obecny system nietknięty — tylko odczyt.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scanner.atlas.contracts import Point2 as Vec


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    """Wczytaj `shops.jsonl` jako listę rekordów (pomija uszkodzone linie)."""
    src = Path(path)
    records: list[dict[str, Any]] = []
    if not src.exists():
        return records
    with src.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_boundary(path: str | Path) -> list[Vec]:
    """Wczytaj polygon granicy z `farm_map.json` (klucz `polygon`)."""
    src = Path(path)
    if not src.exists():
        return []
    try:
        data = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    poly = data.get("polygon") if isinstance(data, dict) else None
    if not poly:
        return []
    out: list[Vec] = []
    for p in poly:
        try:
            out.append((float(p[0]), float(p[1])))
        except (TypeError, ValueError, IndexError):
            continue
    return out


def load_offers_by_seller(path: str | Path) -> dict[str, list[dict[str, Any]]]:
    """Zgrupuj oferty z `ceny.csv` po sprzedawcy (do naniesienia na sklep w UI).

    CSV jest append-only i historyczne — zwracamy surowo pogrupowane; filtrowanie po
    dacie zostawiamy warstwie wyżej (patrz uwaga o append-only w pamięci projektu).
    """
    src = Path(path)
    by_seller: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not src.exists():
        return by_seller
    with src.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seller = (row.get("seller") or "").strip()
            if not seller:
                continue
            by_seller[seller].append(row)
    return dict(by_seller)
