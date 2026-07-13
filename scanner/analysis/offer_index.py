"""Indeks ofert: item -> najtanszy sklep (Etap 6, Faza 2).

Czyste dane, zero gry. Osobny plik `offers.jsonl` obok `shops.jsonl` w partycji.
Kazda linia to jedna oferta:
  {"shop_id": 42, "item": "Kamien Duszy", "unit_price": 120000, "quantity": 5,
   "scanned_at": "2026-06-22T14:30:00", "scan_id": "20260622_143000_Gracz1"}

OfferIndex trzyma w pamieci dict item -> [(shop_id, unit_price, scanned_at), ...].
Query cheapest(item) zwraca shop_id najtanszego (z TTL).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


OFFERS_FILENAME = "offers.jsonl"
DEFAULT_OFFER_TTL_HOURS = 24  # oferta starsza niz 24h = ignorowana


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(slots=True)
class OfferEntry:
    """Pojedyncza oferta w indeksie (w pamieci)."""
    shop_id: int
    item: str
    unit_price: int
    quantity: int = 1
    scanned_at: str = field(default_factory=_now_iso)
    scan_id: str | None = None

    @property
    def scanned_dt(self) -> datetime | None:
        return _parse_ts(self.scanned_at)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OfferEntry":
        return cls(
            shop_id=int(data["shop_id"]),
            item=str(data["item"]),
            unit_price=int(data["unit_price"]),
            quantity=int(data.get("quantity", 1)),
            scanned_at=str(data.get("scanned_at", _now_iso())),
            scan_id=data.get("scan_id"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "shop_id": self.shop_id,
            "item": self.item,
            "unit_price": self.unit_price,
            "quantity": self.quantity,
            "scanned_at": self.scanned_at,
            "scan_id": self.scan_id,
        }


class OfferIndex:
    """Indeks item -> lista ofert. W pamieci; save/load z offers.jsonl."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._by_item: dict[str, list[OfferEntry]] = {}

    # --- IO ----------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self.directory / OFFERS_FILENAME

    def load(self) -> "OfferIndex":
        self._by_item.clear()
        if not self.path.exists():
            return self
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                entry = OfferEntry.from_dict(json.loads(line))
                self._add_to_index(entry)
        return self

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            # zapisz posortowane: item, potem shop_id
            all_entries = sorted(
                (e for entries in self._by_item.values() for e in entries),
                key=lambda e: (e.item, e.shop_id, e.scanned_at),
            )
            for entry in all_entries:
                handle.write(
                    json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
                )
        tmp.replace(self.path)
        return self.path

    # --- mutacja -----------------------------------------------------------

    def add(self, entry: OfferEntry) -> None:
        """Dodaj oferte do indeksu i do pliku (append)."""
        self._add_to_index(entry)
        self.directory.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(entry.to_dict(), ensure_ascii=False) + "\n"
            )

    def add_from_manifest(
        self, shop_id: int, shop_scan: Any, scan_id: str | None = None
    ) -> int:
        """Dodaj oferty z manifestu/ShopScan do indeksu.
        
        shop_scan musi miec atrybut .offers (lista ItemObservation z 
        unit_price, quantity) lub byc dictem z kluczem 'offers'.
        Zwraca liczbe dodanych ofert.
        """
        added = 0
        offers = getattr(shop_scan, "offers", None) or []
        if isinstance(shop_scan, dict):
            offers = shop_scan.get("offers", [])
        for obs in offers:
            if isinstance(obs, dict):
                item = obs.get("item_name") or obs.get("item", "")
                unit = obs.get("unit_price", 0)
                qty = obs.get("quantity", 1)
            else:
                item = getattr(obs, "item_name", "") or getattr(obs, "item", "")
                unit = getattr(obs, "unit_price", 0)
                qty = getattr(obs, "quantity", 1)
            if not item or not unit:
                continue
            self.add(OfferEntry(
                shop_id=shop_id,
                item=item,
                unit_price=int(unit),
                quantity=int(qty),
                scanned_at=_now_iso(),
                scan_id=scan_id,
            ))
            added += 1
        return added

    def _add_to_index(self, entry: OfferEntry) -> None:
        key = entry.item
        if key not in self._by_item:
            self._by_item[key] = []
        self._by_item[key].append(entry)

    # --- query -------------------------------------------------------------

    def cheapest(
        self,
        item: str,
        *,
        ttl_hours: float = DEFAULT_OFFER_TTL_HOURS,
        now: datetime | None = None,
    ) -> OfferEntry | None:
        """Najtansza oferta danego itemu (z TTL). Zwraca None gdy brak."""
        entries = self._by_item.get(item, [])
        if not entries:
            return None
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=ttl_hours)
        fresh = [
            e for e in entries
            if e.scanned_dt is not None and e.scanned_dt >= cutoff
        ]
        if not fresh:
            return None
        return min(fresh, key=lambda e: e.unit_price)

    def all_for(
        self,
        item: str,
        *,
        ttl_hours: float = DEFAULT_OFFER_TTL_HOURS,
        now: datetime | None = None,
    ) -> list[OfferEntry]:
        """Wszystkie swieze oferty danego itemu, posortowane po cenie."""
        entries = self._by_item.get(item, [])
        if not entries:
            return []
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=ttl_hours)
        fresh = [
            e for e in entries
            if e.scanned_dt is not None and e.scanned_dt >= cutoff
        ]
        return sorted(fresh, key=lambda e: e.unit_price)

    def items(self) -> list[str]:
        """Wszystkie nazwy itemow w indeksie."""
        return sorted(self._by_item.keys())

    def __len__(self) -> int:
        return sum(len(entries) for entries in self._by_item.values())

    def __contains__(self, item: str) -> bool:
        return item in self._by_item