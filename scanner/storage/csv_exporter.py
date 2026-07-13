"""Eksport zweryfikowanych ofert do formatu ``metin2_market.py``."""

from __future__ import annotations

import csv
import os
import re
import threading
from pathlib import Path
from typing import Any, Iterable

from scanner.models import ItemObservation, ScanStatus, ShopScan


# ``stack_count`` to FINALNA liczba stacków oferty (z pipeline'u: grupy ikon +
# konsensus Phase B) — to ona idzie do analizatora. Pierwsze szesc kolumn pozostaje
# zgodne z ``metin2_market.py`` — kolumny audytu dopisujemy NA KONCU.
#
# Kolumny audytu KOMPLETNOSCI sklepu (z ``ShopScan.inventory_audit``, patrz
# ``analysis/inventory_audit.py``) — poziom SKLEPU, denormalizowane na kazdy wiersz.
# DETERMINISTYCZNE, bez VLM (nazwy nie ma na shop.png), nie blokuja CSV:
# - ``occupied_slots``   = wszystkie wykryte sloty sklepu;
# - ``unassigned_slots`` = occupied - sloty bezpiecznie przypisane (kompletnosc);
# - ``audit_status``     = complete / partial (partial = zostaly nieprzypisane sloty).
CSV_COLUMNS = (
    "item", "price", "quantity", "stack_count", "timestamp", "source", "seller",
    "occupied_slots", "unassigned_slots", "audit_status",
)
MAX_UNIT_PRICE = 1_000_000_000_000


class CSVExporter:
    def __init__(
        self,
        path: str | Path = "ceny.csv",
        *,
        source: str = "scanner-v10",
        max_unit_price: int = MAX_UNIT_PRICE,
    ):
        self.path = Path(path)
        self.source = source
        self.max_unit_price = int(max_unit_price)
        self._lock = threading.RLock()

    def export(self, scan: ShopScan) -> int:
        rows = self._aggregate(list(self.rows_for(scan)))
        rows = self._apply_audit(rows, scan)
        if not rows:
            return 0
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._ensure_schema()
            existing = self._existing_keys()
            new_rows = [
                row for row in rows if self._row_key(row) not in existing
            ]
            if not new_rows:
                return 0
            write_header = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                if write_header:
                    writer.writeheader()
                writer.writerows(new_rows)
                handle.flush()
                os.fsync(handle.fileno())
            return len(new_rows)

    def rows_for(self, scan: ShopScan) -> Iterable[dict[str, Any]]:
        for observation in scan.slots.values():
            if observation.status is not ScanStatus.VERIFIED:
                continue
            row = self._observation_row(scan, observation)
            if row is not None:
                yield row

    def _observation_row(
        self, scan: ShopScan, observation: ItemObservation
    ) -> dict[str, Any] | None:
        result = observation.validation or {}
        ai = observation.ai or {}
        item = result.get("item") or ai.get("item")
        unit_price = result.get("unit_price") or ai.get("unit_price")
        quantity = result.get("quantity")
        if quantity is None:
            quantity = ai.get("quantity")
        if not isinstance(item, str) or not item.strip():
            return None
        if not isinstance(unit_price, int) or unit_price <= 0:
            return None
        if unit_price > self.max_unit_price:
            return None
        if not isinstance(quantity, int) or quantity <= 0:
            return None
        seller = (
            str(scan.seller).strip()
            or str(scan.shop_fingerprint or "").strip()
            or scan.scan_id
        )
        return {
            "item": item.strip(),
            "price": unit_price,
            "quantity": quantity,
            "stack_count": 1,
            "timestamp": scan.updated_at,
            "source": self.source,
            "seller": seller,
            # Denormalizowane z audytu kompletnosci w _apply_audit (poziom sklepu).
            "occupied_slots": "",
            "unassigned_slots": "",
            "audit_status": "",
        }

    @staticmethod
    def _apply_audit(
        rows: list[dict[str, Any]], scan: ShopScan
    ) -> list[dict[str, Any]]:
        """Denormalizuj kompletnosc sklepu na kazdy wiersz (NIE bramkuje CSV).

        Audyt jest deterministyczny i SKLEPOWY: te same ``occupied_slots`` /
        ``unassigned_slots`` / ``audit_status`` trafiaja na wszystkie oferty sklepu.
        Brak audytu (stare skany) → kolumny puste, bez regresji.
        """

        audit = getattr(scan, "inventory_audit", None)
        if not isinstance(audit, dict):
            return rows
        occupied = audit.get("occupied_slots")
        unassigned = audit.get("unassigned_slots")
        status = audit.get("audit_status")
        for row in rows:
            row["occupied_slots"] = occupied if isinstance(occupied, int) else ""
            row["unassigned_slots"] = unassigned if isinstance(unassigned, int) else ""
            row["audit_status"] = status if isinstance(status, str) else ""
        return rows

    def _existing_keys(self) -> set[tuple[str, ...]]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return set()
        try:
            with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
                return {
                    self._row_key(row)
                    for row in csv.DictReader(handle)
                    if row
                }
        except (OSError, csv.Error):
            return set()

    def _ensure_schema(self) -> None:
        """Rozszerz stary CSV o ``stack_count`` bez utraty danych.

        Historyczne wiersze nie zawieraja wiarygodnej liczby stackow: znaja
        tylko zsumowana ilosc. Zostawiamy wiec to pole puste zamiast wpisywac
        falszywe ``1``. Nowe wpisy zawsze maja dokladna wartosc.
        """

        if not self.path.exists() or self.path.stat().st_size == 0:
            return
        try:
            with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                fieldnames = tuple(reader.fieldnames or ())
                if fieldnames == CSV_COLUMNS:
                    return
                rows = list(reader)
        except (OSError, csv.Error):
            return

        # Nie nadpisuj pliku o nieznanym, uszkodzonym schemacie. Wymagamy
        # minimum kontraktu potrzebnego analizatorowi rynku.
        if not {"item", "price", "quantity"}.issubset(fieldnames):
            return

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                writer.writeheader()
                for row in rows:
                    writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    @staticmethod
    def _row_key(row: dict[str, Any]) -> tuple[str, ...]:
        return tuple(str(row.get(column, "")) for column in CSV_COLUMNS)

    @staticmethod
    def _item_key(item: str) -> str:
        return re.sub(r"\s+", " ", item).strip().casefold()

    def _aggregate(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Połącz identyczne oferty w jednym sklepie i zsumuj liczbę sztuk."""

        grouped: dict[tuple[str, int, str, str], dict[str, Any]] = {}
        for row in rows:
            key = (
                self._item_key(str(row["item"])),
                int(row["price"]),
                str(row["seller"]).casefold(),
                str(row["source"]),
            )
            if key not in grouped:
                grouped[key] = dict(row)
                continue
            grouped[key]["quantity"] = (
                int(grouped[key]["quantity"]) + int(row["quantity"])
            )
            grouped[key]["stack_count"] = (
                int(grouped[key]["stack_count"]) + int(row["stack_count"])
            )
        return list(grouped.values())
