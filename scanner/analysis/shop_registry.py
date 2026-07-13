"""Etap 1 Mapy Rynku: trwały rejestr sklepów (Claude, offline, zero gry).

Rdzeń tożsamości + lokalizacji rynku. Każdy sklep ma:
  * `fingerprint` — KLUCZ GŁÓWNY (tożsamość wizualna, już liczona w manifeście),
  * `shop_id` — sekwencyjny, przyjazny alias nadany przy 1. spotkaniu, STABILNY
    między sesjami (klik w sklep => dokąd biec, Etap 6),
  * `(x, y, map, channel)` — lokalizacja, NULLABLE (None dopóki DeepSeek nie
    podepnie odczytu live w Etapie 4; rejestr łyka brak pozycji bez błędu).

ZASADA (z MARKET_MAP_PLAN): fingerprint = tożsamość (PK, dedup), pozycja =
atrybut zmienny (sklep może zniknąć/się przesunąć => `last_seen` + TTL 24h).
Budujemy fundament danych OD STARTU; lokalizacje dopełnią się same.

Partycja: `market_map/<mapa>_<kanał>/shops.jsonl` — nigdy nie mieszać kanałów
(ten sam punkt na CH2 to inne sklepy). Jeden `ShopRegistry` = jedna partycja.

Czyste — bez importów gry/OCR/storage. Persystencja: JSONL (czytelny, append-
friendly, mały). Testowalny w całości pod unittest bez okna gry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


SHOPS_FILENAME = "shops.jsonl"
# Idealne lokacje (C5, COVERAGE_MAP_CORE §5b): sklep widziany wielokrotnie ma pozycję =
# MEDIANA próbek OCR (nie last-write). Cap próbek, by `shops.jsonl` nie puchł po wielu biegach.
MAX_POS_SAMPLES = 25


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0
# Świeżość sklepu: po tylu godzinach pozycja/obecność jest podejrzana (restart
# serwera dzienny; sklep mógł zniknąć). Live skip pomija tylko ŚWIEŻE fingerprinty.
DEFAULT_TTL_HOURS = 24


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def partition_key(map_name: str | None, channel: int | None) -> str:
    """Nazwa partycji `<mapa>_<kanał>` z manifestu. None => człony zastępcze.

    Mapa/kanał z OCR bywają dziś śmieciem (oba None) — wtedy `unknown_CHx`.
    Caller, który zna stałą partycję (np. ręcznie zmierzony `glevia_market`),
    podaje ją wprost do `ShopRegistry.open(partition=...)`, omijając kruchy OCR.
    """

    head = (map_name or "unknown").strip().replace(" ", "_") or "unknown"
    tail = f"CH{channel}" if channel is not None else "CHx"
    return f"{head}_{tail}"


@dataclass(slots=True)
class ShopRecord:
    """Trwały rekord jednego sklepu (jedna linia `shops.jsonl`)."""

    shop_id: int
    fingerprint: str
    seller: str = ""
    x: int | None = None
    y: int | None = None
    map_name: str | None = None
    channel: int | None = None
    first_seen: str = field(default_factory=_now_iso)
    last_seen: str = field(default_factory=_now_iso)
    scan_ids: list[str] = field(default_factory=list)
    offers_ref: str | None = None  # wskaźnik do najtańszej oferty (Etap 6)
    # C5 idealne lokacje: surowe obserwacje pozycji (postać przy otwarciu) + pewność.
    # `x,y` to AGREGAT (mediana OCR), nie last-write. Por. COVERAGE_MAP_CORE §5b.
    pos_samples: list[dict[str, Any]] = field(default_factory=list)
    pos_conf: dict[str, Any] | None = None   # {n_ocr, spread} — którym lokacjom ufać

    def add_pos_sample(self, x: float, y: float, *, source: str = "unknown",
                       ts: str | None = None) -> None:
        """Dołóż obserwację pozycji i przelicz idealny punkt (mediana OCR).

        Próbki nie nadpisują się — akumulują (z metką źródła/czasu). Idealny `(x,y)` =
        mediana próbek `ocr*`; jeśli brak OCR, mediana wszystkich (dead-reckoning, niski conf).
        """

        self.pos_samples.append(
            {"x": float(x), "y": float(y), "source": source, "ts": ts or _now_iso()}
        )
        if len(self.pos_samples) > MAX_POS_SAMPLES:   # zachowaj najnowsze (preferuj OCR)
            ocr = [s for s in self.pos_samples if str(s.get("source", "")).startswith("ocr")]
            rest = [s for s in self.pos_samples if not str(s.get("source", "")).startswith("ocr")]
            self.pos_samples = (ocr + rest)[-MAX_POS_SAMPLES:]
        self._aggregate_position()

    def _aggregate_position(self) -> None:
        """Przelicz `x,y` = mediana próbek (OCR > dead-reckoning) + `pos_conf`."""

        ocr = [(s["x"], s["y"]) for s in self.pos_samples
               if str(s.get("source", "")).startswith("ocr")]
        pool = ocr or [(s["x"], s["y"]) for s in self.pos_samples]
        if not pool:
            return
        mx, my = _median([p[0] for p in pool]), _median([p[1] for p in pool])
        spread = max((math.hypot(p[0] - mx, p[1] - my) for p in pool), default=0.0)
        self.x, self.y = int(round(mx)), int(round(my))
        self.pos_conf = {"n_ocr": len(ocr), "spread": round(spread, 1)}

    @property
    def location(self) -> tuple[int, int] | None:
        """`(x, y)` jeśli znana, inaczej None (lokalizacja jeszcze niestemplowana)."""

        if self.x is None or self.y is None:
            return None
        return (self.x, self.y)

    def is_fresh(self, *, now: datetime | None = None, ttl_hours: float = DEFAULT_TTL_HOURS) -> bool:
        """Czy `last_seen` mieści się w oknie TTL (sklep wart pominięcia w skanie)."""

        seen = _parse_ts(self.last_seen)
        if seen is None:
            return False
        now = now or datetime.now(timezone.utc)
        return (now - seen) <= timedelta(hours=ttl_hours)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shop_id": self.shop_id,
            "fingerprint": self.fingerprint,
            "seller": self.seller,
            "x": self.x,
            "y": self.y,
            "map": self.map_name,
            "channel": self.channel,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "scan_ids": list(self.scan_ids),
            "offers_ref": self.offers_ref,
            "pos_samples": list(self.pos_samples),
            "pos_conf": self.pos_conf,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShopRecord":
        samples = list(data.get("pos_samples") or [])
        x, y = data.get("x"), data.get("y")
        # Kompatybilność wsteczna: STARY plik (klucz pos_samples NIEOBECNY) z x,y → zalążkuj
        # próbką „legacy". Klucz obecny-ale-pusty zostaje pusty (round-trip lossless).
        if "pos_samples" not in data and x is not None and y is not None:
            samples = [{"x": float(x), "y": float(y), "source": "legacy",
                        "ts": str(data.get("last_seen") or _now_iso())}]
        return cls(
            shop_id=int(data["shop_id"]),
            fingerprint=str(data["fingerprint"]),
            seller=str(data.get("seller") or ""),
            x=x,
            y=y,
            map_name=data.get("map"),
            channel=data.get("channel"),
            first_seen=str(data.get("first_seen") or _now_iso()),
            last_seen=str(data.get("last_seen") or _now_iso()),
            scan_ids=list(data.get("scan_ids") or []),
            offers_ref=data.get("offers_ref"),
            pos_samples=samples,
            pos_conf=data.get("pos_conf"),
        )


def _as_manifest_dict(manifest: Any) -> dict[str, Any]:
    """Znormalizuj wejście `ingest` do dicta manifestu (dict albo ShopScan)."""

    if isinstance(manifest, dict):
        return manifest
    to_dict = getattr(manifest, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    raise TypeError(f"ingest oczekuje dict lub obiektu z to_dict, dostał {type(manifest)!r}")


class ShopRegistry:
    """Trwały rejestr sklepów dla JEDNEJ partycji (mapa+kanał).

    Klucz główny = fingerprint. `ingest` robi upsert (dedup), `by_id`/`nearest`/
    `in_zone` to haki pod strefowanie (Etap 3) i dobieganie (Etap 6). Rekordy żyją
    w pamięci; `save()` przepisuje `shops.jsonl` (skala setek sklepów => tani full
    rewrite, prostszy i bezpieczniejszy niż append+kompakcja).
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._by_fp: dict[str, ShopRecord] = {}
        self._by_id: dict[int, ShopRecord] = {}
        self._next_id = 1

    # --- konstrukcja / IO -------------------------------------------------

    @classmethod
    def open(
        cls,
        root: str | Path,
        *,
        partition: str | None = None,
        map_name: str | None = None,
        channel: int | None = None,
    ) -> "ShopRegistry":
        """Otwórz rejestr partycji pod `root` (wczytuje istniejący `shops.jsonl`).

        Partycję podaj wprost (`partition="glevia_market"`, omija kruchy OCR mapy)
        albo wyprowadź z (mapa, kanał). Brak katalogu = pusty rejestr (bez błędu).
        """

        part = partition or partition_key(map_name, channel)
        registry = cls(Path(root) / part)
        registry.load()
        return registry

    @property
    def path(self) -> Path:
        return self.directory / SHOPS_FILENAME

    def load(self) -> "ShopRegistry":
        """Wczytaj rekordy z dysku; `shop_id` startuje od max+1 (nigdy nie reużywa)."""

        self._by_fp.clear()
        self._by_id.clear()
        self._next_id = 1
        if not self.path.exists():
            return self
        max_id = 0
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = ShopRecord.from_dict(json.loads(line))
                self._by_fp[record.fingerprint] = record
                self._by_id[record.shop_id] = record
                max_id = max(max_id, record.shop_id)
        self._next_id = max_id + 1
        return self

    def save(self) -> Path:
        """Przepisz `shops.jsonl` (posortowane po shop_id dla stabilnego diffu)."""

        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            for record in sorted(self._by_fp.values(), key=lambda r: r.shop_id):
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        tmp.replace(self.path)  # atomowa podmiana — nie zostawia połowicznego pliku
        return self.path

    # --- mutacja ----------------------------------------------------------

    def ingest(self, manifest: Any) -> ShopRecord | None:
        """Upsert sklepu po fingerprincie. Zwraca rekord albo None (brak PK).

        Manifest bez `shop_fingerprint` (skan nieudany/pusty) NIE trafia do
        rejestru — bez tożsamości nie da się dedupować. Pozycja (`game_position`)
        nullable: gdy None, zostawia poprzednią; gdy podana, odświeża lokalizację.
        """

        data = _as_manifest_dict(manifest)
        fingerprint = data.get("shop_fingerprint")
        if not fingerprint:
            return None

        scan_id = data.get("scan_id")
        seller = str(data.get("seller") or "")
        pos = data.get("game_position")
        loc = tuple(pos) if pos else None
        pos_source = str(data.get("game_position_source") or "unknown")  # C5: ocr/dead_reckoning…
        map_name = data.get("map_name")
        channel = data.get("channel")
        created = str(data.get("created_at") or _now_iso())
        updated = str(data.get("updated_at") or created)

        existing = self._by_fp.get(fingerprint)
        if existing is None:
            record = ShopRecord(
                shop_id=self._next_id,
                fingerprint=str(fingerprint),
                seller=seller,
                map_name=map_name,
                channel=channel,
                first_seen=created,
                last_seen=updated,
                scan_ids=[scan_id] if scan_id else [],
            )
            if loc is not None:  # C5: pierwsza próbka pozycji → idealny punkt = ona
                record.add_pos_sample(loc[0], loc[1], source=pos_source, ts=updated)
            self._next_id += 1
            self._by_fp[fingerprint] = record
            self._by_id[record.shop_id] = record
            return record

        # Upsert istniejącego: rośnie okno czasowe, dokleja scan_id, dopełnia dane.
        if scan_id and scan_id not in existing.scan_ids:
            existing.scan_ids.append(scan_id)
        if _parse_ts(created) and (_parse_ts(existing.first_seen) is None
                                   or _parse_ts(created) < _parse_ts(existing.first_seen)):
            existing.first_seen = created
        if _parse_ts(updated) and (_parse_ts(existing.last_seen) is None
                                   or _parse_ts(updated) > _parse_ts(existing.last_seen)):
            existing.last_seen = updated
        if seller:  # nie nadpisuj znanego sellera pustym
            existing.seller = seller
        if loc is not None:  # C5: dokładaj PRÓBKĘ (mediana OCR), nie nadpisuj last-write
            existing.add_pos_sample(loc[0], loc[1], source=pos_source, ts=updated)
            if map_name is not None:
                existing.map_name = map_name
            if channel is not None:
                existing.channel = channel
        return existing

    def add_pos_sample(self, fingerprint: str, x: float, y: float, *,
                       source: str = "unknown") -> ShopRecord | None:
        """C5: dołóż próbkę pozycji do sklepu (po fingerprincie) → przelicz idealny punkt.

        Dla DeepSeeka (D6): po otwarciu sklepu, gdy znamy pozycję postaci i jej źródło
        (`ocr`/`dead_reckoning…`). `None` gdy fingerprint nieznany.
        """

        rec = self._by_fp.get(fingerprint)
        if rec is None:
            return None
        rec.add_pos_sample(x, y, source=source)
        return rec

    def reaggregate(self) -> None:
        """C5: przelicz idealne pozycje wszystkich sklepów (wołać na końcu biegu)."""

        for rec in self._by_fp.values():
            rec._aggregate_position()

    def ingest_all(self, manifests: Iterable[Any]) -> int:
        """Zassij wiele manifestów; zwraca liczbę faktycznie zapisanych (z PK)."""

        return sum(1 for m in manifests if self.ingest(m) is not None)

    # --- odczyt / zapytania (haki pod strefy i dobieganie) ----------------

    def by_id(self, shop_id: int) -> ShopRecord | None:
        """Rekord po stabilnym aliasie (klik w sklep => dokąd biec, Etap 6)."""

        return self._by_id.get(shop_id)

    def by_fingerprint(self, fingerprint: str) -> ShopRecord | None:
        return self._by_fp.get(fingerprint)

    def nearest(self, x: int, y: int) -> ShopRecord | None:
        """Najbliższy sklep o ZNANEJ lokalizacji (Euklides²). None jeśli brak."""

        best: ShopRecord | None = None
        best_key: tuple[int, int] | None = None
        for record in self._by_fp.values():
            loc = record.location
            if loc is None:
                continue
            dist = (loc[0] - x) ** 2 + (loc[1] - y) ** 2
            key = (dist, record.shop_id)  # remis => niższy shop_id (determinizm)
            if best_key is None or key < best_key:
                best, best_key = record, key
        return best

    def in_zone(self, box: tuple[int, int, int, int]) -> list[ShopRecord]:
        """Sklepy o znanej lokalizacji wewnątrz boxu `(x0, y0, x1, y1)` (inclusive).

        Spina się ze strefowaniem (Etap 3): „które sklepy są już w tej strefie".
        Sklepy bez lokalizacji są pomijane (nie wiadomo, do której strefy należą).
        """

        x0, y0, x1, y1 = box
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        out = [
            r for r in self._by_fp.values()
            if r.location is not None
            and x0 <= r.x <= x1 and y0 <= r.y <= y1
        ]
        out.sort(key=lambda r: r.shop_id)
        return out

    def is_known_fresh(
        self,
        fingerprint: str,
        *,
        now: datetime | None = None,
        ttl_hours: float = DEFAULT_TTL_HOURS,
    ) -> bool:
        """Czy fingerprint jest znany I świeży (=> live skip może go pominąć)."""

        record = self._by_fp.get(fingerprint)
        return record is not None and record.is_fresh(now=now, ttl_hours=ttl_hours)

    def fresh(
        self,
        *,
        now: datetime | None = None,
        ttl_hours: float = DEFAULT_TTL_HOURS,
    ) -> list[ShopRecord]:
        """Sklepy widziane w oknie TTL (świeże), posortowane po shop_id."""

        out = [r for r in self._by_fp.values() if r.is_fresh(now=now, ttl_hours=ttl_hours)]
        out.sort(key=lambda r: r.shop_id)
        return out

    def all(self) -> list[ShopRecord]:
        """Wszystkie rekordy, posortowane po shop_id (kopie nie — read-only użycie)."""

        return sorted(self._by_fp.values(), key=lambda r: r.shop_id)

    def located(self) -> list[ShopRecord]:
        """Tylko sklepy ze znaną lokalizacją (do mapy/nawigacji)."""

        return [r for r in self.all() if r.location is not None]

    def __len__(self) -> int:
        return len(self._by_fp)

    def __contains__(self, fingerprint: object) -> bool:
        return fingerprint in self._by_fp

    def __iter__(self):
        return iter(self.all())


def snapshot(record: ShopRecord) -> ShopRecord:
    """Kopia rekordu (gdy caller chce mutować bez wpływu na rejestr)."""

    return replace(record, scan_ids=list(record.scan_ids))
