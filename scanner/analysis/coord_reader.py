"""Etap 0 Mapy Rynku: odczyt współrzędnych świata gry ze sceny (Claude, offline).

Glevia wyświetla pod minimapą absolutną pozycję `(X, Y)` + „Glevia2 Farm, CHn".
Ten moduł wyciąga `(x, y, map, channel)` z obrazu sceny — kotwica pokrycia rynku.

ZASADA NADRZĘDNA (z MARKET_MAP_PLAN): cel to ~0% BŁĘDNYCH odczytów, nie 100%
odczytów. Zły `(X,Y)` truje mapę; brak odczytu wypełni dead-reckoning z WASD.
Stąd twarde walidacje i „odrzucaj przy wątpliwości":
  * wymóg NAWIASÓW `(x, y)` — odrzuca zegar `20:06:26` (dwukropki),
  * granice mapy (x,y w sensownym zakresie),
  * `plausible_jump` — odrzut nieprawdopodobnych skoków względem poprzedniej pozycji.

Rdzeń (`parse_coord_text`, `plausible_jump`) czysty i testowalny bez OCR/obrazów.
Warstwa I/O (`read_scene`) używa `win_ocr` na ciasnym ROI. Nie dotyka gry.

Uruchomienie (walidacja niezawodności na istniejących scenach):
    python -m scanner.analysis.coord_reader <session_id> [--dbg dbg/auto]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Szerszy ROI (3 linie: tytuł / współrzędne / zegar). Fallback gdy TIGHT nie trafi.
DEFAULT_ROI = (0.93, 0.155, 1.0, 0.195)
# Ciasny ROI tylko na linii `(x, y)` — domyślny dla live. Zwalidowany na 5 oknach
# 22.06: izoluje linię współrzędnych, odcina tytuł/zegar (mniej śmieci OCR), a przy
# tym poprawia cyfry (DEFAULT mylił 752->742; TIGHT czyta 752).
TIGHT_ROI = (0.93, 0.168, 1.0, 0.183)
# Granice sensownych współrzędnych mapy (Metin2: 0..~1000 na sektor).
COORD_MIN, COORD_MAX = 0, 9999
# Maks. wiarygodny skok między kolejnymi odczytami (jednostki gry).
MAX_JUMP = 60

# Sekwencja podejść dla read_image (live). Tekst (X,Y) renderuje się PROSTO na
# świecie gry — jeden ROI/preprocessing nie wystarcza. Próbujemy po kolei, bierzemy
# pierwszy parsujący się odczyt. (roi, scale, binarize|None). Zwalidowane: pokrywa
# 4/5 okien 22.06 z wartościami zgodnymi z truth; 5. (ekstremalny tłok) nieczytelne.
DEFAULT_ATTEMPTS = (
    (TIGHT_ROI, 5, 190),
    (TIGHT_ROI, 5, "white_outline"),
    (TIGHT_ROI, 5, "white"),
    (TIGHT_ROI, 5, None),
    (DEFAULT_ROI, 5, 190),
    (DEFAULT_ROI, 5, "white_outline"),
    (DEFAULT_ROI, 5, "white"),
)

# Ratunkowy odczyt startowy. Na przesuniętym oknie Glevii linia `(X, Y)` bywa
# odrobinę bardziej w lewo niż TIGHT/DEFAULT (x=0.93 ucinało np. `(485,` i OCR
# widział samo `720)`). NIE wkładamy tego do DEFAULT_ATTEMPTS, bo szerszy crop
# potrafi złapać zegar/śmieci; używać tylko po pełnym missie domyślnego OCR i
# zawsze filtrować przez `accept_reading(..., bounds=...)`.
STARTUP_FALLBACK_ATTEMPTS = (
    ((0.91, 0.160, 1.0, 0.190), 4, None),
    ((0.91, 0.160, 1.0, 0.190), 5, "white_outline"),
)

# OCR myli przecinek z wieloma znakami na tle świata gry: , . ; * % _ ' ` ’.
# Separator WYMAGANY (nie opcjonalny): bez niego „(4731665)" nie dopasuje się →
# pomijamy zamiast zgadywać zły split na (4731,665). ':' CELOWO wykluczony — to
# strażnik zegara 20:06:26 / 21:37:24 (wraz z wymogiem nawiasów).
_SEP = r"[,.;*%_'’`\s]"
_COORD_RE = re.compile(r"\(\s*(\d{1,4})\s*" + _SEP + r"\s*(\d{1,4})\s*\)?")
_CHANNEL_RE = re.compile(r"\bCH\s*(\d+)\b", re.IGNORECASE)
_OCR_DIGIT_TRANSLATION = str.maketrans(
    {
        "B": "8",
        "b": "8",
        "O": "0",
        "o": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "Z": "2",
        "G": "6",
    }
)
@dataclass(frozen=True, slots=True)
class CoordParse:
    x: int
    y: int
    channel: int | None
    map_name: str | None


def parse_coord_text(text: str) -> CoordParse | None:
    """Wyciągnij `(x, y[, CHn, map])` z surowego tekstu OCR. Czyste — bez I/O.

    Wymaga NAWIASÓW wokół pary liczb (zegar `HH:MM:SS` ma dwukropki, nie nawiasy),
    odrzuca współrzędne poza granicami. Kanał/mapa best-effort (mogą być None).
    """

    text = text or ""
    match = _COORD_RE.search(text.translate(_OCR_DIGIT_TRANSLATION))
    if not match:
        return None
    x, y = int(match.group(1)), int(match.group(2))
    if not (COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX):
        return None

    channel = None
    channel_match = _CHANNEL_RE.search(text)
    if channel_match:
        channel = int(channel_match.group(1))

    # Nazwa mapy: tekst przed znacznikiem kanału lub przed nawiasem, best-effort.
    map_name = None
    head = text[: channel_match.start()] if channel_match else text[: match.start()]
    head = head.strip(" ,.-\n\t")
    if head:
        map_name = head.splitlines()[-1].strip(" ,.-")[:40] or None

    return CoordParse(x=x, y=y, channel=channel, map_name=map_name)


def plausible_jump(
    previous: tuple[int, int] | None,
    current: tuple[int, int],
    max_jump: int = MAX_JUMP,
) -> bool:
    """Czy skok z `previous` do `current` jest wiarygodny (Czebyszew <= max_jump).

    Brak poprzedniej pozycji = akceptuj (pierwszy odczyt nie ma odniesienia).
    """

    if previous is None:
        return True
    return (
        abs(current[0] - previous[0]) <= max_jump
        and abs(current[1] - previous[1]) <= max_jump
    )


# Granice sensownych współrzędnych dla trybu LIVE — twardszy filtr niż
# COORD_MIN/MAX (0..9999 w parse). Łapie śmieci OCR typu „471"->4713, „722"->1732.
DEFAULT_BOUNDS = (50, 1500)


def in_bounds(value: int, bounds: tuple[int, int] = DEFAULT_BOUNDS) -> bool:
    return bounds[0] <= value <= bounds[1]


def accept_reading(
    previous: tuple[int, int] | None,
    current: tuple[int, int],
    *,
    bounds: tuple[int, int] | tuple[int, int, int, int] = DEFAULT_BOUNDS,
    max_jump: int = MAX_JUMP,
) -> bool:
    """Czy odczyt `current` jest wiarygodny do ZAPISU (live + offline).

    Odrzuca: poza granicami (śmieć OCR jak 4713/1732) oraz nieprawdopodobny skok
    względem poprzedniego (chód ciągły => realne skoki małe). Pierwszy odczyt
    (`previous=None`) akceptowany tylko jeśli w granicach (nie kotwicz na śmieciu).

    Współdzielone: `tools/perimeter_mapper.py` (Claude) ORAZ stemplowanie
    `game_position` w manifeście (DeepSeek, Stage 4) — jedno źródło prawdy o tym,
    czy odczyt współrzędnej można zaufać.

    bounds: 2-krotka = (min, max) dla obu osi (perimeter_mapper).
            4-krotka = (x_min, x_max, y_min, y_max) per-oś (Stage 4 farma).
    """

    if len(bounds) == 4:
        x_min, x_max, y_min, y_max = bounds
        if not (x_min <= current[0] <= x_max and y_min <= current[1] <= y_max):
            return False
    else:
        if not (in_bounds(current[0], bounds) and in_bounds(current[1], bounds)):
            return False
    return plausible_jump(previous, current, max_jump)


def _ocr_text(
    image,
    roi: tuple[float, float, float, float],
    scale: int,
    *,
    binarize: int | str | None = None,
) -> str:
    import win_ocr
    from PIL import Image

    width, height = image.size
    box = (
        int(width * roi[0]),
        int(height * roi[1]),
        int(width * roi[2]),
        int(height * roi[3]),
    )
    crop = image.crop(box)
    if binarize in ("white", "white_outline"):
        import numpy as np

        rgb = np.asarray(crop.convert("RGB"), dtype=np.uint8)
        white_mask = (
            (rgb[:, :, 0] > 170)
            & (rgb[:, :, 1] > 170)
            & (rgb[:, :, 2] > 170)
        )
        if binarize == "white_outline":
            # Koordy w Metinie to biały font z czarnym obrysem renderowany
            # bezpośrednio na świecie. Na jasnym tle zwykły white-key łapie
            # trawę/piasek razem z tekstem; warunek "obok jest ciemny obrys"
            # zostawia głównie glyphy i odcina jasne tło.
            dark_mask = (
                (rgb[:, :, 0] < 95)
                & (rgb[:, :, 1] < 95)
                & (rgb[:, :, 2] < 95)
            )
            padded = np.pad(dark_mask, 1, mode="constant", constant_values=False)
            near_dark = np.zeros_like(dark_mask, dtype=bool)
            for dy in range(3):
                for dx in range(3):
                    near_dark |= padded[dy : dy + dark_mask.shape[0], dx : dx + dark_mask.shape[1]]
            mask = white_mask & near_dark
        else:
            mask = white_mask
        crop = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        crop = crop.resize(
            (crop.width * scale, crop.height * scale), Image.Resampling.NEAREST
        )
    elif binarize is not None:
        # Tekst (X,Y) jest jasny na ciemniejszym świecie: progujemy w skali szarości,
        # potem upscale NEAREST (po progu zachowuje ostre krawędzie glifów lepiej niż
        # LANCZOS). Odzyskuje odczyty na zatłoczonym tle, które RGB+LANCZOS gubi.
        gray = crop.convert("L").point(lambda p: 255 if p > binarize else 0)
        crop = gray.resize(
            (gray.width * scale, gray.height * scale), Image.Resampling.NEAREST
        )
    else:
        # LANCZOS znacząco poprawia OCR małego fontu (bez niego rate spada ~6x).
        crop = crop.resize(
            (crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS
        )
    return " ".join(str(line.get("text") or "") for line in win_ocr.recognize(crop))


def read_image(
    image,
    *,
    roi: tuple[float, float, float, float] | None = None,
    scale: int = 5,
    attempts=DEFAULT_ATTEMPTS,
) -> CoordParse | None:
    """Odczytaj współrzędne z obrazu PIL (I/O: win_ocr). Dla trybu live.

    Tekst (X,Y) renderuje się PROSTO na świecie gry (bez tła UI) → jeden
    ROI/preprocessing nie wystarcza. Próbujemy listy podejść (`attempts`:
    (roi, scale, binarize|None)) i zwracamy PIERWSZY parsujący się odczyt.
    Domyślnie `DEFAULT_ATTEMPTS` (TIGHT czysty → TIGHT bin → DEFAULT bin).

    `roi=` (pojedynczy) wymusza tryb JEDNO-podejściowy (sweep offline / testy),
    z zachowaniem wstecznej kompatybilności starego API jedno-ROI.

    Nie otwiera pliku — przyjmuje gotowy `Image` (np. zrzut okna gry).
    """

    if roi is not None:
        attempts = ((roi, scale, None),)
    for a_roi, a_scale, a_bin in attempts:
        parsed = parse_coord_text(_ocr_text(image, a_roi, a_scale, binarize=a_bin))
        if parsed is not None:
            return parsed
    return None


def read_scene(
    path: str | Path,
    *,
    roi: tuple[float, float, float, float] | None = None,
    scale: int = 5,
) -> CoordParse | None:
    """Odczytaj współrzędne z pliku sceny (I/O: PIL + win_ocr).

    `roi=None` (domyślnie) → multi-attempt jak live. `roi=` wymusza pojedynczy ROI
    (sweep offline).
    """

    from PIL import Image

    image = Image.open(path).convert("RGB")
    return read_image(image, roi=roi, scale=scale)


@dataclass(frozen=True, slots=True)
class ReadStats:
    scenes: int
    read: int
    rejected_jump: int
    positions: tuple[tuple[int, int], ...]

    @property
    def read_rate(self) -> float:
        return self.read / self.scenes if self.scenes else 0.0

    @property
    def distinct_positions(self) -> int:
        return len(set(self.positions))


def validate_session(
    session_dir: Path,
    *,
    roi: tuple[float, float, float, float] | None = None,
    max_jump: int = MAX_JUMP,
) -> ReadStats:
    """Przejdź sceny rundami; zlicz odczyty, odrzuty skoków, ciąg pozycji.

    `roi=None` → multi-attempt jak live (DEFAULT_ATTEMPTS).
    """

    scenes = sorted(session_dir.glob("round_*_scene.png"))
    read = rejected = 0
    positions: list[tuple[int, int]] = []
    previous: tuple[int, int] | None = None
    for scene in scenes:
        parsed = read_scene(scene, roi=roi)
        if parsed is None:
            continue
        current = (parsed.x, parsed.y)
        if not plausible_jump(previous, current, max_jump):
            rejected += 1
            continue
        read += 1
        positions.append(current)
        previous = current
    return ReadStats(len(scenes), read, rejected, tuple(positions))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scanner.analysis.coord_reader",
        description="Walidacja odczytu współrzędnych świata ze scen sesji.",
    )
    parser.add_argument("session", help="id sesji w --dbg")
    parser.add_argument("--dbg", default="dbg/auto", help="katalog sesji")
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    session_dir = Path(args.dbg) / args.session
    if not session_dir.exists():
        print(f"Brak sesji {session_dir}.")
        return 2
    stats = validate_session(session_dir)
    print(f"=== ODCZYT WSPÓŁRZĘDNYCH {args.session} ===")
    print(f"  scen            : {stats.scenes}")
    print(f"  odczytano       : {stats.read} ({stats.read_rate * 100:.0f}%)")
    print(f"  odrzucone skoki  : {stats.rejected_jump}")
    print(f"  różne pozycje    : {stats.distinct_positions}")
    if stats.positions:
        print(f"  zakres x        : {min(p[0] for p in stats.positions)}..{max(p[0] for p in stats.positions)}")
        print(f"  zakres y        : {min(p[1] for p in stats.positions)}..{max(p[1] for p in stats.positions)}")
        print(f"  ciąg            : {' '.join(f'{p[0]},{p[1]}' for p in stats.positions[:12])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
