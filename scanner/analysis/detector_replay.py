"""Offline A/B detektora dymka na zamrożonych ramkach (działka Claude, bez gry).

Problem: detektor `_tooltip_bbox` był strojony na ŻYWYM, ruchomym tłumie o zmiennej
gęstości — każdy bieg to inny tłum, więc nie dało się odróżnić sygnału od szumu
(v10.46 `darkened_share` „poprawił", a realnie obniżył dymki 87%→60%). Ten harness
odtwarza warianty detektora na TYCH SAMYCH ramkach (baseline+hover zapisanych przez
pipeline dla nieudanych dymków), więc porównanie jest deterministyczne.

Kontrakt wejścia (zob. STACK_AWARE_RECOVERY_SPEC.md §6):
    scans/<id>/frames/slot_<NNN>_baseline.png
    scans/<id>/frames/slot_<NNN>_hover.png
To są te same crop-boxy, które dostaje produkcyjny `_tooltip_bbox`.

Metryka: dla każdego wariantu — ile z ZNANYCH porażek (dark_share ich nie złapał)
dany wariant by teraz wykrył. „Wykrył" = znalazł bbox, którego crop przechodzi
marker „Cena sprzedaży" (jeśli podano `marker_fn`/dostępny `win_ocr`); inaczej
liczymy samo znalezienie bboxa (luźniej — patrz UWAGA o precyzji).

UWAGA o precyzji: na zbiorze samych porażek mierzymy ZYSK (recall na pominiętych),
nie regresje. Żeby zmierzyć false-positives nowego wariantu, pipeline musi logować
ramki także dla próbki SUKCESÓW (STACK_AWARE_RECOVERY_SPEC.md §6). Wariant
„ciemny ∩ stabilny w czasie" wymaga KILKU klatek hover — tu nieobjęty (logujemy 1).

Rdzeń (predykaty akceptacji + `detect`) czysty i testowalny. Warstwa I/O czyta PNG.

Uruchomienie:
    python -m scanner.analysis.detector_replay --scans scans [--glob "20260621_16*"]
        [--no-ocr]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Parametry geometryczne zgodne z produkcyjnym _tooltip_bbox (config defaults).
DIFF_THRESHOLD = 18
TOOLTIP_MIN_AREA = 4_000

AcceptFn = Callable[[np.ndarray, np.ndarray], bool]


# --- predykaty akceptacji panelu (czyste, testowalne) -----------------------
# Wejście: panel_gray (uint8/int) wycinek hover, base_gray ten sam wycinek baseline.

def accept_dark_share(panel: np.ndarray, base: np.ndarray) -> bool:
    """Produkcja (po revert v10.46): panel jest CIEMNY i ma jasne linie tekstu."""
    dark = float((panel < 55).mean())
    bright = float((panel > 150).mean())
    return dark >= 0.55 and bright >= 0.015


def accept_darkened_share(panel: np.ndarray, base: np.ndarray) -> bool:
    """v10.46 (zrevertowane): panel POCIEMNIAŁ względem baseline.

    Czuły na RUCH tłumu (jasna postać odeszła → 'pociemniało'), stąd regresja.
    """
    darkened = base.astype(np.int16) - panel.astype(np.int16)
    darkened_share = float((darkened > 12).mean())
    bright = float((panel > 150).mean())
    return darkened_share >= 0.35 and bright >= 0.01


def accept_darkened_uniform(panel: np.ndarray, base: np.ndarray) -> bool:
    """Propozycja: panel POCIEMNIAŁ ORAZ jest JEDNORODNY (gładki ciemny panel).

    Tłum jest teksturowany/wielomodalny → niski uniform_share; dymek to prawie
    jednolite ciemne tło → wysoki. Łączy odporność na jasność (darkened) z
    odróżnieniem od ruchu tłumu (uniform). Liczone z jednej klatki.
    """
    darkened = base.astype(np.int16) - panel.astype(np.int16)
    darkened_share = float((darkened > 12).mean())
    bright = float((panel > 150).mean())
    median = float(np.median(panel))
    uniform_share = float((np.abs(panel.astype(np.int16) - median) < 25).mean())
    return darkened_share >= 0.30 and uniform_share >= 0.55 and bright >= 0.01


VARIANTS: dict[str, AcceptFn] = {
    "dark_share": accept_dark_share,
    "darkened_share": accept_darkened_share,
    "darkened_uniform": accept_darkened_uniform,
}


# --- detekcja (geometria zgodna z produkcją, akceptacja = wariant) ----------

@dataclass(frozen=True, slots=True)
class _Cand:
    x: int
    y: int
    w: int
    h: int
    area: int


def _geometric_candidates(
    baseline_rgb: np.ndarray, hover_rgb: np.ndarray, *, diff_threshold: int
) -> list[_Cand]:
    import cv2

    before = baseline_rgb.astype(np.int16)
    after = hover_rgb.astype(np.int16)
    if before.shape != after.shape:
        return []
    difference = np.max(np.abs(after - before), axis=2).astype(np.uint8)
    mask = (difference > diff_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = hover_rgb.shape[:2]
    minimum_area = min(TOOLTIP_MIN_AREA, max(100, int(w * h * 0.15)))
    minimum_width = min(180, max(30, w // 2))
    minimum_height = min(80, max(20, h // 3))
    out: list[_Cand] = []
    for index in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[index])
        if area < minimum_area:
            continue
        if not (minimum_width <= width <= min(500, w)
                and minimum_height <= height <= min(420, h)):
            continue
        if width * height <= 0 or area / (width * height) < 0.08:
            continue
        out.append(_Cand(x, y, width, height, area))
    return out


def detect(
    baseline_rgb: np.ndarray,
    hover_rgb: np.ndarray,
    accept: AcceptFn,
    *,
    diff_threshold: int = DIFF_THRESHOLD,
) -> tuple[int, int, int, int] | None:
    """Zwróć bbox (l,t,r,b) największego zaakceptowanego panelu albo None.

    Geometria (diff→maska→komponenty→filtry rozmiaru) identyczna jak produkcja;
    różni się tylko predykat `accept` (wariant detektora).
    """
    import cv2

    cands = _geometric_candidates(
        baseline_rgb, hover_rgb, diff_threshold=diff_threshold
    )
    if not cands:
        return None
    hover_gray = cv2.cvtColor(hover_rgb, cv2.COLOR_RGB2GRAY)
    base_gray = cv2.cvtColor(baseline_rgb, cv2.COLOR_RGB2GRAY)
    accepted: list[_Cand] = []
    for c in cands:
        panel = hover_gray[c.y : c.y + c.h, c.x : c.x + c.w]
        base = base_gray[c.y : c.y + c.h, c.x : c.x + c.w]
        if panel.size == 0:
            continue
        if accept(panel, base):
            accepted.append(c)
    if not accepted:
        return None
    best = max(accepted, key=lambda c: c.area)
    h, w = hover_rgb.shape[:2]
    pad = 10
    return (
        max(0, best.x - pad), max(0, best.y - pad),
        min(w, best.x + best.w + pad), min(h, best.y + best.h + pad),
    )


# --- I/O: pary ramek ze skanów ----------------------------------------------

@dataclass(frozen=True, slots=True)
class FramePair:
    scan_id: str
    slot: str
    baseline: Path
    hover: Path


def find_frame_pairs(scans_root: str | Path, pattern: str) -> list[FramePair]:
    root = Path(scans_root)
    pairs: list[FramePair] = []
    if not root.exists():
        return pairs
    for scan_dir in sorted(root.glob(pattern)):
        frames = scan_dir / "frames"
        if not frames.is_dir():
            continue
        for base in sorted(frames.glob("*_baseline.png")):
            hover = base.with_name(base.name.replace("_baseline.png", "_hover.png"))
            if hover.exists():
                slot = base.name.replace("_baseline.png", "")
                pairs.append(FramePair(scan_dir.name, slot, base, hover))
    return pairs


@dataclass(frozen=True, slots=True)
class VariantResult:
    name: str
    detected: int          # znalazł bbox
    marker_confirmed: int  # crop przeszedł marker (gdy marker_fn dostępny)


def replay(
    pairs: list[FramePair],
    *,
    marker_fn: Callable[[np.ndarray], bool] | None = None,
    variants: dict[str, AcceptFn] | None = None,
) -> list[VariantResult]:
    from PIL import Image

    variants = variants or VARIANTS
    det = {n: 0 for n in variants}
    conf = {n: 0 for n in variants}
    for pair in pairs:
        base = np.asarray(Image.open(pair.baseline).convert("RGB"))
        hover = np.asarray(Image.open(pair.hover).convert("RGB"))
        for name, accept in variants.items():
            bbox = detect(base, hover, accept)
            if bbox is None:
                continue
            det[name] += 1
            if marker_fn is not None:
                crop = hover[bbox[1] : bbox[3], bbox[0] : bbox[2]]
                if crop.size and marker_fn(crop):
                    conf[name] += 1
    return [VariantResult(n, det[n], conf[n]) for n in variants]


def _default_marker_fn() -> Callable[[np.ndarray], bool] | None:
    """Marker 'Cena sprzedaży' przez win_ocr (jeśli dostępny), inaczej None."""
    try:
        import re

        import win_ocr  # type: ignore
        from PIL import Image

        if not getattr(win_ocr, "AVAILABLE", False):
            return None

        def marker(crop_rgb: np.ndarray) -> bool:
            image = Image.fromarray(crop_rgb)
            scaled = image.resize(
                (image.width * 2, image.height * 2), Image.Resampling.LANCZOS
            )
            try:
                lines = win_ocr.recognize(scaled)
            except Exception:
                return False
            for line in lines:
                norm = re.sub(r"[^a-z]", "", str(line.get("text") or "").casefold())
                if "cena" in norm and "sprze" in norm:
                    return True
            return False

        return marker
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scanner.analysis.detector_replay",
        description="Offline A/B detektora dymka na zamrożonych ramkach porażek.",
    )
    parser.add_argument("--scans", default="scans", help="katalog skanów")
    parser.add_argument("--glob", default="*", help="wzorzec skanów")
    parser.add_argument(
        "--no-ocr", action="store_true", help="nie weryfikuj markerem (szybciej)"
    )
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

    pairs = find_frame_pairs(args.scans, args.glob)
    print("=== DETECTOR REPLAY (zamrożone ramki porażek) ===")
    if not pairs:
        print(f"  Brak par ramek dla '{args.glob}' w {args.scans}/<id>/frames/.")
        print("  -> Najpierw napraw logowanie ramek (STACK_AWARE_RECOVERY_SPEC §6)")
        print("     i puść bieg, by zapisać slot_<NNN>_baseline/_hover.png.")
        return 2
    marker_fn = None if args.no_ocr else _default_marker_fn()
    results = replay(pairs, marker_fn=marker_fn)
    n = len(pairs)
    print(f"  par ramek (znanych porazek): {n}")
    print(f"  weryfikacja markerem: {'TAK (win_ocr)' if marker_fn else 'NIE'}")
    print(f"\n  {'wariant':<20}{'wykryl':>8}{'marker_ok':>11}")
    base_det = next((r for r in results if r.name == "dark_share"), None)
    for r in results:
        metric = r.marker_confirmed if marker_fn else r.detected
        print(f"  {r.name:<20}{r.detected:>8}{r.marker_confirmed:>11}"
              f"   ({metric}/{n} = {metric/n*100:.0f}% odzysku)")
    if base_det is not None:
        print(f"\n  baseline dark_share odzyskuje {base_det.detected}/{n} "
              f"(z definicji to porazki tego detektora — niska liczba oczekiwana)")
    print("  -> wariant z najwyzszym marker_ok przy zachowanej precyzji = kandydat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
