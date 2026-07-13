# -*- coding: utf-8 -*-
"""Harvest crops celów z `dbg/auto/*` etykietowanych WYNIKIEM kliknięcia.

Po co. Klasyfikator celu (`ShopTargetVerifier`) jest ślepy na realne confusery
(postać/tekstury), bo trenowano go na 30 ręcznych negatywach. Tymczasem każdy
`capture_outcome` w `events.jsonl` jest DARMOWĄ etykietą: `queued`/`duplicate_*`
= sklep się otworzył (REAL), `shop_window_not_detected` = klik w nic (FALSE).
Crop odtwarzamy 1:1 z inferencji: `crop_target(round_*_scene.png, candidate.local, 96)`.

Wynik = `dataset/targets/{real,false}/<sesja>_<track>.png` + `manifest.csv`,
gotowe dla `python -m scanner.detection.train_target_verifier --data dataset/targets`.

Uruchomienie:
    python -m scanner.detection.harvest_targets
    python -m scanner.detection.harvest_targets --debug-root dbg/auto --out dataset/targets
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from .target_verifier import crop_target

# Wynik kliknięcia → etykieta. duplicate_* = okno SIĘ otworzyło (mamy fingerprint)
# → realny sklep. shop_window_not_detected = klik w teksturę/postać → false.
REAL_STATUSES = {"captured", "queued"}
REAL_REASONS = {"duplicate_known_fresh", "duplicate_shop_fingerprint"}
FALSE_REASONS = {"shop_window_not_detected"}


def _iter_events(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _label(status: str, reason: str | None) -> str | None:
    # ``captured`` jest sukcesem bez asynchronicznego workera analizy;
    # ``queued`` jest tym samym sukcesem, gdy worker jest włączony.
    if status in REAL_STATUSES:
        return "real"
    if reason in REAL_REASONS:
        return "real"
    if reason in FALSE_REASONS:
        return "false"
    return None  # capture_exception itp. — pomijamy (niejednoznaczne)


def harvest_session(session_dir: Path, out_root: Path, crop_size: int) -> list[dict]:
    events = session_dir / "events.jsonl"
    if not events.exists():
        return []
    session = session_dir.name
    sess_key = session.replace("_", "x")  # by session_from_name zwrócił pełną sesję

    rows: list[dict] = []
    cur_scene: str | None = None
    cur_cands: dict[str, dict] = {}
    scene_cache: dict[str, Image.Image | None] = {}

    def load_scene(name: str) -> Image.Image | None:
        if name not in scene_cache:
            p = session_dir / name
            try:
                scene_cache[name] = Image.open(p).convert("RGB") if p.exists() else None
            except (OSError, ValueError):
                scene_cache[name] = None
        return scene_cache[name]

    for e in _iter_events(events):
        ev = e.get("event")
        if ev == "detection":
            cur_scene = (e.get("files") or {}).get("scene")
            cur_cands = {}
            for c in e.get("candidates", []):
                tid = c.get("track_id")
                if tid and c.get("local"):
                    cur_cands[tid] = c
        elif ev == "capture_outcome":
            tid = e.get("track_id")
            label = _label(e.get("status"), e.get("reason"))
            cand = cur_cands.get(tid)
            if label is None or cand is None or cur_scene is None:
                continue
            scene = load_scene(cur_scene)
            if scene is None:
                continue
            local = tuple(int(v) for v in cand["local"])
            crop = crop_target(scene, local, size=crop_size)
            out_dir = out_root / label
            out_dir.mkdir(parents=True, exist_ok=True)
            fname = f"{sess_key}_{tid}.png"
            crop.save(out_dir / fname)
            rows.append({
                "session": session,
                "track_id": tid,
                "label": label,
                "status": e.get("status"),
                "reason": e.get("reason") or "",
                "local_x": local[0], "local_y": local[1],
                "area": cand.get("area"),
                "hybrid_score": cand.get("hybrid_score"),
                "scene": cur_scene,
                "crop": str((out_root / label / fname).as_posix()),
            })
    return rows


def run(args: argparse.Namespace) -> int:
    debug_root = Path(args.debug_root)
    out_root = Path(args.out)
    if getattr(args, "clean", False):
        for label in ("real", "false"):
            directory = out_root / label
            if directory.exists():
                for path in directory.glob("*.png"):
                    path.unlink()
        manifest = out_root / "manifest.csv"
        if manifest.exists():
            manifest.unlink()
    sessions = sorted(p for p in debug_root.glob("*") if (p / "events.jsonl").exists())
    if not sessions:
        print(f"BRAK sesji z events.jsonl w {debug_root}")
        return 1

    all_rows: list[dict] = []
    for s in sessions:
        rows = harvest_session(s, out_root, args.crop_size)
        all_rows.extend(rows)

    if not all_rows:
        print("Nic nie zebrano (brak capture_outcome z etykietą / brak scen).")
        return 1

    manifest = out_root / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    labels = Counter(r["label"] for r in all_rows)
    reasons = Counter(r["reason"] or r["status"] for r in all_rows)
    per_session = Counter(r["session"] for r in all_rows)
    sess_label = Counter((r["session"], r["label"]) for r in all_rows)

    print(f"Sesje: {len(sessions)}  zebrane crops: {len(all_rows)}")
    print(f"Etykiety: {dict(labels)}")
    print(f"Wg wyniku: {dict(reasons)}")
    print(f"Sesji z danymi: {len(per_session)}")
    print("Per sesja (real/false):")
    for sess in sorted(per_session):
        print(f"  {sess}: real={sess_label.get((sess,'real'),0)} false={sess_label.get((sess,'false'),0)}")
    print(f"\nZapisano: {out_root}/real, {out_root}/false, {manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m scanner.detection.harvest_targets")
    parser.add_argument("--debug-root", default="dbg/auto",
                        help="katalog sesji diagnostycznych (z round_*_scene.png + events.jsonl)")
    parser.add_argument("--out", default="dataset/targets",
                        help="katalog wyjściowy datasetu (real/ false/ + manifest.csv)")
    parser.add_argument("--clean", action="store_true",
                        help="usuń stare cropy z --out przed odtworzeniem zbioru")
    parser.add_argument("--crop-size", type=int, default=96,
                        help="rozmiar cropu = crop_size modelu (domyślnie 96)")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
