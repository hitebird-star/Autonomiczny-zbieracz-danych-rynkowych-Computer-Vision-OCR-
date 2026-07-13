"""CLI analizy: uruchom VlmAnalysisEngine na zapisanym skanie (bez żywej gry).

    python -m scanner.analysis <scan_id>     # analizuj jeden skan
    python -m scanner.analysis --pending     # analizuj wszystkie nieukończone
    python -m scanner.analysis --check        # tylko sprawdź dostępność Ollamy

Po analizie eksportuje VERIFIED do ceny.csv. Pozwala przetestować całą warstwę
analizy na realnych obrazach z `scans/` zanim Codex wepnie ją w pętlę.
"""

from __future__ import annotations

import argparse

from scanner.storage import CSVExporter, ScanRepository

from . import ollama_reader
from .engine import VlmAnalysisEngine


def _summary(scan) -> str:
    counts: dict[str, int] = {}
    for obs in scan.slots.values():
        counts[obs.status.value] = counts.get(obs.status.value, 0) + 1
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return f"{scan.scan_id}: {scan.status.value} | sloty: {parts or 'brak'}"


def main() -> int:
    ap = argparse.ArgumentParser(prog="scanner.analysis")
    ap.add_argument("scan_id", nargs="?", help="identyfikator skanu w scans/")
    ap.add_argument("--scans-dir", default="scans")
    ap.add_argument("--csv", default="ceny.csv")
    ap.add_argument("--pending", action="store_true",
                    help="analizuj wszystkie nieukończone skany")
    ap.add_argument("--no-ocr", action="store_true",
                    help="bez potwierdzenia OCR (sam VLM)")
    ap.add_argument("--check", action="store_true",
                    help="tylko sprawdź dostępność modelu")
    args = ap.parse_args()

    available = ollama_reader.available()
    print(f"Ollama / {ollama_reader.MODEL}: "
          f"{'dostępny' if available else 'NIEDOSTĘPNY'}")
    if args.check:
        return 0 if available else 1
    if not available:
        print("Model niedostępny — uruchom Ollamę i pobierz model.")
        return 1

    repository = ScanRepository(args.scans_dir)
    exporter = CSVExporter(args.csv)
    engine = VlmAnalysisEngine(use_ocr=not args.no_ocr)

    if args.pending:
        targets = [scan.scan_id for scan in repository.pending()]
    elif args.scan_id:
        targets = [args.scan_id]
    else:
        ap.error("podaj scan_id albo --pending")
        return 2

    if not targets:
        print("Brak skanów do analizy.")
        return 0

    for scan_id in targets:
        scan = repository.load(scan_id)
        from scanner.models import ScanStatus
        if scan.status in {ScanStatus.CAPTURED, ScanStatus.QUEUED,
                           ScanStatus.REVIEW}:
            scan.transition(ScanStatus.ANALYZING)
            repository.save_manifest(scan)
        result = engine.analyze(scan, repository)
        repository.save_manifest(result)
        rows = exporter.export(result)
        print(_summary(result) + f" | ceny.csv += {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
