"""Składanie zależności i CLI nowego skanera v10."""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from pathlib import Path

from scanner.analysis.grid_calibration import (
    apply_calibration,
    calibration_from_points,
    render_grid_overlay,
)
from scanner.capture import SellerReader, ShopCapturer, TooltipCapturer
from scanner.config import ScannerSettings, load_settings
from scanner.detection import ShopDetector, ShopInteractor, ShopTracker, ShopWindowProbe
from scanner.navigation import MovementController, SerpentineRoutePlanner
from scanner.pipeline import AnalysisWorker, AutonomousMarketLoop, GameCapturePipeline
from scanner.runtime import (
    GameWindow,
    InputBackend,
    MSSScreen,
    PyAutoGUIInput,
    disable_console_quick_edit,
)
from scanner.storage import CSVExporter, ScanRepository


MARKET_PARTITION_DIR = Path("market_map/glevia_market")
RUNTIME_MAP_STATE_FILES = (
    "coverage.json",
    "movement_memory.json",
    "shops.jsonl",
    "zones.json",
)


def _reset_market_runtime_state(
    market_dir: Path = MARKET_PARTITION_DIR,
    *,
    backup: bool = True,
) -> list[tuple[Path, Path | None]]:
    """Wyczyść stan operacyjny mapy, zostawiając ręcznie wyznaczoną granicę.

    `farm_map.json` jest źródłem prawdy o obszarze rynku/farmy i nie należy go
    usuwać przy zwykłym "czystym runie". Reset obejmuje tylko dane, które mogą
    zatruć kolejny bieg: pokrycie, pamięć ruchu i rejestr pozycji sklepów.
    """

    moved: list[tuple[Path, Path | None]] = []
    market_dir.mkdir(parents=True, exist_ok=True)
    backup_dir = (
        market_dir / "reset_backups" / time.strftime("%Y%m%d_%H%M%S")
        if backup
        else None
    )
    for name in RUNTIME_MAP_STATE_FILES:
        path = market_dir / name
        if not path.exists():
            continue
        if backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            target = backup_dir / name
            shutil.move(str(path), str(target))
            moved.append((path, target))
        else:
            path.unlink()
            moved.append((path, None))
    return moved


def _countdown(
    window: GameWindow,
    input_backend: InputBackend,
    *,
    clock=None,
    stable_seconds: float = 1.5,
    poll_interval: float = 0.1,
) -> None:
    """Czekaj pasywnie, aż Glevia2 będzie stabilnie aktywnym oknem."""

    clock = clock or time
    print(
        "\n=== ARMED ===\n"
        "Przygotuj otwarty sklep i przejdź do Glevia2.\n"
        "Start nastąpi automatycznie po stabilnym wykryciu gry.\n"
        "Nie wracaj do PowerShella. Awaryjne zatrzymanie: Ctrl+C."
    )
    foreground_since = None
    while True:
        if window.is_foreground():
            now = clock.monotonic()
            if foreground_since is None:
                foreground_since = now
                print("Glevia2 wykryta — potwierdzam stabilny fokus...")
            if now - foreground_since >= stable_seconds:
                print("Glevia2 aktywna — start.")
                return
        else:
            if foreground_since is not None:
                print("Fokus utracony przed startem — czekam ponownie.")
            foreground_since = None
        clock.sleep(poll_interval)


def _await_initial_position(
    pipeline: GameCapturePipeline,
    *,
    timeout: float = 6.0,
    poll_interval: float = 0.20,
    clock=None,
) -> tuple[int, int] | None:
    """Poczekaj na pierwszy wiarygodny odczyt ``(X,Y)`` bez ruchu postaci.

    Odczyt współrzędnych renderowany jest bezpośrednio na scenie i pojedyncza
    klatka może przypadkowo wypaść (animacja, dymek, tekstura tła). Coverage
    drive potrzebuje kotwicy, ale nie powinien przez taki pojedynczy miss
    odmawiać startu. Ten helper tylko ponawia OCR — nie wysyła kliknięć ani
    klawiszy, a po limicie nadal bezpiecznie blokuje automatyczny ruch.
    """

    clock = clock or time
    deadline = clock.monotonic() + max(0.0, timeout)
    attempts = 0
    while True:
        attempts += 1
        position = pipeline.read_current_position()
        if position is not None:
            if attempts > 1:
                print(f"Współrzędne odczytane po {attempts} próbach.")
            return position
        if clock.monotonic() >= deadline:
            return None
        clock.sleep(max(0.01, poll_interval))


def build_live_pipeline(
    settings: ScannerSettings,
    *,
    scans_dir: str | Path,
    analyze: bool = False,
    csv_path: str | Path = "ceny.csv",
    use_ocr: bool = True,
    units_per_step: float = 7.0,
    odometry_vectors: dict[str, tuple[float, float]] | None = None,
    vlm_shop_audit: bool = False,
) -> tuple[
    GameCapturePipeline,
    ShopDetector,
    ShopTracker,
    MovementController,
    GameWindow,
    MSSScreen,
    AnalysisWorker | None,
]:
    screen = MSSScreen()
    input_backend = PyAutoGUIInput()
    window = GameWindow(settings.window_title)
    window_rect = window.locate()
    tracker = ShopTracker()
    repository = ScanRepository(scans_dir)
    worker = None
    if analyze:
        from scanner.analysis import VlmAnalysisEngine

        worker = AnalysisWorker(
            repository,
            VlmAnalysisEngine(use_ocr=use_ocr, use_vlm_shop_audit=vlm_shop_audit),
            exporter=CSVExporter(csv_path, source=settings.source),
        )
    probe = ShopWindowProbe(screen, settings.grid)
    interactor = ShopInteractor(
        input_backend,
        probe,
        retry_timeout=settings.retry_open_timeout,
    )
    shop_capture = ShopCapturer(screen, input_backend, settings.grid)
    seller_reader = SellerReader(screen, settings.grid)
    tooltip_capture = TooltipCapturer(
        screen,
        input_backend,
        settings.grid,
        settings.capture,
        bounds=window_rect.box,
        focus=window,
    )
    pipeline = GameCapturePipeline(
        interactor,
        shop_capture,
        tooltip_capture,
        repository,
        tracker,
        input_backend,
        close_key=settings.close_key,
        open_timeout=settings.open_timeout,
        seller_provider=seller_reader.read,
        analysis_queue=worker,
        focus=window,
        window_box=window_rect.box if window_rect else None,
        units_per_step=units_per_step,
        odometry_vectors=odometry_vectors,
    )
    return (
        pipeline,
        ShopDetector(settings.detector),
        tracker,
        MovementController(input_backend),
        window,
        screen,
        worker,
    )


def _analysis_ready(enabled: bool) -> bool:
    if not enabled:
        return True
    from scanner.analysis import ollama_reader

    if ollama_reader.available():
        return True
    print(
        f"BŁĄD: Ollama lub model {ollama_reader.MODEL!r} jest niedostępny. "
        "Uruchom Ollamę albo pomiń --analyze."
    )
    return False


def command_capture_open(args: argparse.Namespace) -> int:
    if not _analysis_ready(args.analyze):
        return 1
    settings = load_settings(args.config)
    pipeline, _, tracker, _, window, screen, worker = build_live_pipeline(
        settings,
        scans_dir=args.scans,
        analyze=args.analyze,
        csv_path=args.csv,
        use_ocr=not args.no_ocr,
        vlm_shop_audit=getattr(args, "vlm_shop_audit", False),
    )
    _countdown(window, pipeline.input)
    # Ten tryb nie może przypadkiem kliknąć rynku, gdy sklep nie jest otwarty.
    probe = ShopWindowProbe(screen, settings.grid)
    score = probe.score()
    if score < probe.minimum_grid_score:
        window_rect = window.locate()
        debug_dir = Path("dbg")
        debug_dir.mkdir(exist_ok=True)
        pipeline.shop_capturer.capture_grid().save(
            debug_dir / "capture_open_failed_grid.png"
        )
        screen.grab(window_rect.box).save(
            debug_dir / "capture_open_failed_client.png"
        )
        print(
            f"grid_score={score:.2f}, "
            f"wymagane={probe.minimum_grid_score:.2f}"
        )
        print("Diagnostyka: dbg/capture_open_failed_grid.png i dbg/capture_open_failed_client.png")
        print("BŁĄD: nie wykryto otwartego okna sklepu. Otwórz sklep i uruchom ponownie.")
        return 1
    from scanner.detection import TrackedShop

    outcome = pipeline.capture(
        TrackedShop("manual-open", settings.grid.origin)
    )
    if worker is not None:
        worker.join()
        worker.stop(2.0)
        outcome_status = ScanRepository(args.scans).load(outcome.scan.scan_id).status
    else:
        outcome_status = outcome.scan.status
    print(
        f"scan={outcome.scan.scan_id} status={outcome_status.value} "
        f"sloty={outcome.scan.captured_slots}/{outcome.scan.occupied_slots}"
    )
    return 0 if outcome_status.value in {
        "captured", "queued", "provisional", "verified", "review"
    } else 1


def command_probe(args: argparse.Namespace) -> int:
    from PIL import ImageDraw

    settings = load_settings(args.config)
    screen = MSSScreen()
    input_backend = PyAutoGUIInput()
    window = GameWindow(settings.window_title)
    _countdown(window, input_backend)
    probe = ShopWindowProbe(screen, settings.grid)
    capturer = ShopCapturer(screen, input_backend, settings.grid)
    score = probe.score()
    is_open = score >= probe.minimum_grid_score
    grid = capturer.capture_grid()
    debug_dir = Path("dbg")
    debug_dir.mkdir(exist_ok=True)
    grid.save(debug_dir / "probe_grid.png")
    if not is_open:
        print(
            f"grid_score={score:.2f} open=False\n"
            "Nie potwierdzono siatki sklepu — nie wyliczam zajętości, "
            "bo wynik pochodziłby z przypadkowego fragmentu ekranu.\n"
            "Wycinek diagnostyczny: dbg/probe_grid.png"
        )
        return 1

    occupied = capturer.occupied_slots(grid)
    occupied_ids = {slot.slot for slot in occupied}
    overlay = grid.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    cell = settings.grid.cell
    for slot in range(settings.grid.rows * settings.grid.columns):
        row, column = divmod(slot, settings.grid.columns)
        x0, y0 = column * cell, row * cell
        color = (0, 255, 0) if slot in occupied_ids else (255, 40, 40)
        draw.rectangle(
            (x0 + 1, y0 + 1, x0 + cell - 2, y0 + cell - 2),
            outline=color,
            width=2,
        )
    overlay.resize(
        (overlay.width * 2, overlay.height * 2)
    ).save(debug_dir / "probe_overlay.png")
    print(
        f"grid_score={score:.2f} open=True "
        f"occupied={len(occupied)}/100"
    )
    print("sloty:", ", ".join(str(slot.slot) for slot in occupied))
    print("overlay: dbg/probe_overlay.png (zielony=zajęty, czerwony=pusty)")
    return 0


def command_reset_map(args: argparse.Namespace) -> int:
    moved = _reset_market_runtime_state(Path(args.market_dir), backup=not args.no_backup)
    if not moved:
        print("Mapa runtime już czysta — nie było czego usuwać.")
        return 0
    print("Wyczyszczono stan runtime mapy:")
    for source, target in moved:
        if target is None:
            print(f"  - usunięto {source}")
        else:
            print(f"  - {source} -> {target}")
    print("Granica farmy/rynku (farm_map.json) została nietknięta.")
    return 0


def command_coords(args: argparse.Namespace) -> int:
    """Sprawdź OCR pozycji bez ruchu, kliknięć ani otwierania sklepu."""

    from PIL import ImageDraw

    from scanner.analysis.coord_reader import (
        DEFAULT_ROI,
        STARTUP_FALLBACK_ATTEMPTS,
        TIGHT_ROI,
        accept_reading,
        read_image,
    )

    settings = load_settings(args.config)
    screen = MSSScreen()
    input_backend = PyAutoGUIInput()
    window = GameWindow(settings.window_title)
    _countdown(window, input_backend)

    rect = window.locate()
    image = screen.grab(rect.box)
    debug_dir = Path("dbg")
    debug_dir.mkdir(exist_ok=True)
    image.save(debug_dir / "coords_client.png")

    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    for name, roi, color in (
        ("TIGHT", TIGHT_ROI, (255, 60, 60)),
        ("DEFAULT", DEFAULT_ROI, (255, 215, 0)),
        ("FALLBACK", STARTUP_FALLBACK_ATTEMPTS[0][0], (80, 180, 255)),
    ):
        box = (
            int(image.width * roi[0]),
            int(image.height * roi[1]),
            int(image.width * roi[2]),
            int(image.height * roi[3]),
        )
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0] - 80, max(0, box[1] - 16)), name, fill=color)
    overlay.save(debug_dir / "coords_overlay.png")

    default_parsed = read_image(image)
    fallback_parsed = read_image(image, attempts=STARTUP_FALLBACK_ATTEMPTS)
    bounds = GameCapturePipeline._COORD_BOUNDS_FARM
    parsed = None
    source = None
    for label, candidate in (
        ("default", default_parsed),
        ("fallback", fallback_parsed),
    ):
        if candidate is None:
            continue
        if accept_reading(None, (candidate.x, candidate.y), bounds=bounds):
            parsed = candidate
            source = label
            break
    print(f"okno klienta={rect.box}; obraz={image.width}x{image.height}")
    print(
        "OCR default="
        + (
            f"({default_parsed.x}, {default_parsed.y})"
            if default_parsed is not None
            else "(brak)"
        )
    )
    print(
        "OCR fallback="
        + (
            f"({fallback_parsed.x}, {fallback_parsed.y})"
            if fallback_parsed is not None
            else "(brak)"
        )
    )
    if parsed is None:
        print(f"OCR wybrany=(brak), bounds={bounds}")
        print(
            "Zapisano dbg/coords_client.png i dbg/coords_overlay.png. "
            "Czerwony/żółty prostokąt musi obejmować napis (X, Y) pod minimapą."
        )
        return 1
    print(f"OCR wybrany=({parsed.x}, {parsed.y}) source={source}")
    print("Zapisano dbg/coords_client.png i dbg/coords_overlay.png.")
    return 0


def command_grid_calibrate(args: argparse.Namespace) -> int:
    """Zapisz podgląd kalibracji siatki i opcjonalnie nanieś wartości do configu."""

    config_path = Path(args.config)
    if config_path.exists():
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raw_config = {}
    settings = load_settings(config_path)
    raw_grid = raw_config.get("grid") or {}

    origin_raw = raw_config.get("shop_origin")
    if args.origin:
        ox, oy = (int(part) for part in args.origin.split(",", 1))
        origin = (ox, oy)
    elif origin_raw:
        origin = (int(origin_raw[0]), int(origin_raw[1]))
    else:
        origin = settings.grid.origin

    cols = int(raw_grid.get("cols", settings.grid.columns))
    rows = int(raw_grid.get("rows", settings.grid.rows))
    cell = int(args.cell if args.cell is not None else raw_grid.get("cell", settings.grid.cell))
    grid_dx = int(
        args.grid_dx if args.grid_dx is not None else raw_grid.get("grid_dx", settings.grid.offset[0])
    )
    grid_dy = int(
        args.grid_dy if args.grid_dy is not None else raw_grid.get("grid_dy", settings.grid.offset[1])
    )

    grid_x = origin[0] + grid_dx
    grid_y = origin[1] + grid_dy
    grid_w = cols * cell
    grid_h = rows * cell

    screen = MSSScreen()
    grid_image = screen.grab((grid_x, grid_y, grid_w, grid_h))
    overlay = render_grid_overlay(grid_image, cols, rows, cell)

    debug_dir = Path("dbg")
    debug_dir.mkdir(exist_ok=True)
    out_path = debug_dir / "grid_calibrate.png"
    overlay.save(out_path)

    print(
        f"grid-calibrate: origin={origin} offset=({grid_dx},{grid_dy}) "
        f"cell={cell} size={cols}x{rows}"
    )
    print(f"overlay -> {out_path}")
    print("Sprawdź: czerwone linie = ramki slotów, zielone kropki = środki ikon.")

    if args.save:
        updated = apply_calibration(
            raw_config,
            origin=origin,
            cell=cell,
            grid_dx=grid_dx,
            grid_dy=grid_dy,
        )
        config_path.write_text(
            json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Zapisano konfigurację -> {config_path}")
    else:
        print("Bez --save: config nie został zmieniony.")
    return 0


def _wait_calibration_key(key: str) -> str:
    try:
        import keyboard

        keyboard.wait(key)
        time.sleep(0.05)
        return key.upper()
    except Exception:
        input("Moduł keyboard/hotkey niedostępny. Ustaw kursor i naciśnij ENTER...")
        return "ENTER"


def _capture_calibration_point(
    prompt: str,
    input_backend: InputBackend,
    *,
    key: str,
) -> tuple[int, int]:
    print(f"\n{prompt}")
    print(f"  Ustaw kursor dokładnie w punkcie i naciśnij {key.upper()} (nie ruszaj myszką).")
    used_key = _wait_calibration_key(key)
    point = input_backend.position()
    print(f"  złapano {point} ({used_key})")
    return point


def command_window_calibrate(args: argparse.Namespace) -> int:
    """Interaktywna kalibracja okna sklepu hotkeyem, bez starego zapisu configu."""

    config_path = Path(args.config)
    if config_path.exists():
        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raw_config = {}
    raw_grid = raw_config.get("grid") or {}
    settings = load_settings(config_path)
    input_backend = PyAutoGUIInput()

    print("=== WINDOW-CALIBRATE ===")
    print("Otwórz JEDEN sklep offline, nie ruszaj okna sklepu w trakcie pomiaru.")
    print("Ta komenda zmienia tylko shop_origin oraz grid.cell/grid_dx/grid_dy.")

    origin = _capture_calibration_point(
        "1) Najedź na LEWY-GÓRNY róg okna sklepu (ramka 'Sklep Offline').",
        input_backend,
        key=args.key,
    )
    slot00 = _capture_calibration_point(
        "2) Najedź na ŚRODEK pierwszego slotu (kolumna 0, wiersz 0).",
        input_backend,
        key=args.key,
    )
    slot10 = _capture_calibration_point(
        "3) Najedź na ŚRODEK slotu obok (kolumna 1, wiersz 0).",
        input_backend,
        key=args.key,
    )

    try:
        calibration = calibration_from_points(
            origin,
            slot00,
            slot10,
            min_cell=args.min_cell,
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    cols = int(raw_grid.get("cols", settings.grid.columns))
    rows = int(raw_grid.get("rows", settings.grid.rows))
    grid_x = calibration.origin[0] + calibration.grid_dx
    grid_y = calibration.origin[1] + calibration.grid_dy
    grid_w = cols * calibration.cell
    grid_h = rows * calibration.cell

    screen = MSSScreen()
    grid_image = screen.grab((grid_x, grid_y, grid_w, grid_h))
    overlay = render_grid_overlay(grid_image, cols, rows, calibration.cell)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)

    print("\nWynik pomiaru:")
    print(
        f"  shop_origin={list(calibration.origin)}  "
        f"cell={calibration.cell}  "
        f"grid_dx={calibration.grid_dx}  grid_dy={calibration.grid_dy}"
    )
    print(f"  overlay -> {output_path}")
    print("  Sprawdź: czerwone linie = ramki slotów, zielone kropki = środki ikon.")

    if args.no_save:
        print("Bez zapisu (--no-save).")
        return 0

    backup_path = config_path.with_name(f"{config_path.stem}.bak{config_path.suffix}")
    if config_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"Backup configu -> {backup_path}")

    updated = apply_calibration(
        raw_config,
        origin=calibration.origin,
        cell=calibration.cell,
        grid_dx=calibration.grid_dx,
        grid_dy=calibration.grid_dy,
    )
    config_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Zapisano minimalną kalibrację -> {config_path}")
    print("Następnie odpal: python -m scanner probe")
    return 0


def command_hover_bench(args: argparse.Namespace) -> int:
    """Zmierz na żywo, jak często syntetyczny ruch wywołuje dymek."""

    settings = load_settings(args.config)
    pipeline, _, _, _, window, screen, _ = build_live_pipeline(
        settings,
        scans_dir=args.scans,
    )
    _countdown(window, pipeline.input)
    window_rect = window.locate()
    pipeline.tooltip_capturer.bounds = window_rect.box

    probe = ShopWindowProbe(screen, settings.grid)
    score = probe.score()
    if score < probe.minimum_grid_score:
        debug_dir = Path("dbg")
        debug_dir.mkdir(exist_ok=True)
        pipeline.shop_capturer.capture_grid().save(
            debug_dir / "hover_bench_failed_grid.png"
        )
        screen.grab(window_rect.box).save(
            debug_dir / "hover_bench_failed_client.png"
        )
        print(
            f"grid_score={score:.2f}, "
            f"wymagane={probe.minimum_grid_score:.2f}"
        )
        print("Diagnostyka: dbg/hover_bench_failed_grid.png i dbg/hover_bench_failed_client.png")
        print(
            "BŁĄD: nie wykryto otwartego okna sklepu. "
            "Otwórz sklep i uruchom ponownie."
        )
        return 1

    grid = pipeline.shop_capturer.capture_grid()
    occupied = pipeline.shop_capturer.occupied_slots(grid)
    if not occupied:
        print("BŁĄD: nie wykryto żadnego zajętego slotu.")
        return 1

    if args.slot is None:
        selected = occupied[0]
    else:
        selected = next(
            (slot for slot in occupied if slot.slot == args.slot), None
        )
        if selected is None:
            ids = ", ".join(str(slot.slot) for slot in occupied)
            print(
                f"BŁĄD: slot {args.slot} nie jest wykryty jako zajęty. "
                f"Zajęte: {ids}"
            )
            return 1

    center = settings.grid.slot_center(selected.column, selected.row)
    print(
        f"window={window_rect.box} origin={settings.grid.origin} "
        f"grid={settings.grid.box}\n"
        f"testowany slot={selected.slot} "
        f"(kol={selected.column}, wiersz={selected.row}) center={center}"
    )

    debug_dir = Path("dbg") / "hover_bench"
    debug_dir.mkdir(parents=True, exist_ok=True)
    hits = 0
    for attempt in range(1, args.attempts + 1):
        result = pipeline.tooltip_capturer.capture(selected)
        actual = pipeline.input.position()
        if result.frames:
            hits += 1
            for frame_index, frame in enumerate(result.frames, start=1):
                frame.save(
                    debug_dir
                    / f"attempt_{attempt:02d}_frame_{frame_index}.png"
                )
            label = f"HIT ({len(result.frames)} klatki)"
        else:
            label = "MISS"
        print(
            f"  próba {attempt:02d}/{args.attempts}: {label}; "
            f"kursor={actual}"
        )

    rate = 100.0 * hits / args.attempts
    print(
        f"WYNIK: {hits}/{args.attempts} = {rate:.1f}% wywołanych dymków\n"
        f"udane klatki: {debug_dir}"
    )
    return 0


def command_calibrate(args: argparse.Namespace) -> int:
    """Spacer kalibracyjny: N kroków w jedną oś, OCR co krok → steps.jsonl.

    Zero capture – tylko marsz WASD + odczyt współrzędnych świata.
    Wynik zapisywany jako JSONL do --output. Używane przez
    odometry.calibrate() do wyliczenia units_per_step.
    """
    import json

    settings = load_settings(args.config)
    from scanner.analysis.coord_reader import read_image

    direction = args.direction
    steps = args.steps
    hold = args.hold
    settle = args.settle

    screen = MSSScreen()
    input_backend = PyAutoGUIInput()
    window = GameWindow(settings.window_title)
    movement = MovementController(input_backend)

    _countdown(window, input_backend)

    output_path = Path(args.output)
    print(
        f"Spacer kalibracyjny: {steps} kroków, "
        f"kierunek={direction.upper()}, "
        f"hold={hold}s, settle={settle}s"
    )
    print(f"Wynik zapisywany do: {output_path}")

    results = []
    ocr_count = 0
    ocr_miss = 0
    window_box = window.locate().box

    for i in range(1, steps + 1):
        movement.execute(direction, hold, settle)
        # OCR współrzędnej z pełnego okna gry
        record = {
            "step": i,
            "direction": direction,
            "ocr": None,
            "source": "miss",
        }
        try:
            img = screen.grab(window_box)
            parsed = read_image(img)
            if parsed is not None:
                record["ocr"] = [parsed.x, parsed.y]
                record["source"] = "ocr"
                ocr_count += 1
            else:
                ocr_miss += 1
        except Exception:
            ocr_miss += 1

        results.append(record)
        print(f"  krok {i:3d}/{steps}: OCR={record['source']:4s}", end="")
        if record["ocr"]:
            print(f" ({record['ocr'][0]:.0f}, {record['ocr'][1]:.0f})", end="")
        print(flush=True)

    # Zapisz JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    rate = 100 * ocr_count / steps if steps else 0
    print(f"\nGotowe: {steps} kroków, OCR={ocr_count}/{steps} ({rate:.1f}%)")
    print(f"Plik: {output_path}")
    return 0


def command_trace_boundary(args: argparse.Namespace) -> int:
    """Obrys farmy: ręczny spacer po obwodzie, OCR ~20fps, SPACJA = waypoint.
    
    PERIMETER_WALK_SPEC: jeden obieg → dokładny wielokąt.
    Uśrednianie przez robust_point (per-oś mediana), spread<5u = stabilny punkt.
    Wynik → farm_map.json (FarmBoundary, source="perimeter_walk").
    """
    import json
    import threading
    from collections import deque

    settings = load_settings(args.config)
    from scanner.analysis.coord_reader import read_image
    from scanner.analysis.farm_boundary import robust_point, FarmBoundary

    screen = MSSScreen()
    input_backend = PyAutoGUIInput()
    window = GameWindow(settings.window_title)
    from scanner.navigation import MovementController
    movement = MovementController(input_backend)

    _countdown(window, input_backend)

    window_box = window.locate().box
    samples = args.samples
    spread_max = args.spread_max
    output_path = Path(args.output)

    # Bufor ostatnich N odczytów OCR (dla uśredniania waypointu)
    ocr_buffer: deque[tuple[float, float]] = deque(maxlen=samples)
    waypoints: list[tuple[float, float]] = []  # zebrane punkty obrysu

    # Stan OCR loop (biegnie w tle)
    ocr_running = threading.Event()
    ocr_running.set()
    ocr_lock = threading.Lock()
    latest_ocr: tuple[float, float] | None = None

    def ocr_loop():
        """Ciągły odczyt OCR co ~50ms (20fps)."""
        nonlocal latest_ocr
        while ocr_running.is_set():
            try:
                img = screen.grab(window_box)
                parsed = read_image(img)
                if parsed is not None:
                    pt = (parsed.x, parsed.y)
                    with ocr_lock:
                        ocr_buffer.append(pt)
                        latest_ocr = pt
            except Exception:
                pass
            ocr_running.wait(0.04)  # ~25 fps

    ocr_thread = threading.Thread(target=ocr_loop, daemon=True)
    ocr_thread.start()

    print()
    print("=" * 60)
    print("  TRACE BOUNDARY — obrys farmy")
    print("  Chodź WASD w grze po obwodzie farmy.")
    print("  [SPACJA] = zapisz waypoint (uśrednia ostatnie odczyty OCR)")
    print("  [ENTER]  = zakończ i zbuduj wielokąt → farm_map.json")
    print("  [ESC] w grze = anuluj")
    print(f"  Bufor: {samples} odczytów, spread max: {spread_max}u")
    print("=" * 60)

    def _read_key(virtual_key: int) -> bool:
        """Stan klawisza (GetAsyncKeyState), MSB=1 = wciśnięty."""
        import ctypes
        result = ctypes.windll.user32.GetAsyncKeyState(virtual_key)
        return (result & 0x8000) != 0

    VK_SPACE = 0x20
    VK_RETURN = 0x0D
    VK_ESCAPE = 0x1B

    space_was_down = False
    enter_was_down = False
    esc_was_down = False

    print("\nNasłuch klawiszy aktywny... (Ctrl+C = przerwij)\n")

    try:
        while True:
            # OCR status (live feedback)
            with ocr_lock:
                buf_len = len(ocr_buffer)
                current = latest_ocr

            if current:
                print(
                    f"\r  OCR: ({current[0]:.0f}, {current[1]:.0f}) | "
                    f"bufor: {buf_len}/{samples} | waypointy: {len(waypoints)}   ",
                    end="", flush=True,
                )

            # SPACJA = waypoint
            space_down = _read_key(VK_SPACE)
            if space_down and not space_was_down:
                with ocr_lock:
                    readings = list(ocr_buffer)
                est = robust_point(readings, reject_beyond=8.0)
                if est is not None:
                    # PILNE #1: capture-time gate – odrzuć waypointy poza kopertą farmy
                    X0, X1, Y0, Y1 = 348, 501, 672, 794
                    if not (X0 <= est.point[0] <= X1 and Y0 <= est.point[1] <= Y1):
                        print(f"\n  ✗ ODRZUCAM waypoint poza farmą ({est.point[0]:.1f}, {est.point[1]:.1f})")
                        ocr_buffer.clear()
                        space_was_down = space_down
                        continue
                    if est.spread > spread_max:
                        print(f"\n  ⚠ WAYPOINT: ({est.point[0]:.1f}, {est.point[1]:.1f}) "
                              f"spread={est.spread:.1f}u (>{spread_max}u – niestabilny!)")
                    else:
                        print(f"\n  ✓ WAYPOINT #{len(waypoints)+1}: ({est.point[0]:.1f}, {est.point[1]:.1f}) "
                              f"spread={est.spread:.1f}u, samples={est.samples}")
                    waypoints.append(est.point)
                    # Wyczyść bufor po waypoincie (unikaj duplikatów)
                    with ocr_lock:
                        ocr_buffer.clear()
                else:
                    print("\n  ✗ WAYPOINT: brak odczytów w buforze")
            space_was_down = space_down

            # ENTER = zakończ
            enter_down = _read_key(VK_RETURN)
            if enter_down and not enter_was_down:
                break
            enter_was_down = enter_down

            # ESC w grze = anuluj
            esc_down = _read_key(VK_ESCAPE)
            if esc_down and not esc_was_down and window.is_foreground():
                print("\n\n  ANULOWANO (ESC)")
                ocr_running.clear()
                ocr_thread.join(timeout=1.0)
                return 0
            esc_was_down = esc_down

            time.sleep(0.03)  # ~30Hz pętla klawiszy

    except KeyboardInterrupt:
        print("\n\n  ANULOWANO (Ctrl+C)")
        ocr_running.clear()
        ocr_thread.join(timeout=1.0)
        return 0

    ocr_running.clear()
    ocr_thread.join(timeout=2.0)

    print("\n")

    if len(waypoints) < 3:
        print(f"BŁĄD: za mało waypointów ({len(waypoints)}). Minimum 3 do wielokąta.")
        return 1

    # Zbuduj wielokąt z uporządkowanych waypointów
    boundary = FarmBoundary.from_perimeter(waypoints)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundary.save(output_path)

    print(f"✓ Wielokąt zapisany: {output_path}")
    print(f"  source: {boundary.source}")
    print(f"  wierzchołki: {len(boundary.polygon)}")
    print(f"  pole: {boundary.area():.0f} u²")
    bbox = boundary.bbox()
    print(f"  bbox: x∈[{bbox[0]:.0f},{bbox[1]:.0f}] y∈[{bbox[2]:.0f},{bbox[3]:.0f}]")
    print("  Gotowe — farm_map.json gotowy do użycia z coverage_map/should_turn_at_boundary.")

    return 0


def command_auto(args: argparse.Namespace) -> int:
    if not _analysis_ready(args.analyze):
        return 1
    if args.zone:
        # D4: --zone wycofany (COVERAGE_MAP_CORE). CoverageMap przejmuje
        # wszystkie odpowiedzialności Zone (pokrycie, nasycenie, confinement).
        print(
            "DEPRECATED: --zone wycofany. Użyj --odometry zamiast."
        )
        return 1
    if args.coverage_drive and not args.odometry:
        print("BŁĄD: --coverage-drive wymaga --odometry.")
        return 1

    if args.fresh_map:
        moved = _reset_market_runtime_state()
        if moved:
            print("Fresh map: wyczyszczono runtime mapy przed startem:")
            for source, target in moved:
                print(f"  - {source} -> {target}")
        else:
            print("Fresh map: runtime mapy już był czysty.")

    settings = load_settings(args.config)
    pipeline, detector, tracker, movement, window, screen, worker = build_live_pipeline(
        settings,
        scans_dir=args.scans,
        analyze=args.analyze,
        csv_path=args.csv,
        use_ocr=not args.no_ocr,
        units_per_step=3.2 if args.odometry else 7.0,
        odometry_vectors={"w": (-2.59, -1.87), "s": (2.59, 1.87)} if args.odometry else None,
        vlm_shop_audit=getattr(args, "vlm_shop_audit", False),
    )
    _countdown(window, pipeline.input)
    if args.coverage_drive:
        print("Coverage drive: odczytuję współrzędne startowe (maks. 6 s)...")
        initial_position = _await_initial_position(pipeline)
        if initial_position is None:
            print(
                "BŁĄD: --coverage-drive wymaga odczytu współrzędnych (X,Y) "
                "przed startem. Widok współrzędnych jest obecny, ale OCR nie "
                "trafił go w 6 s — uruchom `python -m scanner coords` i "
                "sprawdź dbg/coords_client.png."
            )
            return 1
        print(f"Coverage drive: start OCR={initial_position}")
    diagnostics = None
    if args.debug_live:
        from scanner.diagnostics import AutoDiagnostics

        diagnostics = AutoDiagnostics(
            args.debug_dir,
            max_image_rounds=args.debug_max_images,
            images_every=args.debug_images_every,
        )
        pipeline.interactor.observer = diagnostics.record_interaction
        print(f"Diagnostyka live -> {diagnostics.directory}")

    def view_provider():
        rect = window.locate()
        return screen.grab(rect.box), (rect.x, rect.y)

    pipeline.max_popup_topups_per_shop = args.popup_budget
    if args.phase_b:
        pipeline.enable_phase_b(quorum=2, big_quorum=3)
        print(
            "Phase B: hover reprezentantow stosu "
            "(quorum=2, big_quorum=3)"
        )
    print(f"Popup budget/sklep: {pipeline.max_popup_topups_per_shop}")

    loop = AutonomousMarketLoop(
        detector, tracker, pipeline, movement, view_provider, diagnostics
    )
    if args.odometry and not args.zone:
        # Gęste stemplowanie + coverage map (Filar 2)
        from scanner.analysis.shop_registry import ShopRegistry
        from scanner.analysis.coverage_map import CoverageMap
        from scanner.analysis.farm_boundary import FarmBoundary
        from scanner.analysis.movement_memory import MovementMemory

        _stamp_registry = ShopRegistry.open(Path("market_map"), partition="glevia_market")
        _farm_boundary_path = Path("market_map/glevia_market/farm_map.json")
        _farm_boundary = FarmBoundary.load(_farm_boundary_path)
        _coverage_path = Path("market_map/glevia_market/coverage.json")
        _movement_memory_path = Path("market_map/glevia_market/movement_memory.json")
        _coverage_cell_size = float(args.coverage_cell_size)
        _covmap = CoverageMap.load(
            _coverage_path,
            boundary=_farm_boundary.polygon if _farm_boundary else None,
        )
        if _covmap is not None and abs(float(_covmap.cell_size) - _coverage_cell_size) > 1e-6:
            backup_suffix = time.strftime(".cellsize-%Y%m%d_%H%M%S.bak")
            old_cell_size = float(_covmap.cell_size)
            for _path in (_coverage_path, _movement_memory_path):
                if _path.exists():
                    _path.rename(_path.with_name(_path.name + backup_suffix))
            print(
                "Coverage cell_size zmieniony "
                f"{old_cell_size:g} -> {_coverage_cell_size:g}; "
                "startuje swieza coverage mapa i pamiec ruchu."
            )
            _covmap = None
        _covmap = _covmap or CoverageMap(
            (348, 501, 672, 794),
            cell_size=_coverage_cell_size,
            boundary=_farm_boundary.polygon if _farm_boundary else None,
        )
        _movement_memory = MovementMemory.load(_movement_memory_path) or MovementMemory()

        def _odometry_hook(scan):
            nonlocal _farm_boundary, pipeline
            fp = scan.shop_fingerprint
            gp = scan.game_position
            if fp is not None and gp is not None:
                _stamp_registry.ingest({
                    "scan_id": scan.scan_id,
                    "shop_fingerprint": fp,
                    "game_position": list(gp) if gp else None,
                    "game_position_source": pipeline._position_source,
                    "seller": scan.seller,
                    "created_at": scan.created_at,
                })
                _stamp_registry.save()
                _covmap.mark(gp, radius_cells=0)
                # PILNE #2: auto-grow granicy – jeśli sklep poza granicą, poszerz ją
                if args.auto_grow_boundary and _farm_boundary is not None:
                    _farm_boundary, grew = _farm_boundary.grown_to_include(
                        gp, bounds=(348, 501, 672, 794), max_jump=40.0, margin=3.0,
                    )
                    if grew:
                        _farm_boundary.save(_farm_boundary_path)
                        print(f"  [auto-grow] granica poszerzona – {len(_farm_boundary.polygon)} wierzchołków")
                print(f"  [stamp+coverage] {_covmap.covered_count}/{_covmap.farm_cells} farm cells ({_covmap.coverage_fraction():.0%})")

        loop.set_shop_captured_hook(_odometry_hook)
        loop.set_coverage_map(_covmap)
        loop.set_farm_boundary(_farm_boundary)
        loop.set_movement_memory(_movement_memory)
        pipeline.set_known_fresh(_stamp_registry.is_known_fresh)
        print("Dense stamp + Coverage map ON – Filar 2 aktywny (guard włączony)")

    if args.zone:
        def _zone_hook(scan):
            nonlocal registry, zone_map_obj, zone_nav, _zone_farm_boundary
            fp = scan.shop_fingerprint
            if fp is None:
                return
            is_new = registry.by_fingerprint(fp) is None
            registry.ingest({
                "scan_id": scan.scan_id,
                "shop_fingerprint": fp,
                "game_position": list(scan.game_position) if scan.game_position else None,
                "seller": scan.seller,
                "created_at": scan.created_at,
            })
            registry.save()
            gp = scan.game_position
            if gp is not None:
                zone_nav.stamp_position(gp)
                zid = zone_nav.current_zone_id
                if zid is not None:
                    zone_nav.record_shop(zid, is_new_fingerprint=is_new)
                zone_map_obj.save()
                # PILNE #2: auto-grow granicy w trybie --zone
                if _zone_farm_boundary is not None:
                    _zone_farm_boundary, grew = _zone_farm_boundary.grown_to_include(
                        gp, bounds=(348, 501, 672, 794), max_jump=40.0, margin=3.0,
                    )
                    if grew:
                        _zone_farm_boundary.save(_zone_farm_boundary_path)
                        print(f"  [auto-grow] granica poszerzona – {len(_zone_farm_boundary.polygon)} wierzchołków")
        loop.set_shop_captured_hook(_zone_hook)
    if args.zone:
        from scanner.analysis.shop_registry import ShopRegistry
        from scanner.analysis.zone_map import ZoneMap
        from scanner.navigation.map_navigator import MapSynchronizedNavigator
        from scanner.analysis.farm_boundary import FarmBoundary

        FARM_ENVELOPE = (348, 672, 501, 794)
        market_dir = Path("market_map/glevia_market")
        market_dir.mkdir(parents=True, exist_ok=True)

        _zone_farm_boundary_path = market_dir / "farm_map.json"
        _zone_farm_boundary = FarmBoundary.load(_zone_farm_boundary_path)
        loop.set_farm_boundary(_zone_farm_boundary)

        registry = ShopRegistry.open(Path("market_map"), partition="glevia_market")
        if (market_dir / "zones.json").exists():
            zone_map_obj = ZoneMap.load(market_dir)
        else:
            zone_map_obj = ZoneMap(FARM_ENVELOPE, directory=market_dir)
        zone_nav = MapSynchronizedNavigator(zone_map_obj, registry)
        pipeline.set_known_fresh(registry.is_known_fresh)

        walk = None
        route = ()
        print(
            f"Tryb strefowy: koperta={FARM_ENVELOPE}, "
            f"siatka={zone_map_obj.grid}, K={zone_map_obj.saturation_k}"
        )
    elif args.coverage_drive:
        walk = None
        route = ()
        print("Trasa: coverage-drive (skan bieżącej strefy → osiągalna niepokryta komórka)")
    elif args.walk:
        walk = settings_from_walk_config(
            args.config,
            lanes=args.lanes,
            steps_per_lane=args.steps_per_lane,
            step_duration=args.step_hold,
            settle=args.settle,
        )
        route = walk.steps()
        print(
            f"Trasa: lanes={walk.lanes}, "
            f"steps_per_lane={walk.steps_per_lane}, "
            f"kroki={len(route)}, step_hold={walk.step_duration:.2f}s, "
            f"settle={walk.settle:.2f}s"
        )
    else:
        walk = None
        route = ()
    if diagnostics is not None:
        diagnostics.event(
            "session_start",
            walk=args.walk,
            coverage_drive=args.coverage_drive,
            max_shops=args.max_shops,
            route_steps=len(route),
            route_lanes=walk.lanes if walk else 0,
            route_steps_per_lane=walk.steps_per_lane if walk else 0,
            route_step_hold=walk.step_duration if walk else 0.0,
            route_settle=walk.settle if walk else 0.0,
            analyze=args.analyze,
        )
    try:
        outcomes = []
        coverage_passes = args.coverage_passes if args.coverage_drive else 1
        for pass_index in range(coverage_passes):
            if pass_index:
                loop.reset_coverage_pass_state()
                print(
                    f"Coverage pass {pass_index + 1}/{coverage_passes}: "
                    "wznawiam po poprzednim zatrzymaniu."
                )
                if diagnostics is not None:
                    diagnostics.event(
                        "coverage_pass_restart",
                        pass_index=pass_index + 1,
                        max_passes=coverage_passes,
                    )
            successful_before = sum(outcome.successful for outcome in outcomes)
            remaining = (
                max(0, args.max_shops - successful_before)
                if args.max_shops
                else 0
            )
            outcomes.extend(
                loop.run(
                    route,
                    max_shops=remaining,
                    navigator=zone_nav if args.zone else None,
                )
            )
            successful_now = sum(outcome.successful for outcome in outcomes)
            if args.max_shops and successful_now >= args.max_shops:
                break
            if not args.coverage_drive:
                break
            if getattr(loop, "_coverage_done", False):
                break
            if pipeline.close_blocked:
                break
            if pass_index + 1 >= coverage_passes:
                break
            reason = getattr(loop, "_coverage_stop_reason", None) or "coverage_stopped"
            print(
                f"Coverage pass {pass_index + 1}/{coverage_passes}: "
                f"powod={reason}; robię kolejny pass."
            )
        if worker is not None:
            print("Capture zakończony — czekam na analizę pozostałych sklepów...")
            worker.join()
            worker.stop(5.0)
        successful = sum(
            outcome.scan.status.value in {"captured", "queued"}
            for outcome in outcomes
        )
        duplicates = sum(outcome.duplicate for outcome in outcomes)
        if args.max_shops and successful >= args.max_shops:
            end_reason = "max_shops_reached"
        elif pipeline.close_blocked:
            end_reason = "shop_close_blocked"
        elif args.coverage_drive:
            end_reason = getattr(loop, "_coverage_stop_reason", None) or (
                "coverage_done"
                if getattr(loop, "_coverage_done", False)
                else "coverage_stopped"
            )
        elif args.walk:
            end_reason = "route_exhausted"
        else:
            end_reason = "view_exhausted"
        print(
            f"Zakończono: próby={len(outcomes)}, poprawne={successful}, "
            f"duplikaty={duplicates}, powod={end_reason}"
        )
        if diagnostics is not None:
            diagnostics.event(
                "session_end",
                attempts=len(outcomes),
                successful=successful,
                duplicates=duplicates,
                end_reason=end_reason,
            )
    except BaseException as exc:
        if diagnostics is not None:
            diagnostics.event(
                "session_crash",
                error_type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
        raise
    finally:
        # FIX finally-save: persystencja ZAWSZE (też po Ctrl+C/crash)
        if '_covmap' in dir() and '_coverage_path' in dir():
            for c in _covmap.border_adjustments(fail_floor=3):
                _covmap.mark_no_go(c)
            _covmap.save(_coverage_path)
            no_go_count = len(_covmap._no_go)
            print(f"Coverage map zapisana: {_coverage_path} ({_covmap.covered_count}/{_covmap.farm_cells} cells, {no_go_count} no_go)")
        if '_movement_memory' in dir() and '_movement_memory_path' in dir():
            _movement_memory.save(_movement_memory_path)
            print(f"Movement memory zapisana: {_movement_memory_path}")
        if '_stamp_registry' in dir():
            _stamp_registry.reaggregate()
            _stamp_registry.save()
            print("Shop registry: idealne lokacje przeliczone + zapisane")
    return 0


def settings_from_walk_config(
    path: str | Path,
    *,
    lanes: int | None = None,
    steps_per_lane: int | None = None,
    step_duration: float | None = None,
    settle: float | None = None,
) -> SerpentineRoutePlanner:
    import json

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    walk = raw.get("walk") or {}
    return SerpentineRoutePlanner(
        left_key=str(walk.get("key_left", "a")),
        right_key=str(walk.get("key_right", "d")),
        lane_key=str(walk.get("drop_key", "s")),
        step_duration=(
            float(step_duration)
            if step_duration is not None
            else float(walk.get("step_hold", 0.6))
        ),
        steps_per_lane=(
            int(steps_per_lane)
            if steps_per_lane is not None
            else int(walk.get("steps_per_lane", 4))
        ),
        lanes=int(lanes) if lanes is not None else int(walk.get("lanes", 3)),
        settle=(
            float(settle)
            if settle is not None
            else float(walk.get("settle", 0.9))
        ),
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("wartosc musi byc >= 1")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("wartosc nie moze byc ujemna")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("wartosc nie moze byc ujemna")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("wartosc musi byc > 0")
    return parsed


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m scanner")
    root.add_argument("--config", default="scanner_config.json")
    root.add_argument("--scans", default="scans")
    commands = root.add_subparsers(dest="command", required=True)

    probe = commands.add_parser(
        "probe", help="sprawdź kalibrację siatki i detekcję otwartego sklepu"
    )
    probe.set_defaults(handler=command_probe)

    coords = commands.add_parser(
        "coords",
        help="sprawdź odczyt współrzędnych (X,Y) bez ruchu i zapisz overlay ROI",
    )
    coords.set_defaults(handler=command_coords)

    window_calibrate = commands.add_parser(
        "window-calibrate",
        help="interaktywna kalibracja okna sklepu hotkeyem F8",
    )
    window_calibrate.add_argument(
        "--key",
        default="f8",
        help="hotkey do złapania punktu; domyślnie f8",
    )
    window_calibrate.add_argument(
        "--output",
        default="dbg/window_calibrate.png",
        help="ścieżka overlaya weryfikacyjnego",
    )
    window_calibrate.add_argument(
        "--min-cell",
        type=_positive_int,
        default=8,
        help="minimalna akceptowana odległość między środkami slotów",
    )
    window_calibrate.add_argument(
        "--no-save",
        action="store_true",
        help="tylko pokaż wynik i overlay, nie zapisuj configu",
    )
    window_calibrate.set_defaults(handler=command_window_calibrate)

    grid_calibrate = commands.add_parser(
        "grid-calibrate",
        help="zapisz overlay kalibracji siatki sklepu i opcjonalnie popraw config",
    )
    grid_calibrate.add_argument(
        "--origin",
        default=None,
        help="shop_origin jako x,y; domyślnie z configu",
    )
    grid_calibrate.add_argument("--cell", type=_positive_int, default=None)
    grid_calibrate.add_argument("--grid-dx", type=int, default=None)
    grid_calibrate.add_argument("--grid-dy", type=int, default=None)
    grid_calibrate.add_argument(
        "--save",
        action="store_true",
        help="zapisz podane wartości do scanner_config.json",
    )
    grid_calibrate.set_defaults(handler=command_grid_calibrate)

    hover_bench = commands.add_parser(
        "hover-bench",
        help="zmierz niezawodność hovera na jednym zajętym slocie",
    )
    hover_bench.add_argument(
        "--slot",
        type=int,
        default=None,
        help="numer slotu 0-99; domyślnie pierwszy zajęty",
    )
    hover_bench.add_argument(
        "--attempts",
        type=int,
        default=10,
        choices=range(1, 101),
        metavar="1-100",
    )
    hover_bench.set_defaults(handler=command_hover_bench)

    calibrate = commands.add_parser(
        "calibrate",
        help="spacer kalibracyjny: N kroków W (translacja), OCR co krok → steps.jsonl",
        epilog="UWAGA: A/D to SKRĘT (obrót), nie translacja. "
               "Do kalibracji units/step używaj W (przód) lub S (tył). "
               "Klik myszy NIGDY do chodzenia – tylko do otwierania sklepu.",
    )
    calibrate.add_argument("--direction", default="w", choices=["w", "a", "s", "d"])
    calibrate.add_argument("--steps", type=_positive_int, default=30)
    calibrate.add_argument("--hold", type=_non_negative_float, default=0.6)
    calibrate.add_argument("--settle", type=_non_negative_float, default=1.5)
    calibrate.add_argument(
        "--output", default="steps.jsonl", help="ścieżka pliku wyjściowego"
    )
    calibrate.set_defaults(handler=command_calibrate)

    trace = commands.add_parser(
        "trace-boundary",
        help="obrys farmy: chodź po obwodzie, OCR co ~50ms, [SPACJA] = waypoint → farm_map.json",
        epilog="Sterowanie: chodź WASD w grze ręcznie. "
               "SPACJA = zapisz waypoint (uśrednia ostatnie ~20 odczytów OCR). "
               "ENTER = zakończ i zapisz wielokąt. "
               "ESC w grze = anuluj.",
    )
    trace.add_argument("--output", default="market_map/glevia_market/farm_map.json",
                       help="ścieżka docelowa farm_map.json")
    trace.add_argument("--samples", type=_positive_int, default=20,
                       help="ile odczytów OCR uśrednić na waypoint")
    trace.add_argument("--spread-max", type=_non_negative_float, default=5.0,
                       help="maks. spread waypointu (u); >próg → ostrzeżenie o niestabilności")
    trace.set_defaults(handler=command_trace_boundary)

    capture = commands.add_parser(
        "capture-open", help="przechwyć ręcznie otwarty sklep"
    )
    _add_analysis_arguments(capture)
    capture.set_defaults(handler=command_capture_open)

    reset_map = commands.add_parser(
        "reset-map",
        help="wyczyść runtime mapy: coverage/movement/pozycje sklepów; zostaw farm_map.json",
    )
    reset_map.add_argument(
        "--market-dir",
        default=str(MARKET_PARTITION_DIR),
        help="katalog partycji mapy rynku",
    )
    reset_map.add_argument(
        "--no-backup",
        action="store_true",
        help="usuń pliki zamiast przenosić je do reset_backups/<timestamp>",
    )
    reset_map.set_defaults(handler=command_reset_map)

    auto = commands.add_parser("auto", help="autonomiczna pętla rynku")
    auto.add_argument("--walk", action="store_true", help="użyj trasy wężykiem")
    auto.add_argument(
        "--coverage-drive",
        action="store_true",
        help="skanuj bieżącą strefę, potem jedź do osiągalnej niepokrytej komórki; wymaga --odometry",
    )
    auto.add_argument("--max-shops", type=int, default=0)
    auto.add_argument(
        "--fresh-map",
        action="store_true",
        help="przed startem wyczyść coverage/movement/pozycje sklepów; farm_map.json zostaje",
    )
    auto.add_argument(
        "--coverage-passes",
        type=_positive_int,
        default=3,
        help="ile razy coverage-drive ma sam wznowić bieg po coverage_stopped/stuck",
    )
    auto.add_argument(
        "--coverage-cell-size",
        type=_positive_float,
        default=30.0,
        help="rozmiar komorki coverage w jednostkach gry; 30 ogranicza nakladanie postojow",
    )
    auto.add_argument(
        "--popup-budget",
        type=_non_negative_int,
        default=40,
        help="maks. liczba prob PPM-popup na sklep; 0 wylacza popup recovery/topup",
    )
    auto.add_argument("--lanes", type=_positive_int, default=None)
    auto.add_argument("--steps-per-lane", type=_positive_int, default=None)
    auto.add_argument("--step-hold", type=_non_negative_float, default=None)
    auto.add_argument("--settle", type=_non_negative_float, default=None)
    auto.add_argument(
        "--debug-live",
        action="store_true",
        help="zapisz scenę, maskę, overlay klików i events.jsonl",
    )
    auto.add_argument(
        "--debug-dir",
        default="dbg/auto",
        help="katalog bazowy diagnostyki --debug-live",
    )
    auto.add_argument(
        "--debug-max-images",
        type=int,
        default=60,
        help="maksymalna liczba rund z PNG przy --debug-live; 0 = bez limitu",
    )
    auto.add_argument(
        "--debug-images-every",
        type=_positive_int,
        default=1,
        help="zapisuj PNG co N-tą rundę diagnostyki; events.jsonl zawsze pełny",
    )
    auto.add_argument(
        "--zone",
        action="store_true",
        help="(EKSPERYMENTALNE, za bramką P0b) tryb strefowy z mapą rynku",
    )
    auto.add_argument(
        "--phase-b",
        action="store_true",
        help="Phase B: hover tylko reprezentantów stosu (floor=3)",
    )
    auto.add_argument(
        "--odometry",
        action="store_true",
        help="użyj skalibrowanej odometrii (3.2 u/krok) zamiast domyślnego 7.0",
    )

    auto.add_argument(
        "--auto-grow-boundary",
        action="store_true",
        help="eksperymentalnie rozszerzaj farm_map.json, gdy sklep jest poza granicą",
    )

    _add_analysis_arguments(auto)
    auto.set_defaults(handler=command_auto)
    return root


def _add_analysis_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--analyze",
        action="store_true",
        help="analizuj zapisane dymki równolegle przez Ollamę",
    )
    command.add_argument(
        "--no-ocr",
        action="store_true",
        help="wyłącz niezależne potwierdzenie Windows OCR",
    )
    command.add_argument(
        "--csv",
        default="ceny.csv",
        help="plik docelowy dla rekordów VERIFIED",
    )
    command.add_argument(
        "--vlm-shop-audit",
        action="store_true",
        help="EKSPERYMENTALNE: dorzuć surowy odczyt VLM całego shop.png jako "
             "diagnostykę (nazwy ikon, vlm_only). Domyślny audyt kompletności jest "
             "deterministyczny (occupied/unassigned) i NIE wymaga VLM",
    )


def main(argv: list[str] | None = None) -> int:
    disable_console_quick_edit()
    args = parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except RuntimeError as exc:
        print(f"BŁĄD: {exc}")
        return 1
