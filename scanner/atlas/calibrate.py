"""Orkiestracja kalibracji Atlasu: obserwacje ruchu -> fit -> walidacja -> zapis.

Warstwa offline (`fit_calibration`, `fit_and_save`) nie steruje postacią. CLI na
końcu pliku jest pasem live Codexa i uruchamia się tylko przy wykonaniu modułu:
`python -m scanner.atlas.calibrate`. Sam import nigdy nie naciska klawiszy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from scanner.atlas.calibration import (
    FitResult,
    GroundProjection,
    check_direction_coherence,
)
from scanner.atlas.config import ATLAS_CONFIG_PATH, AtlasConfig
from scanner.atlas.contracts import FrameSnapshot, MoveObservation


_OPPOSITE_KEY = {"w": "s", "s": "w", "a": "d", "d": "a"}


@dataclass(slots=True)
class CalibrationOutcome:
    """Werdykt kalibracji do pokazania operatorowi i warunkowego zapisu."""

    ok: bool
    saved: bool
    message: str
    projection: GroundProjection | None = None
    result: FitResult | None = None


def make_version(prefix: str = "v1") -> str:
    """Unikalna etykieta transformu, trafia do `AtlasShop.transform_version`."""

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{ts}"


def fit_calibration(
    moves: list[MoveObservation],
    *,
    anchor: tuple[float, float] = (960.0, 540.0),
    version: str | None = None,
    max_residual_px: float = 6.0,
    min_delta_game_units: float = 2.5,
) -> CalibrationOutcome:
    """Dopasuj i zwaliduj kalibrację. Nie zapisuje — patrz `fit_and_save`."""

    version = version or make_version()
    if len(moves) < 2:
        return CalibrationOutcome(
            False,
            False,
            "za mało kroków kalibracyjnych (potrzeba >=2 kierunki)",
        )
    # Zablokowane o stragany kroki (delta_game ~0) NIE są błędem całego runu — odrzuć
    # je pojedynczo i licz fit z reszty. Zostawienie ich truje fit (Δscreen != 0 przy
    # Δgame = 0 spycha A ku osobliwości).
    def _mag(m: MoveObservation) -> float:
        return float(m.delta_game[0] ** 2 + m.delta_game[1] ** 2) ** 0.5

    usable = [m for m in moves if _mag(m) >= min_delta_game_units]
    dropped = len(moves) - len(usable)
    if len(usable) < 2:
        return CalibrationOutcome(
            False,
            False,
            f"za mało ruchów o wystarczającym dystansie: {len(usable)}/{len(moves)} "
            f">= {min_delta_game_units:.1f}u (reszta zablokowana o stragany). "
            "Zwiększ --hold albo skalibruj w luźniejszym miejscu.",
        )
    try:
        proj, res = GroundProjection.fit_from_moves(
            usable,
            anchor=anchor,
            version=version,
        )
    except ValueError as exc:
        return CalibrationOutcome(False, False, f"kalibracja odrzucona: {exc}")

    total_points = sum(len(m.delta_screen) for m in usable)
    weak_points = sum(
        len(m.delta_screen)
        for m in usable
        if len(m.delta_screen) <= 1 and float(m.confidence) < 0.5
    )
    strong_moves = sum(
        1
        for m in usable
        if len(m.delta_screen) >= 2 or float(m.confidence) >= 0.5
    )
    weak_fraction = weak_points / max(1, total_points)
    enough_consensus = strong_moves >= 2 and weak_fraction <= 0.5
    direction_problems = check_direction_coherence(usable)
    drop_note = f", odrzucono {dropped} zablokowanych" if dropped else ""
    ok = (
        res.inliers >= 4
        and res.residual_px <= max_residual_px
        and res.condition < 1e3
        and enough_consensus
        and not res.opposite_problems
        and not direction_problems
    )
    sx, sy = proj.scale_px_per_unit()
    if ok:
        message = (
            f"kalibracja OK: residual={res.residual_px:.2f}px, "
            f"inliers={res.inliers}/{res.n_points}, "
            f"skala~{sx:.1f}/{sy:.1f}px/u, condition={res.condition:.1f}{drop_note}"
        )
    else:
        quality_hint = ""
        if not enough_consensus:
            quality_hint = (
                f", 1-track={weak_points}/{total_points}, "
                f"strong_moves={strong_moves} - za duzo slabych trackow"
            )
        if res.opposite_problems:
            quality_hint += " | " + " ; ".join(res.opposite_problems)
        if direction_problems:
            quality_hint += " | " + " ; ".join(direction_problems)
        message = (
            f"kalibracja SŁABA: residual={res.residual_px:.2f}px "
            f"(próg {max_residual_px}), inliers={res.inliers}/{res.n_points}, "
            f"condition={res.condition:.1f}{drop_note} — powtórz w otwartym miejscu, "
            ">=2 wyraźnie różne kierunki, więcej widocznych sklepów"
        )
    if not ok and quality_hint:
        message += quality_hint
    return CalibrationOutcome(ok, False, message, projection=proj, result=res)


def fit_and_save(
    moves: list[MoveObservation],
    config: AtlasConfig,
    *,
    version: str | None = None,
    max_residual_px: float | None = None,
) -> CalibrationOutcome:
    """Dopasuj, zwaliduj i tylko gdy jakość OK zapisz projekcję."""

    outcome = fit_calibration(
        moves,
        anchor=config.anchor,
        version=version,
        max_residual_px=(
            config.max_calib_residual_px
            if max_residual_px is None
            else float(max_residual_px)
        ),
    )
    if outcome.ok and outcome.projection is not None:
        outcome.projection.save(config.calibration_file)
        outcome.saved = True
        outcome.message += f"  -> zapisano: {config.calibration_file}"
    return outcome


class CalibrationAbort(RuntimeError):
    """Kontrolowane przerwanie live-kalibracji przed następnym ruchem."""


class AtlasRunLock:
    """Prosty lockfile dla trybów Atlasa."""

    def __init__(self, market_dir: str | Path, *, force: bool = False) -> None:
        self.path = Path(market_dir) / ".atlas_lock"
        self.force = force

    def __enter__(self) -> "AtlasRunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if not self.force and self._lock_process_alive():
                raise CalibrationAbort(
                    f"aktywny lock Atlasa: {self.path}. "
                    "Jeśli to stary plik po crashu, uruchom z --force-lock."
                )
            try:
                self.path.unlink()
            except OSError as exc:
                raise CalibrationAbort(
                    f"nie mogę usunąć starego locka {self.path}: {exc}"
                ) from exc
        payload = {
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "owner": "scanner.atlas.calibrate",
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError:
            pass

    def _lock_process_alive(self) -> bool:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        if os.name == "nt":
            try:
                import ctypes

                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
                return False
            except Exception:
                return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scanner.atlas.calibrate",
        description="Bezpieczna kalibracja live Atlasa. Domyślnie dry-run bez ruchu.",
    )
    parser.add_argument("--config", default=ATLAS_CONFIG_PATH, help="atlas_config.json")
    parser.add_argument("--scanner-config", default="scanner_config.json")
    parser.add_argument("--arm", action="store_true", help="ZEZWÓL na kontrolowany ruch postaci")
    parser.add_argument("--force-lock", action="store_true", help="usuń stary .atlas_lock po crashu")
    parser.add_argument("--keys", help="nadpisz kierunki, np. w,d,a")
    parser.add_argument("--steps-per-key", type=int, help="nadpisz liczbę kroków na kierunek")
    parser.add_argument("--hold", type=float, help="ile sekund trzymać klawisz")
    parser.add_argument("--settle", type=float, default=0.35, help="pauza po kroku")
    parser.add_argument("--after-timeout", type=float, default=4.0, help="ile sekund czekaÄ‡ na stabilny OCR po ruchu")
    parser.add_argument("--after-poll", type=float, default=0.25, help="interwaĹ‚ ponawiania klatki po ruchu")
    parser.add_argument("--after-stable-px", type=float, default=6.0, help="maks. mediana ruchu sceny miÄ™dzy klatkami after")
    parser.add_argument("--max-residual", type=float, help="tymczasowy prĂłg akceptacji residual px dla zapisu kalibracji")
    parser.add_argument("--min-shop-tracks", type=int, default=2)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--preflight-timeout", type=float, default=8.0)
    parser.add_argument("--debug-dir", default="dbg", help="gdzie zapisać debug coordów")
    parser.add_argument(
        "--skip-boundary",
        action="store_true",
        help="nie wymagaj pozycji wewnątrz farm_map.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = AtlasConfig.load(args.config)
        keys = _parse_keys(args.keys, config.calib_keys)
        steps = max(1, int(args.steps_per_key or config.calib_steps_per_key))
        hold = max(0.01, float(args.hold or config.calib_step_hold_s))
        feed, input_backend, window = _build_live_dependencies(
            args.scanner_config,
            config,
        )
        _ensure_foreground_for_snapshot(
            window,
            timeout_s=max(5.0, float(args.countdown)),
        )
        snapshot = feed.capture_once()
        ready = _print_snapshot_readiness(
            snapshot,
            min_shop_tracks=max(1, int(args.min_shop_tracks)),
            boundary_file=None if args.skip_boundary else config.boundary_file,
        )
        if snapshot.player_game is None:
            path = _save_coord_debug(feed, Path(args.debug_dir))
            print(f"debug coordów -> {path}")
        if not args.arm:
            print("DRY-RUN: zero ruchu. Dodaj --arm, żeby wykonać kalibrację.")
            return 0 if ready else 2

        with AtlasRunLock(config.market_dir, force=args.force_lock):
            moves = _collect_live_moves(
                feed=feed,
                input_backend=input_backend,
                window=window,
                config=config,
                keys=keys,
                steps_per_key=steps,
                hold_s=hold,
                settle_s=max(0.0, float(args.settle)),
                after_timeout_s=max(0.0, float(args.after_timeout)),
                after_poll_s=max(0.05, float(args.after_poll)),
                after_stable_px=max(0.5, float(args.after_stable_px)),
                min_shop_tracks=max(1, int(args.min_shop_tracks)),
                countdown_s=max(0.0, float(args.countdown)),
                preflight_timeout_s=max(0.0, float(args.preflight_timeout)),
                debug_dir=Path(args.debug_dir),
                require_boundary=not args.skip_boundary,
            )
            moves_path = _save_moves_debug(moves, Path(args.debug_dir))
            print(f"debug ruchu -> {moves_path}")
            outcome = fit_and_save(
                moves,
                config,
                max_residual_px=args.max_residual,
            )
            print(outcome.message)
            return 0 if outcome.ok else 3
    except KeyboardInterrupt:
        print("Przerwano przez użytkownika — kalibracja zatrzymana.")
        return 130
    except CalibrationAbort as exc:
        print(f"BŁĄD: {exc}")
        return 2


def _build_live_dependencies(scanner_config: str | Path, config: AtlasConfig):
    from scanner.atlas.live_feed import AtlasLiveFeed, LiveFeedOptions
    from scanner.config import load_settings
    from scanner.detection import ShopDetector
    from scanner.runtime import GameWindow, MSSScreen, PyAutoGUIInput

    settings = load_settings(scanner_config)
    screen = MSSScreen()
    window = GameWindow(settings.window_title)
    detector = ShopDetector(settings.detector)
    feed = AtlasLiveFeed(
        screen=screen,
        window=window,
        detector=detector,
        options=LiveFeedOptions(
            coord_bounds=_coord_bounds_from_boundary(config.boundary_file)
        ),
    )
    return feed, PyAutoGUIInput(), window


def _collect_live_moves(
    *,
    feed,
    input_backend,
    window,
    config: AtlasConfig,
    keys: tuple[str, ...],
    steps_per_key: int,
    hold_s: float,
    settle_s: float,
    after_timeout_s: float,
    after_poll_s: float,
    after_stable_px: float,
    min_shop_tracks: int,
    countdown_s: float,
    preflight_timeout_s: float,
    debug_dir: Path,
    require_boundary: bool,
) -> list[MoveObservation]:
    from scanner.atlas.calibrator_live import AtlasLiveCalibrator, CalibrationMovePlan

    moves: list[MoveObservation] = []
    plan = CalibrationMovePlan(
        keys=keys,
        hold_s=hold_s,
        settle_s=settle_s,
        after_timeout_s=after_timeout_s,
        after_poll_s=after_poll_s,
        after_stable_px=after_stable_px,
        repeats=1,
        min_shop_tracks=min_shop_tracks,
    )
    calibrator = AtlasLiveCalibrator(
        input_backend=input_backend,
        snapshot_provider=feed.capture_once,
        image_provider=lambda: getattr(feed, "last_client_image", None),
        plan=plan,
    )

    print(
        "ARMED: kalibracja ruszy postacią. Nie wracaj do PowerShella; "
        "Ctrl+C przerywa przed kolejnym ruchem."
    )
    _countdown(countdown_s, window)
    for key in keys:
        reverse = _OPPOSITE_KEY.get(key.lower())
        if reverse is None:
            raise CalibrationAbort(f"brak klawisza powrotu dla {key!r}")
        print(f"Kierunek {key}: {steps_per_key} kroków + powrót {reverse}.")
        completed = 0
        for index in range(steps_per_key):
            before_snapshot = _wait_for_preflight(
                feed,
                window=window,
                min_shop_tracks=min_shop_tracks,
                boundary_file=config.boundary_file,
                require_boundary=require_boundary,
                timeout_s=preflight_timeout_s,
                debug_dir=debug_dir,
            )
            observation = calibrator.move_once(key, before=before_snapshot)
            if observation is None:
                reason = calibrator.last_failure_reason or "unknown"
                recovered = _recover_observation_after_failed_move(
                    feed=feed,
                    calibrator=calibrator,
                    key=key,
                    before_snapshot=before_snapshot,
                    timeout_s=after_timeout_s,
                )
                if recovered is not None:
                    moves.append(recovered)
                    completed += 1
                    print(
                        f"  {key} {index + 1}/{steps_per_key}: "
                        f"odzysk po {reason}, coord delta={recovered.delta_game}, "
                        f"tracki={len(recovered.delta_screen)}"
                    )
                    continue
                coord_path = _try_save_last_coord_debug(
                    feed,
                    debug_dir,
                    prefix="atlas_failure_last",
                )
                if coord_path is not None:
                    print(f"  debug ostatniej klatki porazki -> {coord_path}")
                trace_path = _try_save_coord_trace_debug(
                    feed,
                    debug_dir,
                    prefix="atlas_failure_last",
                )
                if trace_path is not None:
                    print(f"  trace OCR ostatniej klatki -> {trace_path}")
                _return_steps(
                    input_backend,
                    window,
                    reverse,
                    count=1,
                    hold_s=hold_s,
                    settle_s=settle_s,
                )
                reason = calibrator.last_failure_reason or "unknown"
                if moves:
                    moves_path = _save_moves_debug(moves, debug_dir)
                    if _has_enough_partial_moves(moves):
                        print(
                            "  PARTIAL: mam wystarczajaco obserwacji po awaryjnym powrocie; "
                            "koncze zbieranie i probuje fitu z czesciowych danych."
                        )
                        return moves
                    if _can_skip_failed_direction(moves):
                        print(
                            "  PARTIAL: kierunek ma juz uzyteczne obserwacje; "
                            "pomijam reszte tego kierunku i ide dalej."
                        )
                        break
                    print(f"  debug ruchu czÄ™Ĺ›ciowy -> {moves_path}")
                coord_path = _try_save_coord_debug(feed, debug_dir)
                if coord_path is not None:
                    print(f"  debug coordĂłw po poraĹĽce -> {coord_path}")
                print(f"  powód porażki kroku {key}: {reason}")
                raise CalibrationAbort(
                    f"brak stabilnego OCR/tracków po kroku {key} #{index + 1}; "
                    f"wykonano awaryjny powrót {reverse}"
                )
            moves.append(observation)
            completed += 1
            print(
                f"  {key} {index + 1}/{steps_per_key}: "
                f"coord delta={observation.delta_game}, tracki={len(observation.delta_screen)}"
            )
        reverse_moves = _return_steps_with_observations(
            feed=feed,
            input_backend=input_backend,
            window=window,
            calibrator=calibrator,
            config=config,
            key=reverse,
            count=completed,
            hold_s=hold_s,
            settle_s=settle_s,
            min_shop_tracks=min_shop_tracks,
            preflight_timeout_s=preflight_timeout_s,
            debug_dir=debug_dir,
            require_boundary=require_boundary,
        )
        moves.extend(reverse_moves)
    if len(moves) < 2:
        raise CalibrationAbort("za mało ruchów do fitu")
    return moves


def _has_enough_partial_moves(moves: list[MoveObservation]) -> bool:
    """Czy warto oddać częściowy live-run do fitu zamiast abortować.

    To nie jest ocena jakości. Jakość nadal sprawdza fit: residual,
    condition, guard 1-track i spójność przeciwnych ruchów.
    """

    strong = [m for m in moves if _move_has_strong_screen_delta(m)]
    if len(strong) < 4:
        return False
    keys = {m.key.lower() for m in strong}
    if not (keys & {"w", "s"}) or not (keys & {"a", "d"}):
        return False
    return True


def _can_skip_failed_direction(moves: list[MoveObservation]) -> bool:
    """Czy po późnym błędzie można przejść do następnego kierunku.

    Używane gdy np. `s #3` padnie, ale `w/s` mają już kilka dobrych obserwacji.
    Nie robimy jeszcze fitu, bo może brakować drugiej osi, ale nie wyrzucamy runu.
    """

    if len([m for m in moves if _move_has_strong_screen_delta(m)]) < 3:
        return False
    return True


def _move_has_strong_screen_delta(move: MoveObservation) -> bool:
    return bool(move.delta_screen) and (
        len(move.delta_screen) >= 2 or float(move.confidence) >= 0.5
    )


def _preflight(
    snapshot: FrameSnapshot,
    *,
    window,
    min_shop_tracks: int,
    boundary_file: str,
    require_boundary: bool,
) -> None:
    if not window.is_foreground():
        raise CalibrationAbort("Glevia2 nie jest aktywnym oknem")
    if snapshot.player_game is None:
        raise CalibrationAbort("coord OCR nie odczytał pozycji")
    if len(snapshot.shops_screen) < min_shop_tracks:
        raise CalibrationAbort(
            f"za mało widocznych sklepów: {len(snapshot.shops_screen)} < {min_shop_tracks}"
        )
    if require_boundary and not _point_inside_boundary(snapshot.player_game, boundary_file):
        raise CalibrationAbort(
            f"pozycja {snapshot.player_game} jest poza granicą farm_map.json"
        )


def _wait_for_preflight(
    feed,
    *,
    window,
    min_shop_tracks: int,
    boundary_file: str,
    require_boundary: bool,
    timeout_s: float,
    debug_dir: Path | None = None,
) -> FrameSnapshot:
    deadline = time.monotonic() + max(0.0, timeout_s)
    last_error = "preflight nie wykonany"
    while True:
        snapshot = feed.capture_once()
        try:
            _preflight(
                snapshot,
                window=window,
                min_shop_tracks=min_shop_tracks,
                boundary_file=boundary_file,
                require_boundary=require_boundary,
            )
            return snapshot
        except CalibrationAbort as exc:
            last_error = str(exc)
            if time.monotonic() >= deadline:
                if debug_dir is not None:
                    path = _save_coord_debug(feed, debug_dir)
                    print(f"debug coordów -> {path}")
                raise CalibrationAbort(
                    f"preflight niegotowy po {timeout_s:.1f}s: {last_error}"
                ) from exc
            time.sleep(0.2)


def _return_steps(
    input_backend,
    window,
    key: str,
    *,
    count: int,
    hold_s: float,
    settle_s: float,
) -> None:
    for _ in range(max(0, count)):
        _wait_for_focus(window, timeout_s=10.0)
        _press_key(input_backend, key, hold_s=hold_s, settle_s=settle_s)


def _recover_observation_after_failed_move(
    *,
    feed,
    calibrator,
    key: str,
    before_snapshot: FrameSnapshot,
    timeout_s: float,
) -> MoveObservation | None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while True:
        after_snapshot = feed.capture_once()
        observation = calibrator.observation_from_snapshots(
            key,
            before_snapshot,
            after_snapshot,
            duration_s=0.0,
        )
        if observation is not None:
            return observation
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.2)


def _return_steps_with_observations(
    *,
    feed,
    input_backend,
    window,
    calibrator,
    config: AtlasConfig,
    key: str,
    count: int,
    hold_s: float,
    settle_s: float,
    min_shop_tracks: int,
    preflight_timeout_s: float,
    debug_dir: Path,
    require_boundary: bool,
) -> list[MoveObservation]:
    observations: list[MoveObservation] = []
    for index in range(max(0, count)):
        try:
            before_snapshot = _wait_for_preflight(
                feed,
                window=window,
                min_shop_tracks=min_shop_tracks,
                boundary_file=config.boundary_file,
                require_boundary=require_boundary,
                timeout_s=preflight_timeout_s,
                debug_dir=debug_dir,
            )
        except CalibrationAbort:
            _return_steps(
                input_backend,
                window,
                key,
                count=1,
                hold_s=hold_s,
                settle_s=settle_s,
            )
            print(f"  powrot {key} {index + 1}/{count}: bez obserwacji (preflight)")
            continue

        observation = calibrator.move_once(key, before=before_snapshot)
        if observation is None:
            print(
                f"  powrot {key} {index + 1}/{count}: bez obserwacji "
                f"({calibrator.last_failure_reason or 'unknown'})"
            )
            continue
        observations.append(observation)
        print(
            f"  powrot {key} {index + 1}/{count}: "
            f"coord delta={observation.delta_game}, tracki={len(observation.delta_screen)}"
        )
    return observations


def _wait_for_focus(window, *, timeout_s: float) -> None:
    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if window.is_foreground():
            return
        time.sleep(0.1)
    raise CalibrationAbort("Glevia2 nie jest aktywnym oknem — nie wykonuję powrotu")


def _ensure_foreground_for_snapshot(window, *, timeout_s: float) -> None:
    """Upewnij się, że zrzut pokazuje klienta gry, a nie okno nad nim."""

    if window.is_foreground():
        return
    try:
        activated = bool(window.activate())
    except Exception:
        activated = False
    if activated and window.is_foreground():
        time.sleep(0.2)
        return
    print("Kliknij okno Glevia2 — potrzebuję zrzutu klienta do dry-run/preflight.")
    _wait_for_focus(window, timeout_s=timeout_s)
    time.sleep(0.2)


def _print_snapshot_readiness(
    snapshot: FrameSnapshot,
    *,
    min_shop_tracks: int,
    boundary_file: str | None,
) -> bool:
    coord_ok = snapshot.player_game is not None
    shops_ok = len(snapshot.shops_screen) >= min_shop_tracks
    boundary_ok = True
    if boundary_file and snapshot.player_game is not None:
        boundary_ok = _point_inside_boundary(snapshot.player_game, boundary_file)
    print(f"okno={snapshot.window_rect}")
    print(f"coord={snapshot.player_game if coord_ok else '(brak)'}")
    print(f"sklepy={len(snapshot.shops_screen)} / min={min_shop_tracks}")
    print(f"granica={'OK' if boundary_ok else 'POZA'}")
    print(f"gotowość={'TAK' if coord_ok and shops_ok and boundary_ok else 'NIE'}")
    return coord_ok and shops_ok and boundary_ok


def _save_coord_debug(feed, debug_dir: Path) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    _, image = feed.grab_client_image()
    return _save_coord_debug_image(image, debug_dir, prefix="atlas_coords")


def _save_last_coord_debug(feed, debug_dir: Path, *, prefix: str) -> Path | None:
    image = getattr(feed, "last_client_image", None)
    if image is None:
        return None
    debug_dir.mkdir(parents=True, exist_ok=True)
    return _save_coord_debug_image(image, debug_dir, prefix=prefix)


def _save_coord_debug_image(image, debug_dir: Path, *, prefix: str) -> Path:
    from PIL import ImageDraw

    from scanner.analysis import coord_reader
    from scanner.atlas.live_feed import ATLAS_COORD_FALLBACK_ATTEMPTS

    client_path = debug_dir / f"{prefix}_client.png"
    overlay_path = debug_dir / f"{prefix}_overlay.png"
    image.save(client_path)
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    roi_specs = [
        ("TIGHT", coord_reader.TIGHT_ROI, (255, 60, 60)),
        ("DEFAULT", coord_reader.DEFAULT_ROI, (255, 215, 0)),
    ]
    roi_specs.extend(
        (f"ATLAS_{index}", attempt[0], (80, 180, 255))
        for index, attempt in enumerate(ATLAS_COORD_FALLBACK_ATTEMPTS, start=1)
    )
    seen: set[tuple[float, float, float, float]] = set()
    for name, roi, color in roi_specs:
        if roi in seen:
            continue
        seen.add(roi)
        box = _roi_box(image.size, roi)
        draw.rectangle(box, outline=color, width=3)
        draw.text((max(0, box[0] - 90), max(0, box[1] - 16)), name, fill=color)
        crop = image.crop(box)
        safe_name = name.lower().replace(" ", "_")
        crop.save(debug_dir / f"{prefix}_{safe_name}.png")
    overlay.save(overlay_path)
    return overlay_path


def _try_save_coord_debug(feed, debug_dir: Path) -> Path | None:
    try:
        return _save_coord_debug(feed, debug_dir)
    except Exception:
        return None


def _try_save_last_coord_debug(feed, debug_dir: Path, *, prefix: str) -> Path | None:
    try:
        return _save_last_coord_debug(feed, debug_dir, prefix=prefix)
    except Exception:
        return None


def _try_save_coord_trace_debug(feed, debug_dir: Path, *, prefix: str) -> Path | None:
    trace = getattr(feed, "last_coord_trace", None)
    if trace is None:
        return None
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"{prefix}_coord_trace.json"
        path.write_text(json.dumps(trace, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
    except Exception:
        return None


def _save_moves_debug(moves: list[MoveObservation], debug_dir: Path) -> Path:
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / "atlas_calibration_moves.json"
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "moves": [move.to_dict() for move in moves],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _roi_box(
    size: tuple[int, int],
    roi: tuple[float, float, float, float],
) -> tuple[int, int, int, int]:
    width, height = size
    return (
        int(width * roi[0]),
        int(height * roi[1]),
        int(width * roi[2]),
        int(height * roi[3]),
    )


def _press_key(input_backend, key: str, *, hold_s: float, settle_s: float) -> None:
    input_backend.key_down(key)
    try:
        time.sleep(max(0.0, hold_s))
    finally:
        input_backend.key_up(key)
    time.sleep(max(0.0, settle_s))


def _countdown(seconds: float, window, *, max_wait_s: float = 60.0) -> None:
    """Czekaj na stabilny fokus Glevii zamiast abortować po starcie z PowerShella."""

    stable_for = max(0.0, seconds)
    wait_deadline = time.monotonic() + max(max_wait_s, stable_for + 5.0)
    foreground_since: float | None = None
    prompted = False
    while time.monotonic() < wait_deadline:
        now = time.monotonic()
        if window.is_foreground():
            if foreground_since is None:
                foreground_since = now
                print("Glevia2 aktywna — potwierdzam stabilny fokus...")
            elapsed = now - foreground_since
            remaining = max(0.0, stable_for - elapsed)
            if remaining <= 0.0:
                print("start kalibracji.       ")
                return
            print(f"start za {remaining:.1f}s...", end="\r")
        else:
            if foreground_since is not None:
                print("Fokus utracony — licznik startu zresetowany.")
            foreground_since = None
            if not prompted:
                print("Kliknij okno Glevia2 — kalibracja czeka, nie rusza postaci.")
                prompted = True
        time.sleep(0.25)
    raise CalibrationAbort(
        "Glevia2 nie uzyskała stabilnego fokusu przed limitem startu"
    )


def _parse_keys(value: str | None, fallback: tuple[str, ...]) -> tuple[str, ...]:
    raw = value.split(",") if value else list(fallback)
    keys = tuple(key.strip().lower() for key in raw if key.strip())
    if not keys:
        raise CalibrationAbort("lista klawiszy kalibracji jest pusta")
    invalid = [key for key in keys if key not in _OPPOSITE_KEY]
    if invalid:
        raise CalibrationAbort(f"nieobsługiwane klawisze kalibracji: {invalid}")
    return keys


def _coord_bounds_from_boundary(path: str | Path) -> tuple[int, int, int, int] | tuple[int, int]:
    boundary = _load_boundary(path)
    if boundary is None or not boundary.polygon:
        from scanner.analysis import coord_reader

        return coord_reader.DEFAULT_BOUNDS
    xs = [p[0] for p in boundary.polygon]
    ys = [p[1] for p in boundary.polygon]
    margin = 80
    return (
        int(min(xs) - margin),
        int(max(xs) + margin),
        int(min(ys) - margin),
        int(max(ys) + margin),
    )


def _point_inside_boundary(point: tuple[float, float], path: str | Path) -> bool:
    boundary = _load_boundary(path)
    if boundary is None or len(boundary.polygon) < 3:
        return True
    from scanner.analysis.farm_boundary import point_in_polygon

    return point_in_polygon(point, boundary.polygon)


def _load_boundary(path: str | Path):
    from scanner.analysis.farm_boundary import FarmBoundary

    return FarmBoundary.load(Path(path))


if __name__ == "__main__":
    raise SystemExit(main())
