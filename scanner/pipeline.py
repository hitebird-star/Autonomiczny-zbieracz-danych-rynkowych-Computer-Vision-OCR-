"""Orkiestracja żywej gry; analiza jest podpinana przez wąski interfejs."""

from __future__ import annotations

import queue
import re
import threading
import math
from math import ceil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Protocol

from PIL import Image

from scanner.capture import ShopCapturer, TooltipCapturer, group_slots_by_icon
from scanner.detection import (
    ShopDetector,
    ShopInteractor,
    ShopTracker,
    TrackedShop,
    visual_fingerprint,
)
from scanner.models import ItemObservation, ScanError, ScanStatus, ShopScan
from scanner.navigation import MovementController, MovementStep, RecoveryPolicy
from scanner.runtime import InputBackend
from scanner.runtime import FocusBackend
from scanner.storage import CSVExporter, ScanRepository


class AnalysisEngine(Protocol):
    def analyze(self, scan: ShopScan, repository: ScanRepository) -> ShopScan: ...


class AnalysisSubmitter(Protocol):
    def submit(self, scan_id: str) -> None: ...


def make_scan_id(seller: str = "") -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    safe_seller = re.sub(r"[^A-Za-z0-9_.-]+", "_", seller).strip("_")[:30]
    return f"{timestamp}_{safe_seller}" if safe_seller else timestamp


@dataclass(frozen=True, slots=True)
class CaptureOutcome:
    scan: ShopScan
    duplicate: bool = False

    @property
    def successful(self) -> bool:
        return self.scan.status in {ScanStatus.CAPTURED, ScanStatus.QUEUED}


class GameCapturePipeline:
    """Otwiera jeden sklep, przechwytuje raw i natychmiast zwalnia grę."""

    def __init__(
        self,
        interactor: ShopInteractor,
        shop_capturer: ShopCapturer,
        tooltip_capturer: TooltipCapturer,
        repository: ScanRepository,
        tracker: ShopTracker,
        input_backend: InputBackend,
        *,
        close_key: str = "esc",
        open_timeout: float = 4.0,
        seller_provider: Callable[[], str] | None = None,
        analysis_queue: AnalysisSubmitter | None = None,
        recovery: RecoveryPolicy | None = None,
        focus: FocusBackend | None = None,
        window_box: tuple[int, int, int, int] | None = None,
        units_per_step: float = 7.0,
        odometry_vectors: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.interactor = interactor
        self.shop_capturer = shop_capturer
        self.tooltip_capturer = tooltip_capturer
        self.repository = repository
        self.tracker = tracker
        self.input = input_backend
        self.close_key = close_key
        self.open_timeout = open_timeout
        self.seller_provider = seller_provider or (lambda: "")
        self.analysis_queue = analysis_queue
        self.recovery = recovery or RecoveryPolicy()
        self.focus = focus
        self.window_box = window_box
        self.close_blocked = False
        self._phase_b_quorum: int | None = None
        self._phase_b_big_quorum: int | None = None
        self._known_fresh: Callable[[str], bool] | None = None  # durable dedup (DURABLE_DEDUP_CONTRACT.md)
        self.stall_count = 0       # licznik nieproduktywnych otwarć z rzędu
        self.stall_blocked = False  # przekroczono STALL_THRESHOLD
        self.max_consecutive_recovery_failures = 3  # po 3F z rzedu odpusc → wymuś WASD
        # PPM-popup jest drogi, ale bardzo skuteczny. Limit chroni przed
        # sklepem 100-slotowym, w ktorym kazda niepewna grupa probowalaby
        # otwierac dialog kupna.
        self.max_popup_topups_per_shop = 20
        self._step_index: int | None = None  # stempel (d): numer kroku WASD przy tym capture
        self._scene_for_fail: Image.Image | None = None  # Task 1: scena do zapisu przy failure
        self.units_per_step = units_per_step
        # Wektory odometrii per-klawisz (APPROACH_A_SPEC: tank-control)
        # W=przód (-2.59,-1.87), S=tył (+2.59,+1.87). None → skalar fallback.
        self._odometry_vectors: dict[str, tuple[float, float]] = odometry_vectors or {}
        self._current_position: tuple[int, int] | None = None  # odometria: ostatnia znana pozycja (x,y)
        self._position_source: str = "none"  # "ocr" | "dead_reckoning" | "none"


    def enable_phase_b(self, quorum: int = 3, big_quorum: int | None = None) -> None:
        """Wlacz Phase B: hover tylko reprezentantow stosu (quorum>=1).
        
        big_quorum: wyzsze kworum dla duzych grup (>=big_threshold=8).
        None = uzyj tego samego quorum co dla malych grup.
        """
        if quorum < 1:
            raise ValueError("Phase B quorum musi byc >= 1")
        self._phase_b_quorum = quorum
        self._phase_b_big_quorum = big_quorum

    def set_known_fresh(self, predicate: Callable[[str], bool]) -> None:
        """Wpina trwaly dedup (DURABLE_DEDUP_CONTRACT.md).
        
        predicate(fingerprint) -> True gdy sklep znany i swiezy (TTL).
        Pipeline NIE importuje ShopRegistry - dostaje czysty predykat.
        """
        self._known_fresh = predicate

    def _ensure_focus(self, timeout: float = 30.0) -> None:
        if self.focus is None or self.focus.is_foreground():
            return
        self.repository.append_event(
            self._active_scan_id,
            "capture_paused",
            reason="game_focus_lost",
        )
        print(
            "\n  PAUZA: aktywne okno nie jest Glevia2. "
            "Kliknij grę, aby bezpiecznie wznowić."
        )
        if not self.focus.wait_until_foreground(timeout):
            raise RuntimeError("game_focus_lost")

    def _save_failed_open_ui_probe(self, scan_id: str) -> None:
        """Pasywnie zapisz aktualny UI po failed-open.

        To NIE zamyka żadnych okien i NIE wysyła ESC. Służy wyłącznie do
        rozróżnienia: pusty klik / NPC dialog / popup opcji / inny panel.
        """

        probe_score = None
        try:
            score = getattr(getattr(self.interactor, "probe", None), "score", None)
            if callable(score):
                probe_score = round(float(score()), 2)
        except Exception:
            probe_score = None

        saved = None
        try:
            screen = getattr(self.shop_capturer, "screen", None)
            if self.window_box is not None and screen is not None:
                image = screen.grab(self.window_box)
                saved = self.repository.save_raw_frame(
                    scan_id, "failed_open_ui", image
                )
        except Exception as exc:
            self.repository.append_event(
                scan_id,
                "failed_open_ui_probe_error",
                error_type=type(exc).__name__,
                message=str(exc),
                probe_score=probe_score,
            )
            return

        self.repository.append_event(
            scan_id,
            "failed_open_ui_probe",
            saved=saved,
            probe_score=probe_score,
            passive=True,
            action="none",
        )

    # Granice per-mapa dla Glevia Farm (z MARKET_MAP_PLAN: pelna koperta).
    _COORD_BOUNDS_FARM = (348, 501, 672, 794)

    def _read_position_from_image(
        self,
        image,
        *,
        previous: tuple[int, int] | None = None,
    ) -> tuple[int, int] | None:
        """Odczytaj `(X,Y)` z obrazu okna: normalny OCR, potem ratunkowy ROI."""

        from scanner.analysis.coord_reader import (
            STARTUP_FALLBACK_ATTEMPTS,
            accept_reading,
            read_image,
        )

        parsed = read_image(image)
        if parsed is None:
            parsed = read_image(image, attempts=STARTUP_FALLBACK_ATTEMPTS)
        if parsed is None:
            return None

        current = (parsed.x, parsed.y)
        if not accept_reading(previous, current, bounds=self._COORD_BOUNDS_FARM):
            return None
        return current

    def read_current_position(self, step_key: str | None = None) -> tuple[int, int] | None:
        """Odometria: czytaj (X,Y) z pelnego okna gry PO KAZDYM KROKU WASD.

        W przeciwieństwie do _stamp_game_position (tylko w capture), ta metoda
        jest wołana po każdym movement.execute() — odpięta od otwierania sklepu.
        Daje gęstą trajektorię do kalibracji units/step i śledzenia pokrycia.

        Gdy OCR nie czyta, szacuje pozycję z dead-reckoning (kierunek WASD ×
        units_per_step). Aktualizuje self._current_position i self._position_source.
        Zwraca aktualną pozycję lub None.
        """
        # OCR robimy przez helper z fallbackiem na szerszy ROI.

        # Próbuj OCR z pelnego okna gry
        if self.window_box is not None and self.shop_capturer is not None:
            try:
                window_image = self.shop_capturer.screen.grab(self.window_box)
                current = self._read_position_from_image(
                    window_image,
                    previous=self._current_position,
                )
                if current is not None:
                    self._current_position = current
                    self._position_source = "ocr"
                    return current
            except Exception:
                import traceback
                traceback.print_exc()

        # Dead-reckoning fallback: estymuj z ostatniej znanej pozycji + kierunku WASD
        if step_key is not None and self._current_position is not None:
            x, y = self._current_position
            vec = self._odometry_vectors.get(step_key)
            if vec is not None:
                # Wektorowa odometria (APPROACH_A_SPEC: tank-control)
                x += vec[0]
                y += vec[1]
            else:
                # Skalar fallback (stara metoda per-oś)
                units_per_step = self.units_per_step
                if step_key in ("d", "right"):
                    x += units_per_step
                elif step_key in ("a", "left"):
                    x -= units_per_step
                elif step_key in ("s", "down"):
                    y += units_per_step
                elif step_key in ("w", "up"):
                    y -= units_per_step
            self._current_position = (x, y)
            self._position_source = "dead_reckoning"
            return self._current_position

        return None

    @property
    def current_position(self) -> tuple[int, int] | None:
        return self._current_position

    @property
    def position_source(self) -> str:
        return self._position_source

    def _stamp_game_position(self, scan: ShopScan) -> None:
        """Etap 4 Mapy Rynku: odczytaj (X,Y) z pelnego okna gry i wpisz w manifest.

        Najpierw probuje swiezy OCR z okna gry. Jesli fail → fallback do
        self._current_position (odometria z read_current_position, moze byc
        dead-reckoning). To odblokowuje densyfikacje: stamp trafia nawet gdy
        OCR nie czyta przy otwarciu sklepu.
        """
        from scanner.analysis.coord_reader import (
            STARTUP_FALLBACK_ATTEMPTS,
            accept_reading,
            read_image,
        )

        window_image = self.shop_capturer.screen.grab(self.window_box)
        parsed = read_image(window_image)
        if parsed is None:
            parsed = read_image(window_image, attempts=STARTUP_FALLBACK_ATTEMPTS)
        source = "ocr"

        if parsed is not None:
            current = (parsed.x, parsed.y)
            prev = getattr(self, '_prev_game_position', None)
            if not accept_reading(prev, current, bounds=self._COORD_BOUNDS_FARM):
                self.repository.append_event(
                    scan.scan_id,
                    "game_position_read",
                    status="rejected_by_accept_reading",
                    x=current[0], y=current[1],
                    prev_x=prev[0] if prev else None,
                    prev_y=prev[1] if prev else None,
                    bounds=list(self._COORD_BOUNDS_FARM),
                )
                # Nie fallback – accept_reading odrzuciło, zaufajmy mu
                return
        else:
            # Fallback: użyj odometrycznej pozycji (OCR lub dead-reckoning)
            if self._current_position is not None:
                current = self._current_position
                source = self._position_source + "_fallback"
            else:
                count = getattr(self, '_coord_dbg_count', 0)
                if count < 3:
                    self._coord_dbg_count = count + 1
                    self.repository.save_raw_frame(
                        scan.scan_id, "coord_full_window", window_image
                    )
                    self.repository.append_event(
                        scan.scan_id,
                        "game_position_read",
                        status="ocr_returned_none",
                        hint="sprawdz dbg/coord_full_window – DEFAULT_ROI moze nie trafiac w tekst (X,Y)",
                    )
                return

        scan.game_position = current
        self._prev_game_position = current
        self.repository.append_event(
            scan.scan_id,
            "game_position_stamped",
            x=current[0], y=current[1],
            source=source,
        )

    @staticmethod
    def _has_usable_capture(scan: ShopScan) -> bool:
        return any(
            observation.status is ScanStatus.CAPTURED
            for observation in scan.slots.values()
        )

    def _finalize_partial_capture(
        self,
        scan: ShopScan,
        track: TrackedShop,
        *,
        reason: str,
    ) -> CaptureOutcome | None:
        """Domknij czÄ™Ĺ›ciowo zebrany sklep zamiast zostawiaÄ‡ martwe `capturing`."""

        if not self._has_usable_capture(scan):
            return None
        scan.captured_slots = max(
            scan.captured_slots,
            sum(
                1
                for observation in scan.slots.values()
                if observation.status is ScanStatus.CAPTURED
            ),
        )
        self.repository.append_event(
            scan.scan_id,
            "partial_capture_finalized",
            reason=reason,
            captured_slots=scan.captured_slots,
            occupied_slots=scan.occupied_slots,
        )
        if scan.status is not ScanStatus.CAPTURED:
            self._transition(scan, ScanStatus.CAPTURED, "capture_completed_partial")
        self.tracker.mark_visited(track)
        if self.analysis_queue is not None:
            self._transition(scan, ScanStatus.QUEUED, "analysis_queued")
            self.analysis_queue.submit(scan.scan_id)
        return CaptureOutcome(scan)

    def capture(self, track: TrackedShop) -> CaptureOutcome:
        scan = ShopScan(
            scan_id=make_scan_id(),
            screen_position=track.position,
            step_index=self._step_index,
        )
        self.repository.create(scan)
        self._active_scan_id = scan.scan_id
        try:
            self._ensure_focus()
        except RuntimeError:
            error = ScanError(
                failed_stage="opening",
                reason="game_focus_lost",
                recoverable=True,
            )
            self._fail(scan, error)
            self.tracker.mark_failed(track, terminal=False)
            return CaptureOutcome(scan)
        self._transition(scan, ScanStatus.APPROACHING, "shop_approaching")
        self._transition(scan, ScanStatus.OPENING, "shop_opening")

        interaction = self.interactor.open(
            track.position, timeout=self.open_timeout
        )
        if not interaction.opened:
            error = ScanError(
                failed_stage="opening",
                reason=interaction.reason or "shop_window_not_detected",
                retry_count=track.attempts + max(0, interaction.attempts - 1),
                recoverable=True,
                details={"elapsed": interaction.elapsed},
            )
            # Task 1: zapisz scene przy nieudanym otwarciu (do diagnostyki offline)
            if self._scene_for_fail is not None:
                try:
                    self.repository.save_raw_frame(
                        scan.scan_id, "failed_open_scene", self._scene_for_fail
                    )
                except Exception:
                    import traceback
                    traceback.print_exc()
            self._save_failed_open_ui_probe(scan.scan_id)
            self._fail(scan, error)
            decision = self.recovery.decide(error)
            # ``ShopInteractor`` wykonał już dwie pełne próby kliknięcia.
            # Kolejne cztery kliknięcia w ten sam płot/obiekt tylko ponownie
            # uruchamiały click-to-move. Ten cel i jego bliskie otoczenie
            # pomijamy do końca bieżącej sesji.
            self.tracker.mark_failed(
                track,
                terminal=decision.terminal or interaction.attempts >= 2,
            )
            return CaptureOutcome(scan)

        self._transition(scan, ScanStatus.OPENED, "shop_opened")
        opened = True
        try:
            self._ensure_focus()
            scan.seller = self.seller_provider().strip()
            self.repository.append_event(
                scan.scan_id,
                "seller_read",
                seller=scan.seller,
                detected=bool(scan.seller),
            )
            self._transition(scan, ScanStatus.CAPTURING, "capture_started")
            # Etap 4 Mapy Rynku: stempluj wspolrzedne swiata (adnotacja, nie blokuje).
            if self.window_box is not None:
                try:
                    self._stamp_game_position(scan)
                except Exception as exc:
                    # Stage 4: OCR moze byc niedostepny (Tesseract brak w PATH,
                    # DEFAULT_ROI rozjechany z oknem, itp.). Loguj raz na sesje.
                    if getattr(self, '_coord_logged', None) != type(exc).__name__:
                        self._coord_logged = type(exc).__name__
                        self.repository.append_event(
                            scan.scan_id,
                            "game_position_error",
                            error_type=type(exc).__name__,
                            message=str(exc),
                        )
            shop_image = self.shop_capturer.capture_shop()
            self.repository.save_shop_image(scan.scan_id, shop_image)

            fingerprint = visual_fingerprint(shop_image)
            scan.shop_fingerprint = fingerprint
            
            # Durable dedup: czy sklep znany z poprzedniego biegu?
            known_fresh = self._known_fresh is not None and self._known_fresh(fingerprint)
            
            if self.tracker.attach_fingerprint(track, fingerprint):
                reason = "duplicate_known_fresh" if known_fresh else "duplicate_shop_fingerprint"
                error = ScanError(
                    failed_stage="capturing",
                    reason=reason,
                    recoverable=False,
                )
                self._fail(scan, error)
                return CaptureOutcome(scan, duplicate=True)
            elif known_fresh:
                # Sesyjny tracker nie zna, ale rejestr tak -> trwaly duplikat
                # (sklep widziany w poprzednim biegu, inny tracker).
                error = ScanError(
                    failed_stage="capturing",
                    reason="duplicate_known_fresh",
                    recoverable=False,
                )
                self._fail(scan, error)
                self.tracker.attach_fingerprint(track, fingerprint)  # oznacz tez sesyjnie
                return CaptureOutcome(scan, duplicate=True)

            grid_image = self.shop_capturer.capture_grid()
            occupied = self.shop_capturer.occupied_slots(grid_image)
            scan.occupied_slots = len(occupied)
            icon_groups = group_slots_by_icon(
                grid_image,
                occupied,
                self.shop_capturer.geometry,
            )
            ordered_slots = [
                slot for group in icon_groups for slot in group
            ]
            ordered_slots_all = list(ordered_slots)
            group_by_slot = {
                slot.slot: group_index
                for group_index, group in enumerate(icon_groups, start=1)
                for slot in group
            }
            self.repository.append_event(
                scan.scan_id,
                "grid_captured",
                occupied_slots=len(occupied),
                icon_groups=len(icon_groups),
            )

            # Phase B: wybierz reprezentantow per icon_group (floor=3).
            # Deferred sloty NIE sa hoverowane - dziedzicza consensus_unit z engine.
            deferred_slots: set[int] = set()
            if self._phase_b_quorum is not None:
                from scanner.analysis.representatives import select_representatives

                groups_dict = {
                    idx + 1: [s.slot for s in group]
                    for idx, group in enumerate(icon_groups)
                }
                selection = select_representatives(
                    groups_dict,
                    quorum=self._phase_b_quorum,
                    big_quorum=self._phase_b_big_quorum,
                )
                rep_set: set[int] = set()
                for sel in selection.values():
                    rep_set.update(sel.representatives)
                    deferred_slots.update(sel.deferred)
                # Przefiltruj ordered_slots do samych reprezentantow
                ordered_slots_all = list(ordered_slots)  # kopia przed filtracja
                ordered_slots = [s for s in ordered_slots if s.slot in rep_set]
                self.repository.append_event(
                    scan.scan_id,
                    "phase_b",
                    quorum=self._phase_b_quorum,
                    groups=len(selection),
                    representatives=len(rep_set),
                    deferred=len(deferred_slots),
                )

            group_references: dict[
                int, tuple[Image.Image, list[str], int]
            ] = {}
            failed_slots = []
            popup_budget = [self.max_popup_topups_per_shop]

            def capture_slot(
                slot,
                *,
                index: int,
                total: int,
                capture_pass: int,
            ) -> bool:
                self._ensure_focus()
                icon_group = group_by_slot[slot.slot]
                label = "slot" if capture_pass == 1 else "recovery"
                print(
                    f"  {label} {index}/{total} "
                    f"(kol={slot.column}, wiersz={slot.row}, "
                    f"ikona={icon_group})...",
                    end="",
                    flush=True,
                )
                reference = group_references.get(icon_group)
                capture_stack_member = getattr(
                    self.tooltip_capturer, "capture_stack_member", None
                )
                capture_stack_member_fast = getattr(
                    self.tooltip_capturer,
                    "capture_stack_member_fast",
                    None,
                )
                capture_fast = getattr(
                    self.tooltip_capturer,
                    "capture_fast",
                    None,
                )
                if (
                    capture_pass == 1
                    and reference is not None
                    and callable(capture_stack_member_fast)
                ):
                    captured = capture_stack_member_fast(slot, reference[0])
                elif (
                    capture_pass == 1
                    and reference is None
                    and callable(capture_fast)
                ):
                    captured = capture_fast(slot)
                elif reference is not None and callable(capture_stack_member):
                    captured = capture_stack_member(slot, reference[0])
                else:
                    captured = self.tooltip_capturer.capture(slot)
                if not captured.frames:
                    print(" brak dymka")
                    error = ScanError(
                        failed_stage="capturing",
                        reason="tooltip_not_detected",
                        recoverable=True,
                        details={"slot": slot.slot},
                    )
                    scan.slots[slot.slot] = ItemObservation(
                        slot=slot.slot,
                        row=slot.row,
                        column=slot.column,
                        images=[],
                        icon_group=icon_group,
                        status=ScanStatus.FAILED,
                        error=error,
                    )
                    self.repository.append_event(
                        scan.scan_id,
                        "slot_capture_failed",
                        slot=slot.slot,
                        capture_pass=capture_pass,
                        error=error.to_dict(),
                        **(
                            {"recovery_pass": True}
                            if capture_pass == 2
                            else {}
                        ),
                    )
                    # Zapisz surowe ramki dla diagnostyki detektora (Claude replay).
                    # Uzywamy save_raw_frame (nie save_tooltip_image) bo slot/frame
                    # nie spelniaja ograniczen 0..99/>=1.
                    if captured.baseline is not None:
                        try:
                            self.repository.save_raw_frame(
                                scan.scan_id,
                                f"slot_{slot.slot:03d}_baseline",
                                captured.baseline,
                            )
                        except Exception:
                            import traceback
                            traceback.print_exc()
                    if captured.last_candidate is not None:
                        try:
                            self.repository.save_raw_frame(
                                scan.scan_id,
                                f"slot_{slot.slot:03d}_hover",
                                captured.last_candidate,
                            )
                        except Exception:
                            import traceback
                            traceback.print_exc()
                    self.repository.save_manifest(scan)
                    return False
                if captured.matched_reference and reference is not None:
                    paths = list(reference[1])
                    evidence = [f"icon_duplicate_of:{reference[2]}"]
                    print(f" DUP slotu {reference[2]}")
                    self.repository.append_event(
                        scan.scan_id,
                        "slot_deduplicated",
                        slot=slot.slot,
                        icon_group=icon_group,
                        duplicate_of=reference[2],
                        capture_pass=capture_pass,
                    )
                else:
                    print(f" OK ({len(captured.frames)} klatki)")
                    paths = [
                        self.repository.save_tooltip_image(
                            scan.scan_id, slot.slot, frame_index, frame
                        )
                        for frame_index, frame in enumerate(
                            captured.frames, start=1
                        )
                    ]
                    evidence = (
                        ["tooltip_recovered_on_pass_2"]
                        if capture_pass == 2
                        else []
                    )
                    group_references.setdefault(
                        icon_group,
                        (captured.frames[0], list(paths), slot.slot),
                    )
                scan.slots[slot.slot] = ItemObservation(
                    slot=slot.slot,
                    row=slot.row,
                    column=slot.column,
                    images=paths,
                    icon_group=icon_group,
                    status=ScanStatus.CAPTURED,
                    evidence=evidence,
                )
                scan.captured_slots += 1
                self.repository.append_event(
                    scan.scan_id,
                    "slot_captured",
                    slot=slot.slot,
                    images=paths,
                    icon_group=icon_group,
                    deduplicated=captured.matched_reference,
                    capture_pass=capture_pass,
                    recovered=capture_pass == 2,
                    **(
                        {"recovery_pass": True}
                        if capture_pass == 2
                        else {}
                    ),
                )
                # Manifest po każdym slocie pozwala wznowić pracę po awarii.
                self.repository.save_manifest(scan)
                return True

            for index, slot in enumerate(ordered_slots, start=1):
                if not capture_slot(
                    slot,
                    index=index,
                    total=len(ordered_slots),
                    capture_pass=1,
                ):
                    failed_slots.append(slot)

            # Stack-aware recovery: pomijaj sloty, ktorych icon_group ma juz
            # zlapanego brata (cena jest). Oszczedza ~76% kosztu recovery.
            covered, recoverable = [], []
            for slot in failed_slots:
                if group_by_slot[slot.slot] in group_references:
                    covered.append(slot)
                else:
                    recoverable.append(slot)
            for slot in covered:
                rep = group_references[group_by_slot[slot.slot]][2]
                obs = scan.slots[slot.slot]
                obs.evidence = list(obs.evidence or []) + [f"stack_covered_by:{rep}"]
                self.repository.append_event(
                    scan.scan_id, "recovery_skipped",
                    slot=slot.slot, reason="stack_covered",
                    icon_group=group_by_slot[slot.slot], covered_by=rep,
                )
            self.repository.save_manifest(scan)

            if recoverable:
                before_recovery = scan.captured_slots
                # Minimalny znacznik zgodny z offline harness Claude'a.
                # Bogatsze zdarzenia recovery_pass poniżej pozostają źródłem
                # szczegółów dla diagnostyki live i zgodności wstecznej.
                self.repository.append_event(
                    scan.scan_id,
                    "recovery_started",
                    queued=len(recoverable),
                )
                self.repository.append_event(
                    scan.scan_id,
                    "recovery_pass",
                    phase="started",
                    attempted_slots=[slot.slot for slot in recoverable],
                    attempted=len(recoverable),
                    captured_before=before_recovery,
                )
                recovered = 0
                consecutive_failures = 0
                for index, slot in enumerate(recoverable, start=1):
                    # Faza 1 BUY_DIALOG: PPM-popup zamiast re-hovera (~0% yield → ~100%)
                    if self._recover_slot_via_popup(
                        scan,
                        slot,
                        icon_group=group_by_slot.get(slot.slot),
                        reason="tooltip_recovery",
                        popup_budget=popup_budget,
                    ):
                        recovered += 1
                        consecutive_failures = 0
                        continue
                    if capture_slot(
                        slot,
                        index=index,
                        total=len(failed_slots),
                        capture_pass=2,
                    ):
                        recovered += 1
                        consecutive_failures = 0
                    else:
                        consecutive_failures += 1
                        if consecutive_failures >= self.max_consecutive_recovery_failures:
                            self.repository.append_event(
                                scan.scan_id,
                                "recovery_aborted",
                                reason="consecutive_failures",
                                threshold=self.max_consecutive_recovery_failures,
                            )
                            break
                self.repository.append_event(
                    scan.scan_id,
                    "recovery_pass",
                    phase="completed",
                    attempted=len(recoverable),
                    recovered=recovered,
                    remaining=len(failed_slots) - recovered,
                    captured_before=before_recovery,
                    captured_after=scan.captured_slots,
                )

            # BRAMKA ≥2 (user 23.06): każda grupa ≥2 slotów musi mieć ≥2 UDANE
            # odczyty dymka. quorum=2 WYBIERA 2, ale hover bywa pudłem → dohoveruj
            # nieprzeczytanego członka aż 2 udane lub wyczerpiesz grupę.
            from scanner.analysis.representatives import needs_more_reads

            slots_by_group: dict[int, list] = {}
            for s in ordered_slots_all:
                g = group_by_slot.get(s.slot)
                if g is not None:
                    slots_by_group.setdefault(g, []).append(s)

            for g, members in slots_by_group.items():
                gsize = len(members)
                read = {m.slot for m in members
                        if scan.slots.get(m.slot)
                        and scan.slots[m.slot].status == ScanStatus.CAPTURED
                        and (scan.slots[m.slot].images or "buy_popup" in (scan.slots[m.slot].evidence or []))}
                tried: set[int] = set()  # 🔴 PUŁAPKA: bez tried pętla wisi (test_pipeline)
                while needs_more_reads(len(read), gsize):
                    nxt = next((m for m in members
                                if m.slot not in read and m.slot not in tried),
                               None)
                    if nxt is None:
                        self.repository.append_event(
                            scan.scan_id, "group_reads_insufficient",
                            icon_group=g, group_size=gsize, reads=len(read),
                        )
                        break  # nie ma kogo dohoverować — flaga, nie blokuj na amen
                    tried.add(nxt.slot)  # przed capture_slot: nie próbuj 2× tego samego
                    if self._recover_slot_via_popup(
                        scan,
                        nxt,
                        icon_group=g,
                        reason="group_reads_topup",
                        popup_budget=popup_budget,
                    ):
                        read.add(nxt.slot)
                        self.repository.append_event(
                            scan.scan_id, "group_reads_topup",
                            icon_group=g, reads=len(read), source="buy_popup",
                        )
                        continue
                    if capture_slot(nxt, index=1, total=1, capture_pass=2):
                        read.add(nxt.slot)
                        self.repository.append_event(
                            scan.scan_id, "group_reads_topup",
                            icon_group=g, reads=len(read), source="tooltip",
                        )

            # Phase B: deferred sloty (nie hoverowane) dziedzicza z grupy.
            for slot in ordered_slots_all:
                if slot.slot in deferred_slots:
                    scan.slots.setdefault(slot.slot, ItemObservation(
                        slot=slot.slot,
                        row=slot.row,
                        column=slot.column,
                        images=[],
                        icon_group=group_by_slot.get(slot.slot),
                        status=ScanStatus.CAPTURED,
                        evidence=["stack_representative"],
                    ))

            # Faza 4.1: licz takĹźe deferred (dziedziczone) sloty – Phase B redukuje
            # hovers, wiÄc captured_slots moĹźe byÄ âŞ occupied. Deferred majÄ status
            # CAPTURED (dziedziczÄ wartoĹÄ z reprezentanta) – licz ich na rĂłwni.
            captured_all = max(
                scan.captured_slots,
                sum(1 for o in scan.slots.values() if o.status == ScanStatus.CAPTURED),
            )
            minimum_captured = max(1, ceil(scan.occupied_slots * 0.5))
            if captured_all == 0:
                # NIC nie złapaliśmy – faktyczna porażka (bez zmian)
                error = ScanError(
                    failed_stage="capturing",
                    reason="insufficient_tooltip_yield",
                    retry_count=track.attempts,
                    recoverable=True,
                    details={
                        "occupied_slots": scan.occupied_slots,
                        "captured_slots": scan.captured_slots,
                        "minimum_captured": minimum_captured,
                    },
                )
                self._fail(scan, error)
                decision = self.recovery.decide(error)
                self.tracker.mark_failed(track, terminal=decision.terminal)
                return CaptureOutcome(scan)
            if captured_all < minimum_captured:
                # WARSTWA B: mało, ale coś jest – eksport częściowy, NIE wyrzucaj
                self.repository.append_event(
                    scan.scan_id, "low_yield_partial",
                    captured=captured_all, minimum=minimum_captured,
                    occupied=scan.occupied_slots)

            self._transition(scan, ScanStatus.CAPTURED, "capture_completed")
            self.tracker.mark_visited(track)  # KLUCZ: oznacz odwiedzony → fingerprint do seen-set
            if self.analysis_queue is not None:
                self._transition(scan, ScanStatus.QUEUED, "analysis_queued")
                self.analysis_queue.submit(scan.scan_id)
            return CaptureOutcome(scan)
        except Exception as exc:
            partial = self._finalize_partial_capture(
                scan,
                track,
                reason=f"{type(exc).__name__}:{exc}",
            )
            if partial is not None:
                return partial
            error = ScanError(
                failed_stage="capturing",
                reason="capture_exception",
                recoverable=True,
                details={"type": type(exc).__name__, "message": str(exc)},
            )
            if scan.status is not ScanStatus.FAILED:
                self._fail(scan, error)
            decision = self.recovery.decide(error)
            self.tracker.mark_failed(track, terminal=decision.terminal)
            return CaptureOutcome(scan)
        finally:
            if opened:
                closed = self._force_close(scan.scan_id)
                self.close_blocked = not closed

    def _force_close(self, scan_id: str) -> bool:
        """Zamknij okno sklepu z eskalującymi metodami.

        Próbuje sekwencyjnie: Esc + czekanie (2 próby), klik X okna,
        combo Esc + X. Każdy etap loguje wynik. Zwraca True jeśli
        okno zostało zamknięte.
        """
        # --- faza 0: utrata fokusu = nie możemy zamknąć -----------------
        if self.focus is not None and not self.focus.is_foreground():
            self.repository.append_event(
                scan_id,
                "shop_close_skipped",
                reason="game_not_foreground",
            )
            return False

        self.input.press(self.close_key)
        self.repository.append_event(scan_id, "shop_close_requested")

        wait_closed = getattr(self.interactor, "wait_closed", None)
        if not callable(wait_closed):
            return True  # nie mamy jak sprawdzić – uznajemy za zamknięte

        # --- faza 1: Esc + czekanie 1.5 s (2 próby) -------------------
        closed = wait_closed(1.5)
        if not closed:
            self.input.press(self.close_key)
            closed = wait_closed(1.5)

        if closed:
            self.repository.append_event(scan_id, "shop_close_confirmed")
            return True

        # --- faza 2: klik X okna sklepu ---------------------------------
        geometry = self.shop_capturer.geometry
        shop_x, shop_y, shop_w, shop_h = geometry.shop_box
        # Przycisk X w prawym górnym rogu paska tytułowego (~20 px od
        # prawej krawędzi, ~10 px od góry).
        close_x = shop_x + shop_w - 20
        close_y = shop_y + 10
        self.input.move_to(close_x, close_y, duration=0.05)
        self.input.click()
        closed = wait_closed(1.0)
        if closed:
            self.repository.append_event(
                scan_id, "shop_close_confirmed", method="button"
            )
            return True

        # --- faza 3: combo Esc + klik X (ostatnia szansa) ---------------
        self.input.press(self.close_key)
        self.input.move_to(close_x, close_y, duration=0.03)
        self.input.click()
        closed = wait_closed(1.0)
        if closed:
            self.repository.append_event(
                scan_id, "shop_close_confirmed", method="esc_button"
            )
            return True

        self.repository.append_event(
            scan_id,
            "shop_close_escalation_failed",
            attempts=3,
        )
        return False

    def _recover_slot_via_popup(
        self,
        scan: ShopScan,
        slot,
        *,
        icon_group: int | None,
        reason: str,
        popup_budget: list[int],
    ) -> bool:
        """Sprobuj odczytac slot przez PPM-popup i zapisz spojny event.

        `popup_budget` to mutowalny licznik [pozostalo] wspoldzielony przez caly
        sklep. Dzieki temu rozszerzamy pop-out na brakujace reprezentanty
        stackow, ale nie PPM-ujemy calego sklepu slot po slocie.
        """

        if not self.window_box or popup_budget[0] <= 0:
            return False
        popup_budget[0] -= 1
        self.repository.append_event(
            scan.scan_id,
            "popup_topup_attempted",
            slot=slot.slot,
            icon_group=icon_group,
            reason=reason,
            remaining_budget=popup_budget[0],
        )
        previous = scan.slots.get(slot.slot)
        was_counted = bool(
            previous
            and previous.status == ScanStatus.CAPTURED
            and (
                previous.images
                or "buy_popup" in (previous.evidence or [])
            )
        )
        popup_obs = self._read_slot_from_buy_popup(scan, slot)
        if popup_obs is None:
            self.repository.append_event(
                scan.scan_id,
                "popup_topup_failed",
                slot=slot.slot,
                icon_group=icon_group,
                reason=reason,
            )
            return False
        popup_obs.icon_group = icon_group
        popup_obs.evidence = list(popup_obs.evidence or []) + [
            f"popup_topup:{reason}",
        ]
        scan.slots[slot.slot] = popup_obs
        if not was_counted:
            scan.captured_slots += 1
        self.repository.append_event(
            scan.scan_id,
            "slot_recovered_via_popup",
            slot=slot.slot,
            icon_group=icon_group,
            reason=reason,
            evidence=["buy_popup"],
        )
        self.repository.save_manifest(scan)
        return True

    def _read_slot_from_buy_popup(self, scan: ShopScan, slot) -> ItemObservation | None:
        """Faza 1 BUY_DIALOG: PPM w slot → popup kupna → OCR → parse (zamiast recovery hovera)."""
        from scanner.analysis.buy_dialog import parse_buy_dialog
        import win_ocr, time

        center = self.shop_capturer.geometry.slot_center(slot.column, slot.row)
        try:
            self.input._api.rightClick(center[0], center[1])
        except Exception:
            self.input._api.click(center[0], center[1], button='right')

        lines = []
        deadline = time.monotonic() + 1.5
        while time.monotonic() < deadline:
            time.sleep(0.2)
            popup_grab = self.shop_capturer.screen.grab(
                (self.window_box[0] + self.window_box[2] // 2 - 200,
                 self.window_box[1] + self.window_box[3] // 2 - 100, 400, 220))
            lines = win_ocr.recognize(popup_grab.resize(
                (popup_grab.width * 3, popup_grab.height * 3), Image.LANCZOS))
            if lines: break

        if not lines:
            return None
        d = parse_buy_dialog(str(line.get("text") or "") for line in lines)
        if d is None: return None
        self.input.press("esc")
        # gap B: „Cena wynosi" to TOTAL stacka → cena jednostkowa
        unit = round(d.price / max(1, d.quantity)) if d.price else None
        obs = ItemObservation(slot=slot.slot, row=slot.row, column=slot.column,
                              images=[], icon_group=None,
                              status=ScanStatus.CAPTURED, evidence=["buy_popup"])
        obs.validation = {"status": "provisional", "source": "buy_popup",
                         "item": d.name, "unit_price": unit,
                         "quantity": d.quantity}
        return obs

    def _transition(self, scan: ShopScan, status: ScanStatus, event: str) -> None:
        scan.transition(status)
        self.repository.save_manifest(scan)
        self.repository.append_event(scan.scan_id, event, status=status.value)

    def _fail(self, scan: ShopScan, error: ScanError) -> None:
        scan.transition(ScanStatus.FAILED, error=error)
        self.repository.save_manifest(scan)
        self.repository.append_event(
            scan.scan_id, "scan_failed", error=error.to_dict()
        )


class AnalysisWorker:
    """Opcjonalny worker; działa także z przyszłym readerem/validatorem Claude’a."""

    def __init__(
        self,
        repository: ScanRepository,
        engine: AnalysisEngine,
        *,
        exporter: CSVExporter | None = None,
    ) -> None:
        self.repository = repository
        self.engine = engine
        self.exporter = exporter
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._run, name="scanner-analysis", daemon=True
        )
        self._thread.start()

    def submit(self, scan_id: str) -> None:
        self.start()
        self._queue.put(scan_id)

    def stop(self, timeout: float | None = None) -> None:
        if not self._thread:
            return
        self._queue.put(None)
        self._thread.join(timeout)

    def join(self) -> None:
        self._queue.join()

    def _run(self) -> None:
        while True:
            scan_id = self._queue.get()
            try:
                if scan_id is None:
                    return
                scan = self.repository.load(scan_id)
                scan.transition(ScanStatus.ANALYZING)
                self.repository.save_manifest(scan)
                self.repository.append_event(
                    scan_id, "analysis_started", status=scan.status.value
                )
                result = self.engine.analyze(scan, self.repository)
                self.repository.save_manifest(result)
                self.repository.append_event(
                    scan_id, "analysis_completed", status=result.status.value
                )
                if self.exporter is not None:
                    exported = self.exporter.export(result)
                    self.repository.append_event(
                        scan_id, "csv_exported", rows=exported
                    )
            except Exception as exc:
                try:
                    scan = self.repository.load(str(scan_id))
                    error = ScanError(
                        failed_stage="analyzing",
                        reason="analysis_exception",
                        recoverable=True,
                        details={"type": type(exc).__name__, "message": str(exc)},
                    )
                    if scan.status is not ScanStatus.FAILED:
                        scan.transition(ScanStatus.FAILED, error=error)
                    self.repository.save_manifest(scan)
                    self.repository.append_event(
                        str(scan_id), "analysis_failed", error=error.to_dict()
                    )
                except Exception:
                    pass
            finally:
                self._queue.task_done()


class AutonomousMarketLoop:
    """Detekcja bieżącego widoku, skan nowych sklepów, następnie krok trasy."""

    def __init__(
        self,
        detector: ShopDetector,
        tracker: ShopTracker,
        capture_pipeline: GameCapturePipeline,
        movement: MovementController,
        view_provider: Callable[[], tuple[Image.Image, tuple[int, int]]],
        diagnostics=None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.capture_pipeline = capture_pipeline
        self.movement = movement
        self.view_provider = view_provider
        self.diagnostics = diagnostics
        self._on_shop_captured: Callable[[ShopScan], None] | None = None
        self._step_counter: int = 0
        self._position_buffer: list[tuple[int, int]] = []  # ostatnie N pozycji dla is_stuck
        self._stuck_count: int = 0  # ile razy z rzędu wykryto stuck
        self._last_positions_for_stuck: list[tuple[float, float]] = []  # float dla nav_guards
        self._steps_since_fix: int = 0  # licznik dla fix_stale (B5)
        self._recovery_attempt: int = 0  # licznik dla recovery_plan (naprzemiennie a/d)
        self._covmap = None  # CoverageMap (opcjonalnie, Filar 2)
        self._coverage_done = False
        self._farm_boundary = None  # FarmBoundary (PILNE #3: skręt ±45° przy granicy)
        self._goto_step_budget = 12     # FIX B: 12 zamiast 4 (obroty zjadają iteracje)
        self._goto_max_turns = 1        # max korekt obrotu (anty-pętla)
        self._turn_nudge_steps = 2      # ile 'd' na 1 korektę (≈kąt — STROJENIE LIVE)
        self._cell_size = 20            # rozmiar komórki (dla reached_target)
        self._step_hold = 0.6           # domyślny czas przytrzymania W
        self._settle = 0.9              # domyślny settle po kroku
        self._coverage_dup_floor = 4    # live: mniej przedwczesnego "cell_exhausted"
        self._boundary_escape_turn_limit = 3  # miękkie odbicie od granicy
        self._movement_memory = None
        self._boundary_escape_turns = 0
        self._g4_stuck_count = 0
        self._g4_recovery_attempts = 0
        self._last_g4_pos: tuple[float, float] | None = None
        self._coverage_stop_reason: str | None = None
        self._last_drive_block_reason: str | None = None
        self._last_learned_bad_move_cell: tuple[int, int] | None = None
        self._learned_bad_move_repeats = 0

    def set_shop_captured_hook(self, hook: Callable[[ShopScan], None] | None) -> None:
        """Hook wywolywany po kazdym przechwyconym sklepie (dla ZoneNavigator)."""
        self._on_shop_captured = hook

    def set_coverage_map(self, covmap) -> None:
        """Wpina CoverageMap do sterowania ruchem (Filar 2: jazda-do-celu)."""
        self._covmap = covmap
        self._cell_size = float(getattr(covmap, "cell_size", self._cell_size) or self._cell_size)

    def set_farm_boundary(self, boundary) -> None:
        """Wpina FarmBoundary do guarda ruchu (PILNE #3: skręt ±45° przy granicy)."""
        self._farm_boundary = boundary

    def set_movement_memory(self, memory) -> None:
        """Wpina samouczącą pamięć ruchu po komórkach coverage."""
        self._movement_memory = memory

    def reset_coverage_pass_state(self) -> None:
        """Wyczyść krótkoterminowy stan nawigacji przed kolejnym pass coverage-drive."""

        self._coverage_stop_reason = None
        self._coverage_component_exhausted = False
        self._last_drive_block_reason = None
        self._last_learned_bad_move_cell = None
        self._learned_bad_move_repeats = 0
        self._g4_stuck_count = 0
        self._g4_recovery_attempts = 0
        self._last_g4_pos = None
        self._stuck_count = 0
        self._last_positions_for_stuck.clear()
        self._boundary_escape_turns = 0
        self.capture_pipeline.close_blocked = False
        self.capture_pipeline.stall_blocked = False
        self.capture_pipeline.stall_count = 0

    # Koperta farmy dla recovery (FARM_ENVELOPE z _guard_movement).
    _FARM_ENVELOPE: tuple[float, float, float, float] = (348, 672, 501, 794)

    @staticmethod
    def _inside_coverage_envelope(pos: tuple[float, float], envelope) -> bool:
        x_min, x_max, y_min, y_max = envelope
        return x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max

    @staticmethod
    def _distance_to_coverage_envelope(pos: tuple[float, float], envelope) -> float:
        x_min, x_max, y_min, y_max = envelope
        dx = max(x_min - pos[0], 0.0, pos[0] - x_max)
        dy = max(y_min - pos[1], 0.0, pos[1] - y_max)
        return math.hypot(dx, dy)

    def _movement_vector(self, step_key: str) -> tuple[float, float] | None:
        vectors = getattr(self.capture_pipeline, "_odometry_vectors", {})
        vec = vectors.get(step_key)
        if vec is not None:
            return vec
        if self._covmap is None:
            return None
        units = float(getattr(self.capture_pipeline, "units_per_step", 0.0) or 0.0)
        if units <= 0:
            return None
        if step_key in ("s", "down"):
            return (0.0, units)
        if step_key in ("w", "up"):
            return (0.0, -units)
        return None

    def _movement_cell(self, pos) -> tuple[int, int] | None:
        if self._covmap is None or pos is None:
            return None
        try:
            return self._covmap.cell_of((float(pos[0]), float(pos[1])))
        except Exception:
            return None

    def _learn_blocked_move(self, step_key: str, pos, reason: str) -> None:
        if getattr(self, "_movement_memory", None) is None:
            return
        cell = self._movement_cell(pos)
        if cell is None:
            return
        self._movement_memory.record_blocked(cell, step_key)
        if self.diagnostics is not None:
            stat = self._movement_memory.stats(cell, step_key)
            self.diagnostics.event(
                "movement_learning",
                outcome="blocked",
                key=step_key,
                cell=list(cell),
                reason=reason,
                attempts=stat.attempts,
                failures=stat.failures,
                failure_rate=round(stat.failure_rate, 3),
            )

    def _learn_stuck_move(self, step_key: str, pos, reason: str) -> None:
        if getattr(self, "_movement_memory", None) is None:
            return
        cell = self._movement_cell(pos)
        if cell is None:
            return
        self._movement_memory.record_stuck(cell, step_key)
        if self.diagnostics is not None:
            stat = self._movement_memory.stats(cell, step_key)
            self.diagnostics.event(
                "movement_learning",
                outcome="stuck",
                key=step_key,
                cell=list(cell),
                reason=reason,
                attempts=stat.attempts,
                failures=stat.failures,
                failure_rate=round(stat.failure_rate, 3),
            )

    def _learn_successful_move(self, step_key: str, before, after) -> None:
        if getattr(self, "_movement_memory", None) is None or before is None or after is None:
            return
        if getattr(self.capture_pipeline, "position_source", "") != "ocr":
            return
        cell = self._movement_cell(before)
        if cell is None:
            return
        moved = math.hypot(
            float(after[0]) - float(before[0]),
            float(after[1]) - float(before[1]),
        )
        if moved < 1.0:
            self._movement_memory.record_stuck(cell, step_key)
            outcome = "stuck"
        else:
            self._movement_memory.record_success(cell, step_key)
            outcome = "success"
        if self.diagnostics is not None:
            stat = self._movement_memory.stats(cell, step_key)
            self.diagnostics.event(
                "movement_learning",
                outcome=outcome,
                key=step_key,
                cell=list(cell),
                moved=round(moved, 2),
                attempts=stat.attempts,
                failures=stat.failures,
                failure_rate=round(stat.failure_rate, 3),
            )

    def _escape_learned_bad_move_trap(self, pos, target: tuple[float, float]) -> None:
        """Małe fizyczne wyrwanie z pułapki learned-memory.

        `movement_memory` jest celowo prosta: pamięta (komórka, klawisz), ale nie
        zna aktualnego obrotu postaci. Gdy bot obróci się w miejscu, ten sam klawisz
        W może fizycznie oznaczać już inny kierunek. Dlatego po kilku blokadach
        `learned_bad_move` nie kończymy biegu — robimy miękki obrót i krótki probe,
        żeby odświeżyć heading/pozycję i dać plannerowi nowy stan.
        """

        if self.diagnostics is not None:
            self.diagnostics.event(
                "learned_bad_move_escape",
                pos=list(pos) if pos is not None else None,
                target=list(target),
            )
        print("  ESCAPE: learned_bad_move — obrót + krótki probe zamiast kończenia")
        before_turn = self.capture_pipeline.current_position
        self.movement.execute("d", 0.25, 0.15)
        after_turn = self.capture_pipeline.read_current_position(None)
        self._learn_successful_move("d", before_turn, after_turn)

        # Świadomie NIE pytamy tu movement_memory, bo to ona nas zamroziła.
        # Boundary/coverage i tak złapią błąd w kolejnej iteracji, a krótki probe
        # daje szansę wyjść z tekstury / zaktualizować heading po obrocie.
        before_probe = self.capture_pipeline.current_position
        self.movement.execute("w", 0.25, 0.45)
        after_probe = self.capture_pipeline.read_current_position("w")
        self._learn_successful_move("w", before_probe, after_probe)

    def _handle_failed_drive(
        self,
        *,
        pos,
        target: tuple[float, float],
    ) -> bool:
        """Obsłuż nieudany dojazd. True = przerwij coverage-loop."""

        if self._covmap is None or pos is None:
            return False
        reason = getattr(self, "_last_drive_block_reason", None)
        if reason != "learned_bad_move":
            self._learned_bad_move_repeats = 0
            self._last_learned_bad_move_cell = None
            return False

        cell = self._covmap.cell_of((float(pos[0]), float(pos[1])))
        target_cell = self._covmap.cell_of(target)
        self._covmap.mark_unreachable(target_cell)
        self._covmap.record_block(pos, "stall")
        if cell == self._last_learned_bad_move_cell:
            self._learned_bad_move_repeats += 1
        else:
            self._last_learned_bad_move_cell = cell
            self._learned_bad_move_repeats = 1
        if self.diagnostics is not None:
            self.diagnostics.event(
                "learned_bad_move_repeat",
                cell=list(cell),
                target_cell=list(target_cell),
                repeats=self._learned_bad_move_repeats,
                pos=list(pos),
                target=list(target),
            )
        if self._learned_bad_move_repeats >= 3:
            self._escape_learned_bad_move_trap(pos, target)
            self._learned_bad_move_repeats = 0
            self._last_learned_bad_move_cell = None
            self._coverage_stop_reason = None
            return False
        return False

    def _movement_blocked_by_coverage(
        self, step_key: str
    ) -> tuple[bool, str, tuple[float, float] | None]:
        pos = self.capture_pipeline.current_position
        vec = self._movement_vector(step_key)
        if pos is None or vec is None:
            return False, "", None
        current = (float(pos[0]), float(pos[1]))
        next_pos = (current[0] + float(vec[0]), current[1] + float(vec[1]))
        cell = self._movement_cell(current)
        if self._covmap is not None:
            envelope = self._covmap.envelope
            if not self._inside_coverage_envelope(current, envelope):
                current_dist = self._distance_to_coverage_envelope(current, envelope)
                next_dist = self._distance_to_coverage_envelope(next_pos, envelope)
                if self._inside_coverage_envelope(next_pos, envelope) or next_dist < current_dist:
                    if getattr(self, "diagnostics", None) is not None:
                        self.diagnostics.event(
                            "outside_envelope_return_allowed",
                            key=step_key,
                            pos=list(current),
                            next_pos=list(next_pos),
                            current_dist=round(current_dist, 2),
                            next_dist=round(next_dist, 2),
                        )
                    return False, "", next_pos
                self._learn_blocked_move(step_key, current, "outside_coverage_envelope")
                return True, "outside_coverage_envelope", next_pos

        if (
            getattr(self, "_movement_memory", None) is not None
            and cell is not None
            and self._movement_memory.should_avoid(cell, step_key)
        ):
            return True, "learned_bad_move", next_pos

        if self._covmap is not None:
            envelope = self._covmap.envelope
            if not self._inside_coverage_envelope(next_pos, envelope):
                self._learn_blocked_move(step_key, current, "next_outside_coverage_envelope")
                return True, "next_outside_coverage_envelope", next_pos
            distance = max(1.0, math.hypot(vec[0], vec[1]))
            if self._covmap.blocked_ahead(current, next_pos, lookahead=distance):
                self._learn_blocked_move(step_key, current, "coverage_blocked_ahead")
                return True, "coverage_blocked_ahead", next_pos
            if self._covmap.is_blocked(self._covmap.cell_of(next_pos)):
                self._learn_blocked_move(step_key, current, "coverage_cell_blocked")
                return True, "coverage_cell_blocked", next_pos

        if self._farm_boundary is not None:
            from scanner.analysis.farm_boundary import should_turn_at_boundary

            if should_turn_at_boundary(
                current, next_pos, self._farm_boundary.polygon, margin=7.0
            ):
                self._learn_blocked_move(step_key, current, "farm_boundary")
                return True, "farm_boundary", next_pos
        return False, "", next_pos

    def _handle_preflight_block(
        self,
        reason: str,
        step_key: str,
        next_pos: tuple[float, float] | None,
    ) -> bool:
        pos = self.capture_pipeline.current_position
        print(f"  GUARD RUCHU: blokuje {step_key.upper()} przed ruchem ({reason})")
        if self.diagnostics is not None:
            self.diagnostics.event(
                "movement_blocked_preflight",
                key=step_key,
                reason=reason,
                pos=list(pos) if pos is not None else None,
                next_pos=list(next_pos) if next_pos is not None else None,
            )
        if self._covmap is not None and pos is not None:
            self._covmap.record_block(pos, reason)
            path = self._covmap.path_to_next_target(
                pos,
                until="done",
                prefer_uncovered=True,
                dup_floor=self._coverage_dup_floor,
                heading=getattr(self.capture_pipeline, "_odometry_vectors", {}).get("w"),
            )
            if path is None:
                if self._covmap.all_done(dup_floor=self._coverage_dup_floor):
                    self._coverage_done = True
                    print("  GUARD RUCHU: coverage_done")
                else:
                    reentry = (
                        self._covmap.reentry_path(pos)
                        if hasattr(self._covmap, "reentry_path")
                        else None
                    )
                    if reentry is not None:
                        hop = self._covmap.cell_center(reentry[-1])
                        if self.diagnostics is not None:
                            self.diagnostics.event(
                                "coverage_reentry_required",
                                source="preflight",
                                pos=list(pos),
                                target=list(hop),
                                cell=list(reentry[-1]),
                            )
                        print("  GUARD RUCHU: reentry_required")
                        ok = self._drive_toward_target(hop)
                        if not ok and self._handle_failed_drive(pos=pos, target=hop):
                            return True
                        return ok
                    print("  GUARD RUCHU: brak osiagalnego celu w tej skladowej")
                return True
            hop = self._covmap.cell_center(path[1] if len(path) > 1 else path[0])
            ok = self._drive_toward_target(hop)
            if not ok and self._handle_failed_drive(pos=pos, target=hop):
                return True
            return ok
        self._lane_turn_180()
        return True

    def _drive_toward_target(self, target: tuple[float, float]) -> bool:
        """Dojedź do world-celu: rotate-measure-correct (FIX A+B) + FIX C + budget ~12.

        FIX A: każdy 'w' mierzy żywy heading z pary OCR → _odometry_vectors odświeżane.
        FIX B: budżet 12 zamiast 4 (obroty zjadają iteracje).
        FIX C: ENV_HN w konwencji heading_nav (x_min,x_max,y_min,y_max).
        """
        from scanner.analysis.heading_nav import aim_point, facing_error_deg, better_turn_key
        from scanner.analysis.nav_guards import heading_from_ocr

        cp = self.capture_pipeline
        self._last_drive_block_reason = None
        # FIX C: heading_nav oczekuje (x_min, x_max, y_min, y_max) — INNA kolejność niż nav_guards!
        ENV_HN = (348.0, 501.0, 672.0, 794.0)
        TOL = 30.0
        turn_key = "d"          # domyślna strona próby (douczy się)
        prev_err = None
        backsteps = 0           # limit dyndania 's' do nieosiągalnego celu (anty „pętla w i s")
        # FIX B: budżet podniesiony z 4 do 12 (obroty zjadają iteracje)
        for _ in range(self._goto_step_budget):
            pos = cp.current_position
            if pos is None:
                cp.read_current_position(None)     # spróbuj OCR, BEZ fabrykacji dead-reckoning
                continue
            # FIX C: dobra koperta
            aim = aim_point(pos, target, ENV_HN)
            px, py = float(pos[0]), float(pos[1])
            if ((aim[0] - px) ** 2 + (aim[1] - py) ** 2) ** 0.5 < self._cell_size * 0.5:
                self._last_drive_block_reason = None
                return True
            # ZMIERZ żywy heading: jeden krok 'w' + para OCR
            p0 = (px, py)
            blocked, reason, next_pos = self._movement_blocked_by_coverage("w")
            if blocked:
                self._last_drive_block_reason = reason
                w_vec = self._movement_vector("w")
                s_vec = self._movement_vector("s")
                to_target = (float(target[0]) - px, float(target[1]) - py)
                target_behind = (
                    w_vec is not None
                    and s_vec is not None
                    and (
                        to_target[0] * float(w_vec[0])
                        + to_target[1] * float(w_vec[1])
                    )
                    < 0.0
                    and (
                        to_target[0] * float(s_vec[0])
                        + to_target[1] * float(s_vec[1])
                    )
                    > 0.0
                )
                escape_reasons = {
                    "farm_boundary",
                    "next_outside_coverage_envelope",
                    "coverage_blocked_ahead",
                    "coverage_cell_blocked",
                }
                if target_behind and reason in escape_reasons:
                    s_blocked, s_reason, s_next = self._movement_blocked_by_coverage("s")
                    if not s_blocked and backsteps < 3:
                        backsteps += 1
                        if self.diagnostics is not None:
                            self.diagnostics.event(
                                "goto_escape_backstep",
                                reason=reason,
                                attempt=backsteps,
                                pos=list(pos),
                                target=list(target),
                                w_vec=list(w_vec),
                                s_vec=list(s_vec),
                                next_pos=list(s_next) if s_next is not None else None,
                            )
                        before_s = cp.current_position
                        self.movement.execute("s", self._step_hold, self._settle)
                        after_s = cp.read_current_position("s")
                        self._learn_successful_move("s", before_s, after_s)
                        self._boundary_escape_turns = 0
                        prev_err = None
                        continue
                    if self.diagnostics is not None:
                        self.diagnostics.event(
                            "goto_escape_blocked",
                            reason=reason,
                            s_reason=s_reason,
                            pos=list(pos),
                            target=list(target),
                            next_pos=list(s_next) if s_next is not None else None,
                        )
                    if (
                        s_reason in escape_reasons
                        and self._boundary_escape_turns
                        < self._boundary_escape_turn_limit
                    ):
                        self._boundary_escape_turns += 1
                        if self.diagnostics is not None:
                            self.diagnostics.event(
                                "goto_escape_turn",
                                attempt=self._boundary_escape_turns,
                                reason=reason,
                                s_reason=s_reason,
                                pos=list(pos),
                                target=list(target),
                                soft=True,
                            )
                        self.movement.execute(turn_key, 0.18, 0.12)
                        cp.read_current_position(None)
                        prev_err = None
                        continue
                if self.diagnostics is not None:
                    self.diagnostics.event(
                        "goto_blocked_preflight",
                        reason=reason,
                        pos=list(pos),
                        next_pos=list(next_pos) if next_pos is not None else None,
                        target=list(target),
                    )
                if self._covmap is not None:
                    self._covmap.record_block(pos, reason)
                return False
            self.movement.execute("w", self._step_hold, self._settle)
            self._boundary_escape_turns = 0
            p1 = cp.read_current_position("w")
            self._learn_successful_move("w", p0, p1)
            if cp.position_source == "ocr" and p1 is not None and tuple(p1) != tuple(p0):
                heading = heading_from_ocr(p0, p1, magnitude=3.2)
                cp._odometry_vectors["w"] = heading                    # FIX A: ODŚWIEŻ heading
                cp._odometry_vectors["s"] = (-heading[0], -heading[1])
                err = facing_error_deg(heading, p1, aim)
                if err <= TOL:
                    prev_err = None                                    # patrzę na cel → następna iter znów 'w'
                    continue
                if prev_err is not None:
                    turn_key = better_turn_key(prev_err, err, turn_key) # douczanie strony bez kalibracji
                for _ in range(self._turn_nudge_steps):
                    self.movement.execute(turn_key, 0.25, 0.2)
                prev_err = err
                cp.read_current_position(None)                          # odśwież pozycję (bez fabrykacji)
            # brak OCR → następna iteracja po prostu znów 'w' (heading bez zmian)
        return False

    def _movement_recovery(self) -> None:
        """Recovery przy zacięciu: recovery_plan → heading_from_OCR (Filar B)."""
        from scanner.analysis.nav_guards import recovery_plan, heading_from_ocr

        self._recovery_attempt += 1
        plan = recovery_plan(self._recovery_attempt, back_steps=1)
        print(f"  RECOVERY plan (attempt {self._recovery_attempt}): {len(plan)} ruchów")
        for key, dur, sett in plan:
            blocked, reason, next_pos = self._movement_blocked_by_coverage(key)
            if blocked:
                self._last_drive_block_reason = reason
                if self.diagnostics is not None:
                    self.diagnostics.event(
                        "recovery_step_blocked",
                        key=key,
                        reason=reason,
                        pos=(
                            list(self.capture_pipeline.current_position)
                            if self.capture_pipeline.current_position is not None
                            else None
                        ),
                        next_pos=list(next_pos) if next_pos is not None else None,
                    )
                print(f"  RECOVERY: blokuje {key.upper()} ({reason}) – przerywam plan")
                break
            before = self.capture_pipeline.current_position
            self.movement.execute(key, dur, sett)
            after = self.capture_pipeline.read_current_position(key)
            self._learn_successful_move(key, before, after)

        # Filar B: po skręcie odśwież heading z dwóch OCR-ów
        self.capture_pipeline._position_source = "none"
        pos1 = self.capture_pipeline.read_current_position("w")
        pos2 = self.capture_pipeline.read_current_position("w")
        if pos1 is not None and pos2 is not None and pos1 != pos2:
            new_vec = heading_from_ocr(pos1, pos2, magnitude=3.2)
            self.capture_pipeline._odometry_vectors["w"] = new_vec
            self.capture_pipeline._odometry_vectors["s"] = (-new_vec[0], -new_vec[1])
            print(f"  RECOVERY: nowy wektor W = ({new_vec[0]:.2f}, {new_vec[1]:.2f})")
        else:
            print("  RECOVERY: brak dwóch OCR – wektor W bez zmian")

        self._stuck_count = 0
        self._last_positions_for_stuck.clear()
        self._steps_since_fix = 0
        print("  RECOVERY: gotowe – wznawiam marsz W")

    def _lane_turn_180(self) -> None:
        """Zwrot 180° na granicy farmy – deterministyczna negacja wektora.
        
        APPROACH_A_SPEC fix #2: heading_from_ocr milczy bez 2 OCR (15/19 dead-reckon).
        Negacja wektora = zwrot 180° bez czekania na OCR. s = old_w (cofnij).
        """
        old_w = self.capture_pipeline._odometry_vectors.get("w", (-2.59, -1.87))
        new_w = (-old_w[0], -old_w[1])
        new_s = old_w  # s = stary w (cofnij w przeciwną stronę)
        self.capture_pipeline._odometry_vectors["w"] = new_w
        self.capture_pipeline._odometry_vectors["s"] = new_s
        print(f"  LANE TURN 180: nowy W = ({new_w[0]:.2f}, {new_w[1]:.2f})")

    def _guard_movement(self, step_key: str) -> None:
        """Sprawdz zacięcie / wyjście poza kopertę po każdym kroku WASD.
        
        Używa nav_guards (is_stuck, within_envelope) do detekcji.
        Reakcja: cofnij + zmień kierunek (recovery zamiast jazdy w ścianę).
        """
        # Granice koperty Glevia Farm
        FARM_ENVELOPE = (348, 672, 501, 794)
        STUCK_WINDOW = 3   # wystarczą 3 identyczne pozycje z rzędu

        # Guard 3: stale movement MUSI być PRZED bramką pos=None
        # (w teksturze OCR pada → pos=None → stary kod ucinał się tutaj)
        from scanner.analysis.nav_guards import fix_stale
        self._steps_since_fix += 1
        if fix_stale(self._steps_since_fix, max_steps=7):
            print(
                f"  GUARD RUCHU: stale movement – "
                f"{self._steps_since_fix} kroków bez fixu, wymuszam recovery"
            )
            self._movement_recovery()
            return  # po recovery reszta guardów niepotrzebna

        pos = self.capture_pipeline.current_position
        if pos is None:
            return  # brak pozycji = Guard 1 i 2 nie mają na czym pracować

        # Realny fix OCR → wyzeruj licznik stale (zdrowy marsz nie potrzebuje recovery)
        if self.capture_pipeline.position_source == "ocr":
            self._steps_since_fix = 0

        # Bufor pozycji dla is_stuck
        self._last_positions_for_stuck.append((float(pos[0]), float(pos[1])))
        if len(self._last_positions_for_stuck) > STUCK_WINDOW:
            self._last_positions_for_stuck.pop(0)

        # Guard 1 + Filar 2 + Faza 3: krawędź LUB pokryta komórka → jedź do NIEpokrytej
        from scanner.analysis.nav_guards import within_envelope
        vec = getattr(self.capture_pipeline, "_odometry_vectors", {}).get(
            step_key, (0, 0)
        )
        next_x, next_y = pos[0] + vec[0], pos[1] + vec[1]
        if self._covmap is not None:
            edge = (
                not self._inside_coverage_envelope(
                    (float(pos[0]), float(pos[1])),
                    self._covmap.envelope,
                )
                or not self._inside_coverage_envelope(
                    (float(next_x), float(next_y)),
                    self._covmap.envelope,
                )
                or self._covmap.blocked_ahead(
                    (float(pos[0]), float(pos[1])),
                    (float(next_x), float(next_y)),
                    lookahead=max(1.0, math.hypot(vec[0], vec[1])),
                )
            )
        else:
            edge = (
                not within_envelope(next_x, next_y, FARM_ENVELOPE)
                or not within_envelope(pos[0], pos[1], FARM_ENVELOPE)
            )
        covered = self._covmap is not None and self._covmap.is_covered((next_x, next_y))
        if edge or covered:
            if self._covmap is not None:
                # C6: next_target(until="done") – bot wraca do pokrytej-ale-niewysyconej
                # komórki, aż przestaje dawać nowe sklepy (anti-przedwczesne-przejście).
                path = self._covmap.path_to_next_target(
                    pos,
                    until="done",
                    dup_floor=self._coverage_dup_floor,
                )
                if path is None:
                    if self._covmap.all_done(dup_floor=self._coverage_dup_floor):
                        self._coverage_done = True
                    else:
                        self._coverage_component_exhausted = True
                    return
                target = self._covmap.cell_center(
                    path[1] if len(path) > 1 else path[0]
                )
                logical_target = self._covmap.cell_center(path[-1])
                if target is None or self._covmap.all_done(
                    dup_floor=self._coverage_dup_floor
                ):
                    self._coverage_done = True
                    print("  GUARD RUCHU: farma wysycona – coverage_done")
                    return
                self._stuck_count = 0
                # D2: record_block przed goto
                if edge and self._covmap is not None:
                    self._covmap.record_block(pos, "edge_hit")
                ok = self._drive_toward_target(target)  # FIZYCZNY obrót+jazda do pustej
                if not ok:
                    if self._handle_failed_drive(pos=pos, target=target):
                        return
                    if self._covmap is not None:
                        self._covmap.record_block(pos, "goto_fail")
                print(
                    f"  GUARD RUCHU: go-to uncovered ({target[0]:.0f},{target[1]:.0f}) "
                    f"– {'dotarl' if ok else 'nie dotarl'}"
                )
                if self.diagnostics is not None:
                    self.diagnostics.event(
                        "goto_uncovered",
                        target=list(logical_target),
                        hop=list(target),
                        path_cells=[list(cell) for cell in path],
                        reached=ok,
                        trigger="edge" if edge else "covered",
                    )
                return
            else:
                # PILNE #3: jeśli mamy FarmBoundary, użyj should_turn_at_boundary
                # zamiast ślepego 180° – skręć ±45° i jedź wzdłuż granicy.
                if self._farm_boundary is not None:
                    from scanner.analysis.farm_boundary import should_turn_at_boundary
                    if should_turn_at_boundary(pos, (next_x, next_y), self._farm_boundary.polygon, margin=7.0):
                        print("  GUARD RUCHU: granica farmy – skręt ±45° zamiast jazdy w pustkę")
                        self._stuck_count = 0
                        # Obrót w bok (dwa nudge'y 'd') i krok 'w' wzdłuż granicy
                        before_d = self.capture_pipeline.current_position
                        self.movement.execute("d", 0.18, 0.12)
                        after_d = self.capture_pipeline.read_current_position("d")
                        self._learn_successful_move("d", before_d, after_d)
                        blocked, reason, next_pos = self._movement_blocked_by_coverage("w")
                        if blocked:
                            if self.diagnostics is not None:
                                self.diagnostics.event(
                                    "boundary_turn_step_blocked",
                                    key="w",
                                    reason=reason,
                                    pos=list(pos),
                                    next_pos=list(next_pos) if next_pos is not None else None,
                                )
                            return
                        before_w = self.capture_pipeline.current_position
                        self.movement.execute("w", self._step_hold, self._settle)
                        after_w = self.capture_pipeline.read_current_position("w")
                        self._learn_successful_move("w", before_w, after_w)
                        if self.diagnostics is not None:
                            self.diagnostics.event("boundary_turn", pos=list(pos))
                        return
                self._stuck_count = 0
                self._lane_turn_180()  # fallback bez covmap i bez granicy
                return

        # Guard 2: stuck (pozycja zamrożona mimo kroków)
        from scanner.analysis.nav_guards import is_stuck
        if len(self._last_positions_for_stuck) >= STUCK_WINDOW:
            stuck = is_stuck(self._last_positions_for_stuck, max_same=STUCK_WINDOW, eps=2.0)
            if stuck:
                self._stuck_count += 1
                # G6: stall z guardu uczy teksturę (border_adjustments → no_go)
                if self._covmap is not None:
                    self._covmap.record_block(pos, "stall")
                print(
                    f"  GUARD RUCHU: stuck wykryty (×{self._stuck_count}) – "
                    f"pozycja stoi mimo {STUCK_WINDOW}× '{step_key}'"
                )
                if self._stuck_count >= 2:
                    print("  GUARD RUCHU: recovery – cofam + obrót")
                    self._learn_stuck_move(step_key, pos, "guard_stuck")
                    self._movement_recovery()
            else:
                self._stuck_count = 0

    def _force_step(self) -> None:
        """Wymuś krótki krok w tył, by zmienić scenę i przerwać livelock.

        Używane przez anti-stall (close_blocked + stall_blocked). Niezależne
        od trasy WASD – działa także gdy walk=False.
        """
        blocked, reason, next_pos = self._movement_blocked_by_coverage("s")
        if blocked:
            self._handle_preflight_block(reason, "s", next_pos)
            return
        before = self.capture_pipeline.current_position
        self.movement.execute("s", 0.3, 0.5)
        self._step_counter += 1
        # Odometria: śledź pozycję nawet przy wymuszonym kroku
        pos = self.capture_pipeline.read_current_position("s")
        self._learn_successful_move("s", before, pos)
        if self.diagnostics is not None and pos is not None:
            self.diagnostics.event(
                "position_read",
                step=self._step_counter,
                x=pos[0],
                y=pos[1],
                source=self.capture_pipeline.position_source,
                forced=True,
            )
        if self.diagnostics is not None:
            self.diagnostics.event("movement_forced", reason="anti_stall")

    def _handle_scan_barrier(self) -> bool:
        """Wspólny anti-stall po skanie widoku, także dla coverage-drive."""

        blocked = getattr(self.capture_pipeline, "close_blocked", False)
        stalled = getattr(self.capture_pipeline, "stall_blocked", False)
        if not (blocked or stalled):
            return False

        if hasattr(self, "_last_visible") and self.tracker.has_untried(self._last_visible):
            print("  ANTI-STALL: są nietknięte sklepy – kontynuuję postój")
            self.capture_pipeline.stall_blocked = False
            self.capture_pipeline.stall_count = 0
            return False

        tag = (
            "close_blocked"
            if blocked
            else f"stall_blocked ({self.capture_pipeline.stall_count})"
        )
        print(f"  ANTI-STALL: {tag} - wymuszam krok WASD by zmienic scene.")
        self._force_step()
        self.capture_pipeline.close_blocked = False
        self.capture_pipeline.stall_blocked = False
        self.capture_pipeline.stall_count = 0
        return True

    def scan_current_view(self, *, max_shops: int = 0) -> list[CaptureOutcome]:
        outcomes = []
        while True:
            if getattr(self.capture_pipeline, "close_blocked", False):
                print(
                    "  STOP: okno sklepu nadal otwarte po eskalacji. "
                    "Przerywam skanowanie bieżącego widoku – "
                    "ruch WASD (jeśli aktywny) rozwiąże blokadę."
                )
                break
            if getattr(self.capture_pipeline, "stall_blocked", False):
                print(
                    "  STOP: wykryto livelock – "
                    f"{self.capture_pipeline.stall_count} nieproduktywne "
                    "otwarcia z rzędu. Wymuszam zmianę sceny."
                )
                break
            # Po każdym sklepie scena może się przesunąć, bo postać podchodzi do
            # straganu. Nie używamy więc starych współrzędnych z początku rundy.
            image, offset = self.view_provider()
            candidates = self.detector.detect(image, screen_offset=offset)
            # Preserve detector runtime order: hybrid pushes likely-false
            # targets down, then distance clears the nearest ring. A second
            # "forward" reorder here used to wipe that ranking and made the bot
            # tunnel through shops in one direction instead of clearing the
            # current left/right neighbourhood around the character.
            visible = self.tracker.update(candidates)
            self._last_visible = visible  # SCAN_DRAIN_FIX C: pamiętaj ostatni widok dla has_untried
            track = self.tracker.next_unvisited(visible)
            legacy_pick = None
            # Diagnostyczny kontrfaktyczny wybor: ten sam stan trackera i ten
            # sam zbior kandydatow, ale kolejnosc tylko po odleglosci. Nie
            # zmienia decyzji runtime ani stanu visited/failed.
            if self.diagnostics is not None and hasattr(
                self.tracker, "peek_unvisited"
            ):
                pairs = sorted(
                    zip(candidates, visible),
                    key=lambda pair: pair[0].distance,
                )
                legacy_pick = self.tracker.peek_unvisited(
                    [visible_track for _, visible_track in pairs]
                )
            if self.diagnostics is not None:
                self.diagnostics.record_detection(
                    image,
                    self.detector.mask_image(image),
                    candidates,
                    visible,
                    track,
                    screen_offset=offset,
                    legacy_pick=legacy_pick,
                )
            if track is None:
                break
            # Faza 2 (anty-reskan): nie klikaj PONOWNIE w przebranej komórce.
            #
            # Istotne: ``should_skip_click`` jest statystyką *komórki*, nie
            # tożsamością aktualnego celu. Wcześniej blokowaliśmy także świeży
            # track (attempts == 0), gdy w tej samej komórce były już dwa
            # duplikaty. Skutek był dokładnie odwrotny do zamierzonego: bot
            # docierał do nieprzeskanowanej strefy, widział nowy sklep, lecz
            # kończył skan widoku i ruszał dalej. Nowy, jeszcze niekliknięty
            # kandydat zawsze ma pierwszeństwo przed lokalnym dedupem.
            if self._covmap is not None:
                char_pos = self.capture_pipeline.current_position
                exhausted = (
                    char_pos is not None
                    and self._covmap.should_skip_click(
                        char_pos,
                        dup_floor=self._coverage_dup_floor,
                    )
                )
                fresh_target = track.attempts == 0
                if self.diagnostics is not None:
                    self.diagnostics.event(
                        "click_gate",
                        track_id=track.track_id,
                        attempts=track.attempts,
                        fresh_target=fresh_target,
                        cell_exhausted=exhausted,
                        decision=(
                            "allow_fresh_target"
                            if exhausted and fresh_target
                            else "skip_retry_in_exhausted_cell"
                            if exhausted
                            else "allow"
                        ),
                        pos=list(char_pos) if char_pos is not None else None,
                    )
                if exhausted and not fresh_target:
                    print(
                        f"  SKIP: komórka przebrana "
                        f"({self._covmap.dups_in_cell(self._covmap.cell_of(char_pos))} dups) "
                        f"– idę do niepokrytej"
                    )
                    if self.diagnostics is not None:
                        self.diagnostics.event("skip_exhausted_cell", pos=list(char_pos))
                    break  # zakończ widok → trasa/guard pchnie do niepokrytej
            print(
                f"  sklep {track.track_id}: cel={track.position}, "
                f"próba={track.attempts + 1}"
            )
            self.capture_pipeline._scene_for_fail = image
            self.capture_pipeline._step_index = self._step_counter
            outcome = self.capture_pipeline.capture(track)
            self.capture_pipeline._scene_for_fail = None
            outcomes.append(outcome)
            # Anti-livelock: zliczaj per-track (nie globalnie).
            # Tracker.mark_failed(terminal=True) już obsługuje skipnięcie
            # zablokowanego tracka – pętla naturalnie przechodzi do następnego.
            if outcome.duplicate or outcome.scan.status == ScanStatus.FAILED:
                self.capture_pipeline.stall_count += 1
            else:
                self.capture_pipeline.stall_count = 0
            # Globalny stall TYLKO gdy close_blocked (faza 4 eskalacji close)
            # lub gdy 5 różnych tracków z rzędu failuje (problem z kamerą/oknem)
            if self.capture_pipeline.stall_count >= 5:
                self.capture_pipeline.stall_blocked = True
            if self.diagnostics is not None:
                self.diagnostics.record_capture(track, outcome)
            if self._on_shop_captured is not None and outcome.successful:
                try:
                    self._on_shop_captured(outcome.scan)
                except Exception:
                    pass
            # Faza 2 (anty-reskan): rejestruj wynik w coverage_map
            if self._covmap is not None and outcome.scan.game_position is not None:
                self._covmap.record_scan(
                    outcome.scan.game_position,
                    duplicate=outcome.duplicate,
                )
            print(
                f"  sklep {track.track_id}: wynik={outcome.scan.status.value}"
                + (" (duplikat)" if outcome.duplicate else "")
            )
            # Limit oznacza liczbę poprawnie przechwyconych sklepów, a nie
            # liczbę kliknięć. Pudła detektora nie mogą zakończyć obchodu przed
            # pierwszym krokiem trasy.
            if max_shops and sum(item.successful for item in outcomes) >= max_shops:
                break
        return outcomes

    def run(
        self,
        route: tuple[MovementStep, ...] = (),
        *,
        max_shops: int = 0,
        navigator = None,
    ) -> list[CaptureOutcome]:
        outcomes = []

        # G4: pętla biegu sterowana mapą (gęsty grid, COVERAGE_DENSITY_PLAN)
        if self._covmap is not None and not route and navigator is None:
            print("G4 drive loop: sterowanie next_target aż coverage_done")
            while True:
                # Najpierw skanuj bieżące otoczenie. Coverage-drive nie może
                # ruszać przed opróżnieniem widoku ani ignorować max_shops.
                successful = sum(outcome.successful for outcome in outcomes)
                if max_shops and successful >= max_shops:
                    print("  G4 LOOP: max_shops_reached")
                    break
                remaining = max(0, max_shops - successful) if max_shops else 0
                outcomes.extend(self.scan_current_view(max_shops=remaining))
                if self._handle_scan_barrier():
                    continue
                if max_shops and sum(outcome.successful for outcome in outcomes) >= max_shops:
                    print("  G4 LOOP: max_shops_reached")
                    break

                pos = self.capture_pipeline.current_position
                if pos is None:
                    self._coverage_stop_reason = "position_missing"
                    print("  G4 LOOP: brak pozycji OCR — kończę bez ślepego ruchu")
                    break
                pos_f = (float(pos[0]), float(pos[1]))
                if hasattr(self._covmap, "envelope") and not self._inside_coverage_envelope(pos_f, self._covmap.envelope):
                    self._g4_recovery_attempts += 1
                    if self.diagnostics is not None:
                        self.diagnostics.event(
                            "outside_envelope_recovery",
                            pos=list(pos_f),
                            attempt=self._g4_recovery_attempts,
                            envelope=list(self._covmap.envelope),
                        )
                    if self._g4_recovery_attempts > 10:
                        self._coverage_stop_reason = "outside_envelope_stuck"
                        if self.diagnostics is not None:
                            self.diagnostics.event(
                                "physically_stuck",
                                pos=list(pos_f),
                                repeats=self._g4_stuck_count,
                                recoveries=self._g4_recovery_attempts,
                                reason="outside_envelope_stuck",
                            )
                        break
                    print("  G4 LOOP: poza mapą — wracam do coverage envelope")
                    self._movement_recovery()
                    self._last_g4_pos = None
                    continue
                if self._last_g4_pos is not None:
                    moved = math.hypot(
                        pos_f[0] - self._last_g4_pos[0],
                        pos_f[1] - self._last_g4_pos[1],
                    )
                    if moved < self._cell_size * 0.5:
                        self._g4_stuck_count += 1
                    else:
                        self._g4_stuck_count = 0
                        self._g4_recovery_attempts = 0   # ruszył → seria recovery wyzerowana
                    if self._g4_stuck_count >= 4:
                        # FIX: stuck NIE kończy biegu. Bot ma REAGOWAĆ na blok — cofnij+obróć
                        # (istniejący _movement_recovery), doucz `no_go` w komórce i jedź dalej.
                        # Koniec dopiero gdy recovery NAPRAWDĘ wyczerpane (nigdzie się nie da).
                        self._g4_recovery_attempts += 1
                        if self.diagnostics is not None:
                            self.diagnostics.event(
                                "g4_stuck_recovery",
                                pos=list(pos_f),
                                repeats=self._g4_stuck_count,
                                attempt=self._g4_recovery_attempts,
                            )
                        if self._g4_recovery_attempts > 6:
                            self._coverage_stop_reason = "physically_stuck"
                            print(
                                "  G4 LOOP: unrecoverable_stuck — recovery wyczerpane "
                                f"({self._g4_recovery_attempts}×), kończę"
                            )
                            if self.diagnostics is not None:
                                self.diagnostics.event(
                                    "physically_stuck",
                                    pos=list(pos_f),
                                    repeats=self._g4_stuck_count,
                                    recoveries=self._g4_recovery_attempts,
                                )
                            break
                        print(
                            f"  G4 LOOP: stuck — recovery #{self._g4_recovery_attempts} "
                            "(cofam+obrót, douczam no_go), jadę dalej"
                        )
                        self._learn_stuck_move("w", pos, "g4_stuck")
                        self._covmap.record_block(pos, "stall")
                        self._movement_recovery()
                        self._g4_stuck_count = 0
                        self._last_g4_pos = None
                        continue
                self._last_g4_pos = pos_f
                # G4: next_target w trybie wężyka po wszystkich komórkach
                path = self._covmap.path_to_next_target(
                    pos,
                    order="boustrophedon",
                    until="done",
                    # Domykaj bieżący sektor zanim pójdziesz w świeży teren.
                    # `record_scan()` od razu oznacza komórkę jako covered, ale
                    # to nie znaczy, że sklepowy ring jest wysycony. Preferowanie
                    # pustych komórek powodowało: dotknął -> idzie dalej, mimo
                    # że wokół nadal stoją sklepy.
                    prefer_uncovered=False,
                    dup_floor=self._coverage_dup_floor,
                )
                if path is None:
                    if self._covmap.all_done(dup_floor=self._coverage_dup_floor):
                        self._coverage_done = True
                        end_reason = "coverage_done"
                        print(f"  G4 LOOP: {end_reason} – farma wysycona")
                        break
                    reentry = (
                        self._covmap.reentry_path(pos)
                        if hasattr(self._covmap, "reentry_path")
                        else None
                    )
                    if reentry is not None:
                        target = self._covmap.cell_center(reentry[-1])
                        end_reason = "reentry_required"
                        self._coverage_stop_reason = end_reason
                        if self.diagnostics is not None:
                            self.diagnostics.event(
                                "coverage_reentry_required",
                                pos=list(pos),
                                target=list(target),
                                cell=list(reentry[-1]),
                                pending=len(
                                    self._covmap.pending_cells(
                                        dup_floor=self._coverage_dup_floor
                                    )
                                ),
                            )
                        print(
                            "  G4 LOOP: reentry_required – wracam do najblizszej komorki farmy"
                        )
                    else:
                        end_reason = "local_component_exhausted"
                        self._coverage_component_exhausted = True
                        self._coverage_stop_reason = end_reason
                        print(f"  G4 LOOP: {end_reason}")
                        break
                else:
                    if (
                        hasattr(self._covmap, "is_blocked")
                        and hasattr(self._covmap, "cell_of")
                        and self._covmap.is_blocked(self._covmap.cell_of(pos))
                    ):
                        reentry_cell = path[0]
                        if self.diagnostics is not None:
                            self.diagnostics.event(
                                "coverage_reentry_required",
                                source="g4_path",
                                pos=list(pos),
                                target=list(self._covmap.cell_center(reentry_cell)),
                                cell=list(reentry_cell),
                                pending=len(
                                    self._covmap.pending_cells(
                                        dup_floor=self._coverage_dup_floor
                                    )
                                ),
                            )
                    target = self._covmap.cell_center(
                        path[1] if len(path) > 1 else path[0]
                    )
                if target is None or self._covmap.all_done(
                    dup_floor=self._coverage_dup_floor
                ):
                    self._coverage_done = True
                    end_reason = "coverage_done"
                    print(f"  G4 LOOP: {end_reason} – farma wysycona")
                    break

                # G6: cel nieosiągalny o krok dalej (granica/no_go). NIE pchaj i NIE powtarzaj
                # w kółko („mieszanie się"): zeskanuj stąd (pierścień obejmuje sąsiednią komórkę)
                # i oznacz cel POKRYTYM, by router wybrał następny. mark≠no_go → graf bez zmian
                # (zero fragmentacji, którą dał margines).
                if self._covmap.blocked_ahead(pos, target):
                    self._covmap.record_block(pos, "stall")
                    print(f"  G6 BLOCKED: ({target[0]:.0f},{target[1]:.0f}) – skanuję tu, cel→nieosiągalny, dalej")
                    outcomes.extend(self.scan_current_view(max_shops=0))
                    self._covmap.mark_unreachable(self._covmap.cell_of(target))
                    continue

                reached = self._drive_toward_target(target)
                cell = self._covmap.cell_of(target)
                if not reached:
                    if self._handle_failed_drive(pos=pos, target=target):
                        print(
                            "  G4 LOOP: "
                            f"{self._coverage_stop_reason or 'drive_failed'}"
                        )
                        break
                    # nie dojechaliśmy do ŚRODKA (granica/margines) — zeskanuj stąd i wyklucz cel
                    # z PONOWNEGO wyboru (is_done, ale przejezdny w BFS → zero fragmentacji).
                    outcomes.extend(self.scan_current_view(max_shops=0))
                    self._covmap.mark_unreachable(cell)
                # G6: rejestruj stalled w cell jeśli scan nie znalazł nic
                if self._covmap.scans_in_cell(cell) == 0 and self._covmap.dups_in_cell(cell) == 0:
                    self._covmap.record_block(target, "stall")
            return outcomes

        # Tryb dynamiczny: navigator.next_step() zamiast sztywnej trasy
        if navigator is not None:
            step_index = 0
            while not navigator.is_finished():
                successful = sum(outcome.successful for outcome in outcomes)
                remaining = max(0, max_shops - successful) if max_shops else 0
                outcomes.extend(self.scan_current_view(max_shops=remaining))
                blocked = getattr(self.capture_pipeline, "close_blocked", False)
                stalled = getattr(self.capture_pipeline, "stall_blocked", False)
                if blocked or stalled:
                    # SCAN_DRAIN_FIX C: nie force-stepuj gdy są nietknięte sklepy
                    if hasattr(self, '_last_visible') and self.tracker.has_untried(self._last_visible):
                        print(f"  ANTI-STALL: są nietknięte sklepy – kontynuuję postój")
                        self.capture_pipeline.stall_blocked = False
                        self.capture_pipeline.stall_count = 0
                    else:
                        tag = ("close_blocked" if blocked else
                               f"stall_blocked ({self.capture_pipeline.stall_count})")
                        print(f"  ANTI-STALL: {tag} - wymuszam krok WASD by zmienic scene.")
                        self._force_step()
                        self.capture_pipeline.close_blocked = False
                        self.capture_pipeline.stall_blocked = False
                        self.capture_pipeline.stall_count = 0
                successful = sum(outcome.successful for outcome in outcomes)
                if max_shops and successful >= max_shops:
                    break
                step = navigator.next_step()
                if step is None:
                    break
                step_index += 1
                if self.diagnostics is not None:
                    self.diagnostics.record_movement(index=step_index, total=0, step=step)
                zid = getattr(navigator, "current_zone_id", None) or "?"
                print(f"  ruch {step_index}: {step.key.upper()} przez {step.duration:.2f}s ({step.kind}, strefa={zid})")
                blocked, reason, next_pos = self._movement_blocked_by_coverage(step.key)
                if blocked:
                    self._handle_preflight_block(reason, step.key, next_pos)
                    if getattr(self, "_coverage_done", False):
                        break
                    continue
                before = self.capture_pipeline.current_position
                self.movement.execute(step.key, step.duration, step.settle)
                self._step_counter += 1
                # Odometria: czytaj pozycje po kazdym kroku
                pos = self.capture_pipeline.read_current_position(step.key)
                self._learn_successful_move(step.key, before, pos)
                if self.diagnostics is not None and pos is not None:
                    self.diagnostics.event(
                        "position_read",
                        step=step_index,
                        x=pos[0],
                        y=pos[1],
                        source=self.capture_pipeline.position_source,
                    )
                if self.diagnostics is not None:
                    self.diagnostics.event("movement_done", index=step_index, key=step.key)
                if getattr(self, "_coverage_done", False):
                    break
            return outcomes

        # Tryb statyczny: predefiniowana trasa (wezyk)
        for index in range(len(route) + 1):
            successful = sum(outcome.successful for outcome in outcomes)
            remaining = max(0, max_shops - successful) if max_shops else 0
            outcomes.extend(self.scan_current_view(max_shops=remaining))
            blocked = getattr(self.capture_pipeline, "close_blocked", False)
            stalled = getattr(self.capture_pipeline, "stall_blocked", False)
            if blocked or stalled:
                tag = (
                    "close_blocked" if blocked else
                    f"stall_blocked ({self.capture_pipeline.stall_count})"
                )
                print(
                    f"  ANTI-STALL: {tag} – wymuszam krok WASD "
                    "by zmienić scenę."
                )
                self._force_step()
                self.capture_pipeline.close_blocked = False
                self.capture_pipeline.stall_blocked = False
                self.capture_pipeline.stall_count = 0
            successful = sum(outcome.successful for outcome in outcomes)
            if max_shops and successful >= max_shops:
                break
            if getattr(self, "_coverage_done", False):
                break
            if index < len(route):
                step = route[index]
                if self.diagnostics is not None:
                    self.diagnostics.record_movement(
                        index=index + 1, total=len(route), step=step
                    )
                print(
                    f"  ruch {index + 1}/{len(route)}: "
                    f"{step.key.upper()} przez {step.duration:.2f}s "
                    f"({step.kind}, pas={step.lane + 1})"
                )
                blocked, reason, next_pos = self._movement_blocked_by_coverage(step.key)
                if blocked:
                    self._handle_preflight_block(reason, step.key, next_pos)
                    if getattr(self, "_coverage_done", False):
                        break
                    continue
                before = self.capture_pipeline.current_position
                self.movement.execute(step.key, step.duration, step.settle)
                self._step_counter += 1
                # Odometria: czytaj pozycje po kazdym kroku (tryb statyczny)
                pos = self.capture_pipeline.read_current_position(step.key)
                self._learn_successful_move(step.key, before, pos)
                if self.diagnostics is not None:
                    if pos is not None:
                        self.diagnostics.event(
                            "position_read",
                            step=index + 1,
                            x=pos[0],
                            y=pos[1],
                            source=self.capture_pipeline.position_source,
                        )
                    self.diagnostics.event(
                        "movement_done",
                        index=index + 1,
                        key=step.key,
                    )
                # Guard ruchu: sprawdz zacięcie / kopertę po każdym kroku
                self._guard_movement(step.key)
        return outcomes
