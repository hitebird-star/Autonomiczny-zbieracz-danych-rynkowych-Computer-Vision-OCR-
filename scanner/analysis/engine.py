"""Silnik analizy spinający reader VLM + OCR + walidator w `AnalysisEngine`.

Wpina się w `scanner.pipeline.AnalysisWorker` przez wąski interfejs:
    analyze(scan: ShopScan, repository: ScanRepository) -> ShopScan

Dla każdego zajętego slotu:
  1. VLM (klatka 1)            -> obs.ai  (nazwa, total, unit)
  2. VLM (klatka 2, jeśli jest) + OCR (klatka 1) -> niezależne potwierdzenia
  3. walidator                 -> obs.validation + status (provisional/verified/review)
Statusy slotów agregują się do statusu skanu. CSV bierze tylko VERIFIED.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from scanner.models import ItemObservation, ScanError, ScanStatus, ShopScan
from scanner.storage import ScanRepository

from . import group_consensus, inventory_audit, ollama_reader, validator, windows_ocr

# Statusy slotów, które analizujemy (VERIFIED zostawiamy w spokoju).
_TO_ANALYZE = {
    ScanStatus.CAPTURED, ScanStatus.QUEUED, ScanStatus.ANALYZING,
    ScanStatus.PROVISIONAL, ScanStatus.REVIEW, ScanStatus.FAILED,
}

_STATUS_MAP = {
    validator.PROVISIONAL: ScanStatus.PROVISIONAL,
    validator.VERIFIED: ScanStatus.VERIFIED,
    validator.REVIEW: ScanStatus.REVIEW,
}


class VlmAnalysisEngine:
    """`AnalysisEngine` oparty o qwen3-vl:8b-instruct + OCR jako potwierdzenie."""

    def __init__(
        self,
        *,
        use_ocr: bool = True,
        use_second_frame: bool = True,
        use_shop_audit: bool = True,
        use_vlm_shop_audit: bool = False,
        shop_inventory_reader=None,
    ) -> None:
        self.use_ocr = use_ocr
        self.use_second_frame = use_second_frame
        # Audyt kompletności sklepu — DETERMINISTYCZNY (occupied/assigned/unassigned),
        # bez VLM, domyślnie ON (tani). VLM całego widoku jest EKSPERYMENTALNY i tylko
        # dorzuca surową diagnostykę nazw (NIE autorytet — nazwy nie ma na shop.png).
        self.use_shop_audit = use_shop_audit
        self.use_vlm_shop_audit = use_vlm_shop_audit
        self._shop_inventory_reader = shop_inventory_reader

    # --- interfejs AnalysisEngine -------------------------------------------
    def analyze(self, scan: ShopScan, repository: ScanRepository) -> ShopScan:
        base = repository.scan_dir(scan.scan_id)
        for slot_no, obs in sorted(scan.slots.items()):
            if obs.status is ScanStatus.VERIFIED:
                continue
            if obs.status not in _TO_ANALYZE:
                continue
            try:
                self._analyze_slot(scan, repository, obs, base)
            except Exception as exc:  # jeden slot nie wywraca całego sklepu
                obs.error = ScanError(
                    failed_stage="analyzing",
                    reason="slot_analysis_exception",
                    recoverable=True,
                    details={"type": type(exc).__name__, "message": str(exc)},
                )
                obs.status = ScanStatus.FAILED
                repository.append_event(
                    scan.scan_id, "slot_failed", slot=slot_no,
                    error=obs.error.to_dict(),
                )
        self._apply_group_consensus(scan, repository)
        if self.use_shop_audit:
            try:
                self._audit_inventory(scan, repository, base)
            except Exception as exc:  # audyt nie wywraca analizy slotów
                repository.append_event(
                    scan.scan_id, "inventory_audit_error",
                    type=type(exc).__name__, message=str(exc),
                )
        self._finalize_scan(scan)
        return scan

    @staticmethod
    def _apply_group_consensus(scan, repository) -> None:
        """Materializuj bezpiecznie odroczone sloty Phase B.

        Samo podobienstwo ikony nie wystarcza. Pomocnik wymaga dwoch zgodnych
        reprezentantow, zanim oznaczy cala grupe jako VERIFIED.
        """

        for decision in group_consensus.apply_group_consensus(scan):
            if decision.applied:
                repository.append_event(
                    scan.scan_id,
                    "group_consensus_applied",
                    icon_group=decision.icon_group,
                    representatives=list(decision.representative_slots),
                    inherited_slots=list(decision.inherited_slots),
                    inherited_count=len(decision.inherited_slots),
                )
            else:
                repository.append_event(
                    scan.scan_id,
                    "group_consensus_withheld",
                    icon_group=decision.icon_group,
                    representatives=list(decision.representative_slots),
                    deferred_slots=list(decision.inherited_slots),
                    reason=decision.reason,
                )

    # --- pojedynczy slot -----------------------------------------------------
    def _analyze_slot(self, scan, repository, obs: ItemObservation, base) -> None:
        # Bloker #2: popup-obs z buy_popup ma już nazwę+cenę, promuj do VERIFIED bez VLM
        if self._is_buy_popup(obs):
            self._verify_from_popup(scan, repository, obs)
            return
        frames = self._load_frames(base, obs.images)
        if not frames:
            obs.validation = {"status": validator.REVIEW, "reason": "no_images"}
            obs.status = ScanStatus.REVIEW
            repository.append_event(
                scan.scan_id, "slot_review", slot=obs.slot, reason="no_images")
            return

        primary = ollama_reader.read_tooltip(frames[0])
        obs.ai = {
            "item": primary.get("item"),
            "total_price": primary.get("total_price"),
            "unit_price": primary.get("unit_price"),
            "quantity": primary.get("quantity"),
            "source": "vlm",
            "seconds": primary.get("seconds"),
            "error": primary.get("error"),
        }

        confirmations: list[dict[str, Any]] = []

        if self.use_second_frame and len(frames) >= 2:
            second = ollama_reader.read_tooltip(frames[1])
            obs.ai["frame_2"] = {
                "total_price": second.get("total_price"),
                "unit_price": second.get("unit_price"),
                "quantity": second.get("quantity"),
            }
            confirmations.append({
                "source": "vlm_frame_2",
                "total_price": second.get("total_price"),
                "unit_price": second.get("unit_price"),
                "quantity": second.get("quantity"),
            })

        if self.use_ocr:
            ocr = windows_ocr.read_tooltip(frames[0])
            obs.ocr = {
                "total_price": ocr.get("total_price"),
                "unit_price": ocr.get("unit_price"),
                "error": ocr.get("error"),
            }
            confirmations.append({
                "source": "ocr",
                "total_price": ocr.get("total_price"),
                "unit_price": ocr.get("unit_price"),
                "quantity": ocr.get("quantity"),
            })

        result = validator.validate(obs.ai, confirmations=confirmations)
        obs.validation = result
        obs.evidence = list(result.get("evidence") or [])
        obs.status = _STATUS_MAP[result["status"]]
        repository.append_event(
            scan.scan_id, f"slot_{result['status']}", slot=obs.slot,
            reason=result.get("reason"), evidence=obs.evidence,
        )

    @staticmethod
    def _is_buy_popup(obs) -> bool:
        v = obs.validation or {}
        return (v.get("source") == "buy_popup" or "buy_popup" in (obs.evidence or [])) \
               and bool(v.get("item") or "buy_popup" in (obs.evidence or []))

    def _verify_from_popup(self, scan, repository, obs) -> None:
        v = obs.validation or {}
        item = (v.get("item") or "").strip()
        unit = v.get("unit_price")
        qty = v.get("quantity")
        if not item or not isinstance(unit, int) or unit <= 0 or not isinstance(qty, int) or qty <= 0:
            obs.validation = {"status": validator.REVIEW, "reason": "buy_popup_incomplete", "source": "buy_popup"}
            obs.status = ScanStatus.REVIEW
            repository.append_event(scan.scan_id, "slot_review", slot=obs.slot, reason="buy_popup_incomplete")
            return
        obs.validation = {"status": validator.VERIFIED, "source": "buy_popup",
                          "item": item, "unit_price": unit, "quantity": qty, "evidence": ["buy_popup"]}
        obs.evidence = ["buy_popup"]
        obs.status = ScanStatus.VERIFIED
        repository.append_event(scan.scan_id, "slot_verified", slot=obs.slot,
                                reason="buy_popup", evidence=["buy_popup"])

    # --- audyt kompletności sklepu (DETERMINISTYCZNY, bez VLM) --------------
    def _audit_inventory(self, scan, repository, base) -> None:
        """Policz kompletność: occupied vs bezpiecznie przypisane (= unassigned).

        DETERMINISTYCZNY, bez VLM. Biegnie PO `group_consensus`, więc przypisane
        sloty obejmują odziedziczone przez zgodne reprezentanty. Nie zgaduje nazw
        nierozpoznanych grup — uczciwy wynik to `unassigned_slots`. VLM całego widoku
        odpala się TYLKO za `use_vlm_shop_audit` i dorzuca surową diagnostykę nazw
        (nigdy autorytet — nazwy nie ma na shop.png) ([[inventory-audit]]).
        """
        offers = self._build_offer_tallies(scan)
        audit = inventory_audit.audit_shop(scan.occupied_slots, offers)

        if self.use_vlm_shop_audit:
            audit["vlm"] = self._vlm_shop_diagnostics(base, offers)

        scan.inventory_audit = audit
        if audit["audit_status"] == inventory_audit.PARTIAL:
            repository.append_event(
                scan.scan_id, "shop_partial",
                occupied_slots=audit["occupied_slots"],
                pipeline_stack_count=audit["pipeline_stack_count"],
                unassigned_slots=audit["unassigned_slots"],
            )
        repository.append_event(
            scan.scan_id, "inventory_audited",
            occupied_slots=audit["occupied_slots"],
            pipeline_stack_count=audit["pipeline_stack_count"],
            unassigned_slots=audit["unassigned_slots"],
            audit_status=audit["audit_status"],
            offers=len(offers),
        )

    def _vlm_shop_diagnostics(self, base, offers) -> dict[str, Any]:
        """EKSPERYMENTALNE: surowy odczyt VLM całego shop.png — tylko podgląd."""
        image = self._load_shop_image(base)
        reader = self._shop_inventory_reader or ollama_reader.read_shop_inventory
        result = reader(image) if image is not None else {
            "items": [], "error": "no_shop_image", "seconds": None}
        counts = inventory_audit.build_vlm_counts(result.get("items") or [])
        diag = inventory_audit.vlm_diagnostics(offers, counts)
        return {
            "available": image is not None and not result.get("error"),
            "error": result.get("error"),
            "seconds": result.get("seconds"),
            "items": [[n, c] for n, c in sorted(counts.items())],
            "vlm_only": [[n, c] for n, c in diag["vlm_only"]],
            "matched": diag["matched"],
        }

    @staticmethod
    def _build_offer_tallies(scan) -> list[dict[str, Any]]:
        """Zbierz oferty po TOŻSAMOŚCI (nazwa z dymka) z slotów VERIFIED.

        Wołane PO `group_consensus`, więc VERIFIED obejmuje też sloty odziedziczone
        przez zgodne reprezentanty. ``stack_count`` = sloty potwierdzonej grupy tej
        tożsamości; ``quantity`` = suma sztuk z dymka. Nieprzypisane sloty NIE trafiają
        tu pod żadną nazwą — zlicza je `audit_shop` jako `unassigned_slots`.
        """
        tallies: dict[str, dict[str, Any]] = {}
        for obs in scan.slots.values():
            if obs.status is not ScanStatus.VERIFIED:
                continue
            validation = obs.validation or {}
            name = (validation.get("item") or (obs.ai or {}).get("item") or "")
            name = name.strip() if isinstance(name, str) else ""
            if not name:
                continue
            key = inventory_audit.normalize_item(name)
            rec = tallies.setdefault(key, {"item": name, "stack_count": 0, "quantity": 0})
            rec["stack_count"] += 1
            qty = validation.get("quantity")
            if isinstance(qty, int) and qty > 0:
                rec["quantity"] += qty
        return list(tallies.values())

    @staticmethod
    def _load_shop_image(base) -> Image.Image | None:
        path = base / "shop.png"
        try:
            with Image.open(path) as handle:
                return handle.convert("RGB")
        except (OSError, ValueError):
            return None

    @staticmethod
    def _load_frames(base, images: list[str]) -> list[Image.Image]:
        frames = []
        for relative in images:
            path = base / relative
            try:
                with Image.open(path) as handle:
                    frames.append(handle.convert("RGB"))
            except (OSError, ValueError):
                continue
        return frames

    # --- agregacja na poziom skanu ------------------------------------------
    @staticmethod
    def _finalize_scan(scan: ShopScan) -> None:
        statuses = [obs.status for obs in scan.slots.values()]
        occupied = len(statuses)
        verified = sum(s is ScanStatus.VERIFIED for s in statuses)
        usable = sum(
            s in {ScanStatus.VERIFIED, ScanStatus.PROVISIONAL} for s in statuses)

        if scan.status is not ScanStatus.ANALYZING:
            return
        if occupied and verified == occupied:
            scan.transition(ScanStatus.PROVISIONAL)
            scan.transition(ScanStatus.VERIFIED)
        elif usable > 0:
            scan.transition(ScanStatus.PROVISIONAL)
        else:
            scan.transition(ScanStatus.REVIEW)
