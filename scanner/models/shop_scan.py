"""Model skanu sklepu i maszyna stanów potoku v10.

Ten moduł nie importuje kodu gry, OCR ani warstwy zapisu. Definiuje wyłącznie
kontrakt danych współdzielony przez capture, analizę i storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .item_observation import ItemObservation


def utc_now_iso() -> str:
    """Zwróć znacznik czasu UTC nadający się do manifestu JSON."""

    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ScanStatus(StrEnum):
    DETECTED = "detected"
    APPROACHING = "approaching"
    OPENING = "opening"
    OPENED = "opened"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    QUEUED = "queued"
    ANALYZING = "analyzing"
    PROVISIONAL = "provisional"
    VERIFIED = "verified"
    REVIEW = "review"
    FAILED = "failed"


ALLOWED_TRANSITIONS: dict[ScanStatus, frozenset[ScanStatus]] = {
    ScanStatus.DETECTED: frozenset({ScanStatus.APPROACHING, ScanStatus.FAILED}),
    ScanStatus.APPROACHING: frozenset({ScanStatus.OPENING, ScanStatus.FAILED}),
    ScanStatus.OPENING: frozenset({ScanStatus.OPENED, ScanStatus.FAILED}),
    ScanStatus.OPENED: frozenset({ScanStatus.CAPTURING, ScanStatus.FAILED}),
    ScanStatus.CAPTURING: frozenset({ScanStatus.CAPTURED, ScanStatus.FAILED}),
    ScanStatus.CAPTURED: frozenset({ScanStatus.QUEUED, ScanStatus.FAILED}),
    ScanStatus.QUEUED: frozenset({ScanStatus.ANALYZING, ScanStatus.FAILED}),
    ScanStatus.ANALYZING: frozenset(
        {ScanStatus.PROVISIONAL, ScanStatus.REVIEW, ScanStatus.FAILED}
    ),
    ScanStatus.PROVISIONAL: frozenset(
        {ScanStatus.VERIFIED, ScanStatus.REVIEW, ScanStatus.FAILED}
    ),
    ScanStatus.VERIFIED: frozenset(),
    ScanStatus.REVIEW: frozenset({ScanStatus.QUEUED, ScanStatus.ANALYZING}),
    # Recovery wybiera etap ponowienia na podstawie failed_stage.
    ScanStatus.FAILED: frozenset(
        {
            ScanStatus.APPROACHING,
            ScanStatus.OPENING,
            ScanStatus.CAPTURING,
            ScanStatus.QUEUED,
            ScanStatus.ANALYZING,
        }
    ),
}


@dataclass(slots=True)
class ScanError:
    """Błąd opatrzony etapem i informacją, czy operację można ponowić."""

    failed_stage: str
    reason: str
    retry_count: int = 0
    recoverable: bool = True
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.failed_stage.strip():
            raise ValueError("failed_stage nie może być pusty")
        if not self.reason.strip():
            raise ValueError("reason nie może być pusty")
        if self.retry_count < 0:
            raise ValueError("retry_count nie może być ujemny")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": ScanStatus.FAILED.value,
            "failed_stage": self.failed_stage,
            "reason": self.reason,
            "retry_count": self.retry_count,
            "recoverable": self.recoverable,
            "details": dict(self.details),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScanError":
        return cls(
            failed_stage=str(data["failed_stage"]),
            reason=str(data["reason"]),
            retry_count=int(data.get("retry_count", 0)),
            recoverable=bool(data.get("recoverable", True)),
            details=dict(data.get("details") or {}),
        )


@dataclass(slots=True)
class ShopScan:
    """Aktualny stan jednego skanu sklepu zapisywany w ``manifest.json``."""

    scan_id: str
    seller: str = ""
    status: ScanStatus = ScanStatus.DETECTED
    occupied_slots: int = 0
    captured_slots: int = 0
    slots: dict[int, "ItemObservation"] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    map_name: str | None = None
    channel: int | None = None
    game_position: tuple[int, int] | None = None
    screen_position: tuple[int, int] | None = None
    shop_fingerprint: str | None = None
    step_index: int | None = None  # licznik kroków WASD przy capture (kalibracja units/step)
    error: ScanError | None = None
    status_history: list[dict[str, str]] = field(default_factory=list)
    # Audyt inwentarza całym widokiem (inventory_audit): werdykt VLM-kontrolera per
    # oferta — None dopóki audyt nie pobiegł. Surowy dict do manifestu/eksportu.
    inventory_audit: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.scan_id.strip():
            raise ValueError("scan_id nie może być pusty")
        if self.occupied_slots < 0 or self.captured_slots < 0:
            raise ValueError("liczby slotów nie mogą być ujemne")
        if self.captured_slots > self.occupied_slots and self.occupied_slots:
            raise ValueError("captured_slots nie może przekraczać occupied_slots")
        self.status = ScanStatus(self.status)

    def transition(
        self,
        new_status: ScanStatus | str,
        *,
        error: ScanError | None = None,
        timestamp: str | None = None,
    ) -> None:
        """Przejdź do następnego stanu lub zgłoś niedozwolone przejście."""

        target = ScanStatus(new_status)
        if target == self.status:
            return
        if target not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(
                f"niedozwolone przejście: {self.status.value} -> {target.value}"
            )
        if target is ScanStatus.FAILED and error is None:
            raise ValueError("przejście do FAILED wymaga ScanError")
        if target is not ScanStatus.FAILED and error is not None:
            raise ValueError("ScanError można przypisać tylko do FAILED")

        changed_at = timestamp or utc_now_iso()
        self.status_history.append(
            {
                "from": self.status.value,
                "to": target.value,
                "timestamp": changed_at,
            }
        )
        self.status = target
        self.error = error
        self.updated_at = changed_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scan_id": self.scan_id,
            "seller": self.seller,
            "status": self.status.value,
            "occupied_slots": self.occupied_slots,
            "captured_slots": self.captured_slots,
            "slots": {
                str(slot): observation.to_dict()
                for slot, observation in sorted(self.slots.items())
            },
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "map_name": self.map_name,
            "channel": self.channel,
            "game_position": list(self.game_position) if self.game_position else None,
            "screen_position": (
                list(self.screen_position) if self.screen_position else None
            ),
            "shop_fingerprint": self.shop_fingerprint,
            "step_index": self.step_index,
            "error": self.error.to_dict() if self.error else None,
            "status_history": list(self.status_history),
            "inventory_audit": (
                dict(self.inventory_audit) if self.inventory_audit is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShopScan":
        from .item_observation import ItemObservation

        slots = {
            int(slot): ItemObservation.from_dict(observation)
            for slot, observation in (data.get("slots") or {}).items()
        }
        game_position = data.get("game_position")
        screen_position = data.get("screen_position")
        return cls(
            scan_id=str(data["scan_id"]),
            seller=str(data.get("seller") or ""),
            status=ScanStatus(data.get("status", ScanStatus.DETECTED.value)),
            occupied_slots=int(data.get("occupied_slots", 0)),
            captured_slots=int(data.get("captured_slots", 0)),
            slots=slots,
            created_at=str(data.get("created_at") or utc_now_iso()),
            updated_at=str(data.get("updated_at") or utc_now_iso()),
            map_name=data.get("map_name"),
            channel=data.get("channel"),
            game_position=tuple(game_position) if game_position else None,
            screen_position=tuple(screen_position) if screen_position else None,
            shop_fingerprint=data.get("shop_fingerprint"),
            step_index=data.get("step_index"),
            error=ScanError.from_dict(data["error"]) if data.get("error") else None,
            status_history=list(data.get("status_history") or []),
            inventory_audit=(
                dict(data["inventory_audit"])
                if data.get("inventory_audit") is not None
                else None
            ),
        )

