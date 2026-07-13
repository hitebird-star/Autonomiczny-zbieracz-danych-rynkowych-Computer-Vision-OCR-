"""Model obserwacji pojedynczego slotu sklepu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .shop_scan import ScanError, ScanStatus


@dataclass(slots=True)
class ItemObservation:
    """Dane slotu od capture aż po walidację.

    Pola ``ocr``, ``ai`` i ``validation`` zachowują surowe wyniki poszczególnych
    etapów. Dzięki temu storage nie traci materiału potrzebnego do audytu.
    """

    slot: int
    row: int
    column: int
    images: list[str] = field(default_factory=list)
    icon_group: int | None = None
    status: ScanStatus = ScanStatus.CAPTURED
    ocr: dict[str, Any] | None = None
    ai: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    evidence: list[str] = field(default_factory=list)
    error: ScanError | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.slot <= 99:
            raise ValueError("slot musi mieścić się w zakresie 0..99")
        if not 0 <= self.row <= 9 or not 0 <= self.column <= 9:
            raise ValueError("row i column muszą mieścić się w zakresie 0..9")
        self.status = ScanStatus(self.status)
        if self.status not in {
            ScanStatus.CAPTURED,
            ScanStatus.QUEUED,
            ScanStatus.ANALYZING,
            ScanStatus.PROVISIONAL,
            ScanStatus.VERIFIED,
            ScanStatus.REVIEW,
            ScanStatus.FAILED,
        }:
            raise ValueError(f"nieprawidłowy status obserwacji: {self.status.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "row": self.row,
            "column": self.column,
            "images": list(self.images),
            "icon_group": self.icon_group,
            "status": self.status.value,
            "ocr": dict(self.ocr) if self.ocr is not None else None,
            "ai": dict(self.ai) if self.ai is not None else None,
            "validation": (
                dict(self.validation) if self.validation is not None else None
            ),
            "evidence": list(self.evidence),
            "error": self.error.to_dict() if self.error else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ItemObservation":
        return cls(
            slot=int(data["slot"]),
            row=int(data["row"]),
            column=int(data["column"]),
            images=list(data.get("images") or []),
            icon_group=data.get("icon_group"),
            status=ScanStatus(data.get("status", ScanStatus.CAPTURED.value)),
            ocr=dict(data["ocr"]) if data.get("ocr") is not None else None,
            ai=dict(data["ai"]) if data.get("ai") is not None else None,
            validation=(
                dict(data["validation"])
                if data.get("validation") is not None
                else None
            ),
            evidence=list(data.get("evidence") or []),
            error=ScanError.from_dict(data["error"]) if data.get("error") else None,
        )

