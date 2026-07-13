"""Bezpieczne dziedziczenie wyniku Phase B na odroczone sloty.

Phase B celowo hoveruje tylko reprezentantow grupy identycznych ikon. Ten
modul materializuje wynik dla pozostalych slotow *dopiero* po zgodnym odczycie
reprezentantow. Nie identyfikuje przedmiotow na podstawie samej ikony.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from scanner.models import ItemObservation, ScanStatus, ShopScan


MIN_AGREEING_REPRESENTATIVES = 2


@dataclass(frozen=True, slots=True)
class GroupInheritance:
    """Rezultat rozstrzygniecia pojedynczej grupy ikon."""

    icon_group: int
    representative_slots: tuple[int, ...]
    inherited_slots: tuple[int, ...]
    applied: bool
    reason: str | None = None


def _normalise_item(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _signature(observation: ItemObservation) -> tuple[str, int, int] | None:
    """Tozsamosc oferty, ktora musi byc identyczna w calej probce grupy."""

    result = observation.validation or {}
    item = _normalise_item(result.get("item") or (observation.ai or {}).get("item"))
    unit_price = result.get("unit_price") or (observation.ai or {}).get("unit_price")
    quantity = result.get("quantity")
    if quantity is None:
        quantity = (observation.ai or {}).get("quantity")
    if not item or not isinstance(unit_price, int) or unit_price <= 0:
        return None
    if not isinstance(quantity, int) or quantity <= 0:
        return None
    return item, unit_price, quantity


def _is_deferred(observation: ItemObservation) -> bool:
    return (
        not observation.images
        and "stack_representative" in (observation.evidence or [])
    )


def apply_group_consensus(scan: ShopScan) -> tuple[GroupInheritance, ...]:
    """Dziedzicz walidacje na odroczone sloty tylko dla zgodnej grupy.

    Warunki propagacji:

    * co najmniej dwa reprezentanty maja kompletny, identyczny odczyt;
    * zaden odczytany reprezentant grupy nie przeczy zwycieskiej sygnaturze.

    Zgodnosc dwoch osobno przechwyconych reprezentantow jest potwierdzeniem
    *grupy* Phase B. Dzięki temu nie trzeba hoverowac kazdego slotu, a pojedynczy
    odczyt nadal nigdy nie moze pomnozyc swojej wartosci na caly stos. Sloty
    ``stack_representative`` pozostaja nietkniete, gdy warunek nie przejdzie.
    """

    grouped: dict[int, list[ItemObservation]] = {}
    for observation in scan.slots.values():
        if observation.icon_group is not None:
            grouped.setdefault(observation.icon_group, []).append(observation)

    decisions: list[GroupInheritance] = []
    for icon_group, members in sorted(grouped.items()):
        deferred = sorted(
            (obs for obs in members if _is_deferred(obs)), key=lambda obs: obs.slot
        )
        representatives = sorted(
            (obs for obs in members if not _is_deferred(obs)), key=lambda obs: obs.slot
        )
        signatures = {obs.slot: _signature(obs) for obs in representatives}
        usable = {slot: signature for slot, signature in signatures.items() if signature}
        rep_slots = tuple(sorted(usable))
        inherited_slots = tuple(obs.slot for obs in deferred)

        if len(usable) < MIN_AGREEING_REPRESENTATIVES:
            decisions.append(GroupInheritance(
                icon_group, rep_slots, inherited_slots, False,
                "insufficient_read_representatives",
            ))
            continue

        distinct = set(usable.values())
        if len(distinct) != 1 or len(usable) != len(representatives):
            decisions.append(GroupInheritance(
                icon_group, rep_slots, inherited_slots, False,
                "representative_mismatch",
            ))
            continue

        anchors = [
            obs for obs in representatives
            if obs.status is ScanStatus.VERIFIED and signatures[obs.slot] is not None
        ]
        anchor = min(anchors or representatives, key=lambda obs: obs.slot)
        anchor_validation = dict(anchor.validation or {})
        anchor_validation["status"] = ScanStatus.VERIFIED.value
        anchor_validation["inherited_from_slot"] = anchor.slot
        anchor_validation["group_consensus_slots"] = list(rep_slots)

        # Promujemy zgodnych reprezentantow i sloty odroczone. W efekcie
        # eksporter naturalnie policzy: liczba slotow × ilosc z dymka.
        promoted = [
            *[obs for obs in representatives if obs.status is not ScanStatus.VERIFIED],
            *deferred,
        ]
        for observation in promoted:
            observation.status = ScanStatus.VERIFIED
            observation.validation = dict(anchor_validation)
            observation.evidence = list(observation.evidence or []) + [
                f"group_consensus_of:{anchor.slot}",
            ]

        decisions.append(GroupInheritance(
            icon_group, rep_slots, tuple(obs.slot for obs in promoted), True,
        ))

    return tuple(decisions)
