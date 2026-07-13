"""Polityka reakcji na awarie etapów autonomicznej pętli."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from scanner.models import ScanError


class RecoveryAction(StrEnum):
    RETRY_INTERACTION = "retry_interaction"
    RETRY_CAPTURE = "retry_capture"
    RETRY_ANALYSIS = "retry_analysis"
    REFOCUS = "refocus"
    MOVE_AWAY = "move_away"
    SKIP = "skip"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    action: RecoveryAction
    delay: float = 0.0
    terminal: bool = False


class RecoveryPolicy:
    def __init__(self, *, max_game_retries: int = 2, max_analysis_retries: int = 1):
        self.max_game_retries = max_game_retries
        self.max_analysis_retries = max_analysis_retries

    def decide(self, error: ScanError) -> RecoveryDecision:
        if not error.recoverable:
            return RecoveryDecision(RecoveryAction.SKIP, terminal=True)
        if error.reason == "window_not_focused":
            return RecoveryDecision(RecoveryAction.REFOCUS, delay=0.2)
        if error.reason in {"character_stuck", "no_progress"}:
            return RecoveryDecision(RecoveryAction.MOVE_AWAY, delay=0.3)
        if error.failed_stage in {"approaching", "opening"}:
            if error.retry_count < self.max_game_retries:
                return RecoveryDecision(RecoveryAction.RETRY_INTERACTION, delay=0.25)
            return RecoveryDecision(RecoveryAction.SKIP, terminal=True)
        if error.failed_stage == "capturing":
            if error.retry_count < self.max_game_retries:
                return RecoveryDecision(RecoveryAction.RETRY_CAPTURE, delay=0.15)
            return RecoveryDecision(RecoveryAction.SKIP, terminal=True)
        if error.failed_stage == "analyzing":
            if error.retry_count < self.max_analysis_retries:
                return RecoveryDecision(RecoveryAction.RETRY_ANALYSIS, delay=0.5)
            return RecoveryDecision(RecoveryAction.REVIEW, terminal=True)
        return RecoveryDecision(RecoveryAction.SKIP, terminal=True)
