"""Warstwa analizy (działka Claude): reader VLM, OCR pomocniczy, walidator.

Czyste wejście/wyjście na plikach — bez importów z gry. Wpina się w
`scanner.pipeline.AnalysisWorker` przez `VlmAnalysisEngine`.
"""

from . import ollama_reader, validator, windows_ocr
from .engine import VlmAnalysisEngine

__all__ = ["VlmAnalysisEngine", "ollama_reader", "validator", "windows_ocr"]
