# -*- coding: utf-8 -*-
"""
win_ocr.py - cienki, synchroniczny wrapper na wbudowany OCR Windows
(Windows.Media.Ocr przez PyWinRT). Dużo lepszy od Tesseracta na małym foncie
UI gry. Zwraca linie z pozycją: [{'text': str, 'box': (x0,y0,x1,y1)}, ...].

Wymaga: winrt-Windows.Media.Ocr, winrt-Windows.Graphics.Imaging,
        winrt-Windows.Storage.Streams, winrt-Windows.Globalization, winrt-runtime
oraz pakietu językowego OCR w systemie (u nas: 'pl').
"""
from __future__ import annotations

import io
import asyncio
import threading

from PIL import Image

try:
    from winrt.windows.media.ocr import OcrEngine
    from winrt.windows.globalization import Language
    from winrt.windows.graphics.imaging import BitmapDecoder
    from winrt.windows.storage.streams import (
        InMemoryRandomAccessStream, DataWriter)
    AVAILABLE = True
except Exception:
    AVAILABLE = False


_engine = None
_tls = threading.local()


def available_languages() -> list[str]:
    if not AVAILABLE:
        return []
    return [l.language_tag for l in OcrEngine.available_recognizer_languages]


def _get_engine():
    global _engine
    if _engine is None:
        for tag in ("pl", "pl-PL"):
            try:
                e = OcrEngine.try_create_from_language(Language(tag))
            except Exception:
                e = None
            if e is not None:
                _engine = e
                break
        if _engine is None:
            _engine = OcrEngine.try_create_from_user_profile_languages()
    return _engine


def _get_loop():
    loop = getattr(_tls, "loop", None)
    if loop is None:
        loop = asyncio.new_event_loop()
        _tls.loop = loop
    return loop


async def _recognize(img: Image.Image) -> list[dict]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream.get_output_stream_at(0))
    writer.write_bytes(buf.getvalue())
    await writer.store_async()
    await writer.flush_async()
    stream.seek(0)
    decoder = await BitmapDecoder.create_async(stream)
    bitmap = await decoder.get_software_bitmap_async()
    engine = _get_engine()
    if engine is None:
        return []
    result = await engine.recognize_async(bitmap)
    lines = []
    for ln in result.lines:
        words = list(ln.words)
        if not words:
            continue
        x0 = min(w.bounding_rect.x for w in words)
        y0 = min(w.bounding_rect.y for w in words)
        x1 = max(w.bounding_rect.x + w.bounding_rect.width for w in words)
        y1 = max(w.bounding_rect.y + w.bounding_rect.height for w in words)
        lines.append({"text": ln.text, "box": (x0, y0, x1, y1)})
    return lines


def recognize(img: Image.Image) -> list[dict]:
    """Synchroniczny OCR obrazu PIL -> lista linii {'text','box'}."""
    if not AVAILABLE:
        return []
    return _get_loop().run_until_complete(_recognize(img))
