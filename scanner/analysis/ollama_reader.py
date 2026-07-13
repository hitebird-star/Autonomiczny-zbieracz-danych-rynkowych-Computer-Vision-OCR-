"""Czytnik dymków oparty o lokalny model wizyjny (qwen3-vl:8b-instruct).

CZYSTA funkcja wg D-016/1: obraz -> JSON. ZERO importów z gry/capture/storage.
Konfiguracja i prompt są przypięte w AI_DATA_CONTRACT.md (sekcje 1-3); ten plik
jest produkcyjnym źródłem prawdy, `bench/bakeoff.py` go odzwierciedla.

Decyzja modelu: D-004 (instruct, nie thinking). Parametry: D-005/D-017.
Reguła `kk` w promptcie: D-011 (kk=miliony, NIE miliardy).
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request
from typing import Any

from PIL import Image

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL = "qwen3-vl:8b-instruct"
NUM_CTX = 8192          # jawnie (D-005); inaczej Ollama wraca do 131072 -> CPU
NUM_PREDICT = 512       # D-017: max 59 tokenow w benchmarku, 0 uciec
TEMPERATURE = 0
THINK = False           # D-004: instruct nie mysli
KEEP_ALIVE = "10m"

# Prompt systemowy (pinned — reguła kk jest krytyczna, D-011).
SYSTEM = (
    "Jestes precyzyjnym czytnikiem dymkow przedmiotow z gry Metin2 (serwer "
    "Glevia). Na obrazie jest dymek sprzedazy w sklepie. Odczytaj DOKLADNIE "
    "i zwroc WYLACZNIE JSON, bez komentarzy.\n"
    "Pola:\n"
    "- item: nazwa przedmiotu = gorna linia tytulu (zlota/pogrubiona), "
    "NIE opis pod nia.\n"
    "- total_price: liczba pod naglowkiem '[Cena sprzedazy]'. Zignoruj ikone "
    "monety obok liczby. Usun kropki/spacje separatorow tysiecy. Zwroc "
    "calkowita liczbe Yang.\n"
    "- unit_price: z linii 'X Yang za sztuke'. UWAGA NA SKROTY: k = tysiace "
    "(x1000), kk = MILIONY (x1000000), kkk = MILIARDY (x1000000000). Przyklady: "
    "'23kk'=23000000 (dwadziescia trzy MILIONY, NIE miliardy), '350k'=350000, "
    "'1.7kk'=1700000, '7.0kk'=7000000, '11kk'=11000000. NIGDY nie mnoz kk przez "
    "miliard. Jesli cena jest w 'Won': 1 Won = 1000000000 Yang.\n"
    "- quantity: liczba sztuk. Jesli niewidoczna, policz total_price / "
    "unit_price.\n"
    "Jesli pola nie da sie odczytac, ustaw null. Nie zgaduj liczb."
)

# Schemat wymuszany przez Ollame (D-003 kontraktu).
SCHEMA = {
    "type": "object",
    "properties": {
        "item": {"type": ["string", "null"]},
        "total_price": {"type": ["integer", "null"]},
        "unit_price": {"type": ["integer", "null"]},
        "quantity": {"type": ["integer", "null"]},
    },
    "required": ["item", "total_price", "unit_price", "quantity"],
}

# Audyt inwentarza: VLM patrzy na obraz CAŁEGO sklepu (siatka ikon) i liczy, ile
# slotów zajmuje każdy odrębny przedmiot. To KONTROLER liczby stacków, nie wyrocznia
# (8B VLM liczący drobne ikony bywa zawodny — `inventory_audit.reconcile` flaguje
# rozjazd, nigdy nie nadpisuje). Bez cen: w widoku całości dymki są zamknięte.
SHOP_INVENTORY_SYSTEM = (
    "Patrzysz na CALE okno sklepu w grze Metin2 (serwer Glevia): siatka komorek, "
    "kazda z ikona przedmiotu albo pusta. Policz, ILE KOMOREK (slotow) zajmuje "
    "kazdy ODREBNY przedmiot. Te same ikony to ten sam przedmiot. Liczba w prawym "
    "dolnym rogu ikony to rozmiar stosu w JEDNEJ komorce - NIE dodawaj jej do liczby "
    "slotow (liczymy komorki, nie sztuki). Puste komorki pomijaj. Zwroc WYLACZNIE "
    "JSON: {\"items\": [{\"item\": nazwa, \"slots\": liczba_komorek}]}. Nazw nie "
    "zgaduj - jesli nie znasz nazwy przedmiotu, opisz ja krotko. Nie wymyslaj "
    "przedmiotow, ktorych nie widac."
)

SHOP_INVENTORY_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item": {"type": ["string", "null"]},
                    "slots": {"type": ["integer", "null"]},
                },
                "required": ["item", "slots"],
            },
        },
    },
    "required": ["items"],
}


def _img_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _chat(payload: dict, *, url: str, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url + "/api/chat", data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def read_tooltip(
    image: Image.Image,
    *,
    url: str = OLLAMA_URL,
    model: str = MODEL,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """Odczytaj jeden dymek. Zwraca dict:

        {item, total_price, unit_price, quantity,   # pola wg kontraktu
         source, raw, seconds, eval_count, error}

    Liczby przeliczone do int (kk/k/Won rozwija model). Pole nieczytelne = None.
    Nigdy nie rzuca wyjatkiem na blad sieci/parsowania — zwraca error.
    """
    out: dict[str, Any] = {
        "item": None, "total_price": None, "unit_price": None, "quantity": None,
        "source": "vlm", "raw": "", "seconds": None, "eval_count": None,
        "error": None,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "Przeanalizuj ten dymek sprzedazy i zwroc JSON.",
                "images": [_img_b64(image)],
            },
        ],
        "stream": False,
        "think": THINK,
        "format": SCHEMA,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    start = time.perf_counter()
    try:
        response = _chat(payload, url=url, timeout=timeout)
        out["seconds"] = round(time.perf_counter() - start, 3)
        content = (response.get("message") or {}).get("content", "")
        out["raw"] = content
        out["eval_count"] = response.get("eval_count")
        parsed = json.loads(content)
        out["item"] = parsed.get("item")
        out["total_price"] = _as_int(parsed.get("total_price"))
        out["unit_price"] = _as_int(parsed.get("unit_price"))
        out["quantity"] = _as_int(parsed.get("quantity"))
    except Exception as exc:  # sieć, JSON, schema — nie wywracamy analizy
        out["error"] = f"{type(exc).__name__}: {exc}"
        if out["seconds"] is None:
            out["seconds"] = round(time.perf_counter() - start, 3)
    return out


def read_shop_inventory(
    image: Image.Image,
    *,
    url: str = OLLAMA_URL,
    model: str = MODEL,
    timeout: float = 180.0,
) -> dict[str, Any]:
    """Policz sloty per przedmiot z obrazu CAŁEGO sklepu. Zwraca dict:

        {items: [{item, slots}], source, raw, seconds, eval_count, error}

    KONTROLER, nie wyrocznia: wynik karmi `inventory_audit.reconcile`, który
    flaguje rozjazd z pipeline'em (nigdy nie nadpisuje CSV). Liczenie wielu
    drobnych ikon przez 8B VLM bywa zawodne — dlatego tylko potwierdza/flaguje.
    Nigdy nie rzuca wyjątkiem; błąd sieci/JSON zwraca w polu `error`.
    """
    out: dict[str, Any] = {
        "items": [], "source": "vlm_shop", "raw": "", "seconds": None,
        "eval_count": None, "error": None,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SHOP_INVENTORY_SYSTEM},
            {
                "role": "user",
                "content": "Policz sloty kazdego przedmiotu w tym sklepie i zwroc JSON.",
                "images": [_img_b64(image)],
            },
        ],
        "stream": False,
        "think": THINK,
        "format": SHOP_INVENTORY_SCHEMA,
        "keep_alive": KEEP_ALIVE,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    start = time.perf_counter()
    try:
        response = _chat(payload, url=url, timeout=timeout)
        out["seconds"] = round(time.perf_counter() - start, 3)
        content = (response.get("message") or {}).get("content", "")
        out["raw"] = content
        out["eval_count"] = response.get("eval_count")
        parsed = json.loads(content)
        items = []
        for entry in parsed.get("items") or []:
            if not isinstance(entry, dict):
                continue
            items.append({
                "item": entry.get("item"),
                "slots": _as_int(entry.get("slots")),
            })
        out["items"] = items
    except Exception as exc:  # sieć, JSON, schema — nie wywracamy analizy
        out["error"] = f"{type(exc).__name__}: {exc}"
        if out["seconds"] is None:
            out["seconds"] = round(time.perf_counter() - start, 3)
    return out


def available(url: str = OLLAMA_URL, model: str = MODEL) -> bool:
    """Czy Ollama odpowiada i model jest dostępny (do bramki startowej)."""
    try:
        request = urllib.request.Request(url + "/api/tags")
        with urllib.request.urlopen(request, timeout=5) as response:
            tags = json.loads(response.read().decode("utf-8"))
        names = {entry.get("name", "") for entry in tags.get("models", [])}
        return model in names or any(model.split(":")[0] in n for n in names)
    except Exception:
        return False
