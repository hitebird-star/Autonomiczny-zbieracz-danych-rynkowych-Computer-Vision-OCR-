"""Serwer web Atlasu (MVP: stdlib http.server + SSE) — live podgląd mapy + ustawienia.

Zero zewnętrznych zależności (frontend to HTML/JS/canvas, backend wymienialny na FastAPI
później). Serwuje `web/` + API:
    GET  /api/atlas    → payload mapy (granica, sklepy, statystyki)
    GET  /api/config   → AtlasConfig
    POST /api/config   → zapis ustawień (atlas_config.json)
    POST /api/reload   → przeładuj dane z rejestru/kalibracji
    GET  /api/events   → SSE (live odświeżanie; live-feed woła `AtlasState.push_snapshot`)

Read-only wobec obecnego systemu. Nie steruje botem. Live-feed Codexa (gdy gotowy)
karmi `push_snapshot(FrameSnapshot)` — dopóki brak kalibracji, punkty live nie są rzutowane.
"""

from __future__ import annotations

import json
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from scanner.atlas import ingest
from scanner.atlas.atlas import MarketAtlas
from scanner.atlas.calibration import GroundProjection
from scanner.atlas.config import ATLAS_CONFIG_PATH, AtlasConfig
from scanner.atlas.render import build_render_payload

WEB_ROOT = Path(__file__).parent / "web"
_CONTENT_TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


def build_atlas_from_sources(config: AtlasConfig) -> MarketAtlas:
    """Zbuduj mapę: zapisany atlas (jeśli jest) + świeży rejestr + granica. Read-only."""
    atlas = MarketAtlas.load(config.atlas_store) or MarketAtlas(
        dedup_radius_units=config.dedup_radius_units,
        position_ema=config.position_ema,
    )
    atlas.upsert_from_registry(ingest.load_registry(config.shops_registry))
    if not atlas.boundary:
        atlas.boundary = ingest.load_boundary(config.boundary_file)
    return atlas


class AtlasState:
    """Współdzielony stan serwera (mapa + kalibracja + punkty live)."""

    def __init__(self, config: AtlasConfig, *, config_path: str = ATLAS_CONFIG_PATH) -> None:
        self.config = config
        self.config_path = config_path
        self._lock = threading.RLock()
        self.atlas = build_atlas_from_sources(config)
        self.projection = GroundProjection.load(config.calibration_file)
        self.live_points: list[dict[str, Any]] = []
        self._version = 1

    def payload(self) -> dict[str, Any]:
        with self._lock:
            p = build_render_payload(self.atlas)
            p["version"] = self._version
            p["has_calibration"] = self.projection is not None
            p["transform_version"] = self.projection.version if self.projection else ""
            p["live"] = list(self.live_points)
            return p

    def reload(self) -> None:
        with self._lock:
            self.atlas = build_atlas_from_sources(self.config)
            self.projection = GroundProjection.load(self.config.calibration_file)
            self._version += 1

    def update_config(self, data: dict[str, Any]) -> AtlasConfig:
        with self._lock:
            merged = {**self.config.to_dict(), **(data or {})}
            new = AtlasConfig.from_dict(merged)
            new.save(self.config_path)
            self.config = new
            self._version += 1
            return new

    def push_snapshot(self, snapshot: Any) -> None:
        """Live-feed → rzut widocznych sklepów na mapę (tylko gdy jest kalibracja)."""
        with self._lock:
            if self.projection is None or getattr(snapshot, "player_game", None) is None:
                return
            pts = []
            for px in getattr(snapshot, "shops_screen", []) or []:
                gx, gy = self.projection.screen_to_game(snapshot.player_game, px)
                pts.append({"x": gx, "y": gy})
            self.live_points = pts
            self._version += 1


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, state: AtlasState, **kwargs) -> None:
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, *args) -> None:  # cisza w konsoli
        pass

    # ---- helpers ----
    def _json(self, obj: Any, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name: str) -> None:
        safe = (WEB_ROOT / name).resolve()
        if WEB_ROOT.resolve() not in safe.parents and safe != WEB_ROOT.resolve():
            self.send_error(403)
            return
        if not safe.is_file():
            self.send_error(404)
            return
        data = safe.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _CONTENT_TYPES.get(safe.suffix, "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    # ---- routing ----
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._static("index.html")
        elif path in ("/app.js", "/style.css"):
            self._static(path.lstrip("/"))
        elif path == "/api/atlas":
            self._json(self.state.payload())
        elif path == "/api/config":
            self._json(self.state.config.to_dict())
        elif path == "/api/events":
            self._sse()
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        if path == "/api/config":
            try:
                data = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._json({"error": "zły JSON"}, 400)
                return
            new = self.state.update_config(data)
            self._json(new.to_dict())
        elif path == "/api/reload":
            self.state.reload()
            self._json({"ok": True, "version": self.state._version})
        else:
            self.send_error(404)

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        last = -1
        interval = max(0.1, float(self.state.config.sse_interval_s))
        try:
            while True:
                ver = self.state._version
                if ver != last:
                    payload = json.dumps(self.state.payload(), ensure_ascii=False)
                    self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last = ver
                else:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                threading.Event().wait(interval)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return


def make_server(config: AtlasConfig | None = None) -> tuple[ThreadingHTTPServer, AtlasState]:
    config = config or AtlasConfig.load()
    state = AtlasState(config)
    handler = partial(_Handler, state=state)
    httpd = ThreadingHTTPServer((config.server_host, config.server_port), handler)
    return httpd, state


def serve(config: AtlasConfig | None = None) -> None:
    httpd, state = make_server(config)
    host, port = httpd.server_address
    print(f"Atlas UI: http://{host}:{port}  (sklepów: {len(state.atlas.shops())}, "
          f"kalibracja: {'jest' if state.projection else 'brak'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nAtlas UI: stop")
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    serve()
