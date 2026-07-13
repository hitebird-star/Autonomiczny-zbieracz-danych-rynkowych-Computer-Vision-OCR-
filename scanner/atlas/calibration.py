"""Fotogrametria Atlasu: rzut ekran↔teren dla stałej kamery izometrycznej.

Model: płaska ziemia, kamera podąża za postacią. Offset pikselowy sklepu od kotwicy
(punktu renderu postaci) mapuje się afinicznie na offset w jednostkach gry macierzą
`A` (stałą, bo kamera fixed):

    g_shop = g_player + A · (p_shop − p_anchor)

Kalibracja z KONTROLOWANEGO ruchu: statyczny sklep śledzony przez ruch postaci spełnia

    A · Δp = −Δg          (świat przesuwa się przeciwnie do postaci)

≥2 nierównoległe kierunki ruchu + odporny fit (odrzut outlierów po MAD) → czyste `A`.
Ten moduł jest OFFLINE-CZYSTY i testowalny bez gry: przyjmuje gotowe `MoveObservation`
(produkuje je Codex w `calibrator_live.py`), zwraca `GroundProjection` + `FitResult`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from scanner.atlas.contracts import MoveObservation, Point2 as Vec

DEFAULT_ANCHOR: Vec = (960.0, 540.0)  # render postaci ~środek okna 1920×1080 (kalibrowalne)

OPPOSITE_KEYS = {"w": "s", "s": "w", "a": "d", "d": "a"}


def _median_point(points: Sequence[Vec]) -> Vec:
    xs = sorted(float(p[0]) for p in points)
    ys = sorted(float(p[1]) for p in points)
    n = len(xs)
    m = n // 2
    if n % 2:
        return (xs[m], ys[m])
    return ((xs[m - 1] + xs[m]) / 2.0, (ys[m - 1] + ys[m]) / 2.0)


def _cosine(a: Vec, b: Vec) -> float:
    na = (a[0] * a[0] + a[1] * a[1]) ** 0.5
    nb = (b[0] * b[0] + b[1] * b[1]) ** 0.5
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return (a[0] * b[0] + a[1] * b[1]) / (na * nb)


def check_opposite_consistency(
    moves: Sequence[MoveObservation],
    *,
    min_game_cos: float = -0.5,
    max_screen_cos: float = -0.2,
) -> list[str]:
    """Fizyczny niezmiennik stałej kamery: przeciwne klawisze (w/s, a/d) muszą dać
    ~przeciwne `Δscreen`. Gęsta, regularna krata sklepów łamie dopasowanie tracków tak,
    że OBA kierunki dają zgodny (nie przeciwny) wektor — to wychwytujemy TU, zanim zły
    fit (niski residual, ale przekręcona `A`) trafi do zapisu.

    Zwraca listę komunikatów o parach, które zawiodły (pusta = OK / brak danych do oceny).
    Oceniamy tylko pary, w których ruch W GRZE był przeciwny (`cos Δgame < min_game_cos`);
    gdy postać była zablokowana i `Δgame` nie są przeciwne, pary nie osądzamy.
    """
    screen_by_key: dict[str, list[Vec]] = {}
    game_by_key: dict[str, list[Vec]] = {}
    for m in moves:
        if m.delta_screen:
            screen_by_key.setdefault(m.key, []).extend(m.delta_screen)
        game_by_key.setdefault(m.key, []).append(tuple(m.delta_game))

    problems: list[str] = []
    for key in ("w", "a"):  # po jednym reprezentancie każdej pary (w/s, a/d)
        opp = OPPOSITE_KEYS[key]
        if key not in screen_by_key or opp not in screen_by_key:
            continue
        if _cosine(_median_point(game_by_key[key]), _median_point(game_by_key[opp])) > min_game_cos:
            continue  # ruch w grze nie był przeciwny — tej pary nie oceniamy
        screen_cos = _cosine(_median_point(screen_by_key[key]), _median_point(screen_by_key[opp]))
        if screen_cos > max_screen_cos:
            problems.append(
                f"{key}/{opp}: delty ekranowe nie sa przeciwne (cos={screen_cos:+.2f} > "
                f"prog {max_screen_cos:+.2f}) — aliasing trackow w gestej kracie; "
                "kalibruj w rzadszym miejscu i wiekszym krokiem"
            )
    return problems


def check_direction_coherence(
    moves: Sequence[MoveObservation],
    *,
    min_coherence: float = 0.5,
) -> list[str]:
    """Ten sam klawisz naciskany wielokrotnie musi dać ruch w GRZE w spójnym kierunku.
    Gdy te same naciśnięcia lecą w losowe strony (krata blokuje strafe w poprzek rzędów
    albo coord OCR gubi kierunek na małym ruchu), wektory jednostkowe się znoszą —
    `coherence = |Σ û| / n` spada do ~0. Zwraca listę problematycznych klawiszy (pusta
    = OK). Ocenia tylko klawisze z >=2 ruchami o niezerowej długości.
    """
    by_key: dict[str, list[Vec]] = {}
    for m in moves:
        by_key.setdefault(m.key, []).append(tuple(m.delta_game))
    problems: list[str] = []
    for key, deltas in sorted(by_key.items()):
        units: list[Vec] = []
        for dx, dy in deltas:
            n = (dx * dx + dy * dy) ** 0.5
            if n > 1e-9:
                units.append((dx / n, dy / n))
        if len(units) < 2:
            continue
        sx = sum(u[0] for u in units) / len(units)
        sy = sum(u[1] for u in units) / len(units)
        coherence = (sx * sx + sy * sy) ** 0.5
        if coherence < min_coherence:
            problems.append(
                f"{key}: ruchy niespojne kierunkowo (coherence={coherence:.2f} < "
                f"{min_coherence:.2f}) — krata blokuje ten kierunek albo coord OCR gubi "
                "kierunek; obroc postac lub kalibruj tam, gdzie ten ruch jest czysty"
            )
    return problems


@dataclass(slots=True)
class FitResult:
    """Jakość dopasowania — do raportowania w UI i debugowania kalibracji."""

    residual_px: float          # mediana błędu rzutu w pikselach (mniej = lepiej)
    n_moves: int                # ile kroków kalibracyjnych
    n_points: int               # ile par (sklep×krok) użytych
    inliers: int                # ile przetrwało odrzut outlierów
    condition: float            # uwarunkowanie układu (duże = ruchy prawie równoległe = źle)
    opposite_problems: tuple[str, ...] = ()  # naruszenia niezmiennika przeciwnych ruchów

    @property
    def ok(self) -> bool:
        """Heurystyka „kalibracja wygląda zdrowo"."""
        return (
            self.inliers >= 4
            and self.residual_px < 6.0
            and self.condition < 1e3
            and not self.opposite_problems
        )


class GroundProjection:
    """Afiniczny rzut ekran↔teren (macierz `A` screen-delta → game-delta)."""

    def __init__(
        self,
        matrix: Sequence[Sequence[float]],
        anchor: Vec = DEFAULT_ANCHOR,
        *,
        version: str = "v1",
    ) -> None:
        A = np.asarray(matrix, dtype=float).reshape(2, 2)
        if not np.all(np.isfinite(A)):
            raise ValueError("macierz A zawiera nie-skończone wartości")
        det = float(np.linalg.det(A))
        if abs(det) < 1e-12:
            raise ValueError("macierz A jest osobliwa (nieodwracalna)")
        self.A = A
        self.A_inv = np.linalg.inv(A)
        self.anchor = (float(anchor[0]), float(anchor[1]))
        self.version = version

    # ---- rzutowanie ----------------------------------------------------------
    def screen_to_game(self, player_game: Vec, screen_px: Vec) -> Vec:
        """Piksel sklepu → jego pozycja w jednostkach gry (dla danej pozycji postaci)."""
        dp = np.array([screen_px[0] - self.anchor[0], screen_px[1] - self.anchor[1]])
        g = np.array(player_game, float) + self.A @ dp
        return (float(g[0]), float(g[1]))

    def game_to_screen(self, player_game: Vec, game: Vec) -> Vec:
        """Pozycja w jednostkach gry → piksel na ekranie (dla danej pozycji postaci)."""
        dg = np.array([game[0] - player_game[0], game[1] - player_game[1]])
        p = np.array(self.anchor, float) + self.A_inv @ dg
        return (float(p[0]), float(p[1]))

    def scale_px_per_unit(self) -> Vec:
        """Ile pikseli przypada na jednostkę gry wzdłuż osi X i Y (do sanity-checku)."""
        # kolumny A_inv = obraz wektorów jednostkowych gry w pikselach
        cx = float(np.hypot(self.A_inv[0, 0], self.A_inv[1, 0]))
        cy = float(np.hypot(self.A_inv[0, 1], self.A_inv[1, 1]))
        return (cx, cy)

    # ---- serializacja --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "matrix": self.A.reshape(-1).tolist(),
            "anchor": [self.anchor[0], self.anchor[1]],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "GroundProjection":
        return cls(
            np.asarray(data["matrix"], float).reshape(2, 2),
            tuple(data.get("anchor", DEFAULT_ANCHOR)),  # type: ignore[arg-type]
            version=str(data.get("version", "v1")),
        )

    def save(self, path: str | Path) -> None:
        """Zapis atomowy (tmp + os.replace) — spójny z resztą storage projektu."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, out)

    @classmethod
    def load(cls, path: str | Path) -> "GroundProjection | None":
        src = Path(path)
        if not src.exists():
            return None
        try:
            raw = src.read_text(encoding="utf-8")
            if not raw.strip():
                return None
            return cls.from_dict(json.loads(raw))
        except (json.JSONDecodeError, OSError, ValueError, KeyError):
            return None

    # ---- kalibracja ----------------------------------------------------------
    @classmethod
    def fit_from_moves(
        cls,
        moves: Sequence[MoveObservation],
        *,
        anchor: Vec = DEFAULT_ANCHOR,
        version: str = "v1",
        mad_k: float = 3.0,
    ) -> tuple["GroundProjection", FitResult]:
        """Dopasuj `A` z obserwacji ruchu: `A · Δp = −Δg`, odporny fit z odrzutem outlierów.

        Każdy krok wnosi (dla każdego śledzonego sklepu) parę (Δp, −Δg). Rozwiązujemy
        dwa niezależne układy najmniej-kwadratowe (wiersze A), ważone `confidence`,
        potem jedna runda odrzutu outlierów po MAD i re-fit.
        """
        P: list[list[float]] = []       # [Δpx, Δpy]
        T: list[list[float]] = []       # [−Δgx, −Δgy]
        W: list[float] = []             # wagi (confidence)
        for m in moves:
            dgx, dgy = m.delta_game
            w = max(float(m.confidence), 0.0)
            for (dpx, dpy) in m.delta_screen:
                P.append([float(dpx), float(dpy)])
                T.append([-float(dgx), -float(dgy)])
                W.append(w)
        if len(P) < 2:
            raise ValueError("za mało obserwacji do kalibracji (potrzeba ≥2 par, ≥2 kierunki)")
        Pm = np.asarray(P, float)
        Tm = np.asarray(T, float)
        Wm = np.asarray(W, float)

        # Degeneracja: jeśli Δp wszystkich kroków są (prawie) równoległe, układ jest
        # rzędu 1 → A osobliwa → rzut niemożliwy. Wykryj i zgłoś czytelnie zamiast crasha.
        cond_P = float(np.linalg.cond(Pm))
        if not np.isfinite(cond_P) or cond_P > 1e6:
            raise ValueError(
                "ruchy kalibracyjne są (prawie) równoległe — potrzeba ≥2 "
                f"nierównoległych kierunków (condition={cond_P:.1f})"
            )

        def solve(mask: np.ndarray) -> np.ndarray:
            Pmm, Tmm, Wmm = Pm[mask], Tm[mask], Wm[mask]
            sw = np.sqrt(np.clip(Wmm, 1e-9, None))[:, None]
            Pw = Pmm * sw
            # A: [a b; c d] takie że [a b]·Δp = −Δgx, [c d]·Δp = −Δgy
            row0, *_ = np.linalg.lstsq(Pw, (Tmm[:, 0:1] * sw).ravel(), rcond=None)
            row1, *_ = np.linalg.lstsq(Pw, (Tmm[:, 1:2] * sw).ravel(), rcond=None)
            return np.array([row0, row1])

        mask = np.ones(len(Pm), dtype=bool)
        A = solve(mask)
        # residual w PIKSELACH: przewidziane Δp = A⁻¹·(−Δg) vs faktyczne Δp
        A_inv = np.linalg.inv(A)
        pred_dp = (A_inv @ Tm.T).T
        resid = np.hypot(*(Pm - pred_dp).T)
        med = float(np.median(resid))
        mad = float(np.median(np.abs(resid - med))) or 1e-6
        mask = resid <= med + mad_k * mad
        if mask.sum() >= 2 and mask.sum() < len(Pm):
            A = solve(mask)
            A_inv = np.linalg.inv(A)
            pred_dp = (A_inv @ Tm.T).T
            resid = np.hypot(*(Pm - pred_dp).T)

        proj = cls(A, anchor, version=version)
        result = FitResult(
            residual_px=float(np.median(resid[mask])),
            n_moves=len(moves),
            n_points=len(Pm),
            inliers=int(mask.sum()),
            condition=cond_P,
            opposite_problems=tuple(check_opposite_consistency(moves)),
        )
        return proj, result
