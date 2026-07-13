"""Małe adaptery systemowe używane przez moduły wymagające żywej gry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from PIL import Image


class ScreenBackend(Protocol):
    def grab(self, box: tuple[int, int, int, int]) -> Image.Image: ...


class InputBackend(Protocol):
    def move_to(self, x: int, y: int, duration: float = 0.0) -> None: ...
    def nudge(self, pixels: int = 3) -> None: ...
    def position(self) -> tuple[int, int]: ...
    def click(self, x: int | None = None, y: int | None = None) -> None: ...
    def key_down(self, key: str) -> None: ...
    def key_up(self, key: str) -> None: ...
    def press(self, key: str) -> None: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


class FocusBackend(Protocol):
    def is_foreground(self) -> bool: ...
    def wait_until_foreground(self, timeout: float = 15.0) -> bool: ...


class SystemClock:
    monotonic = staticmethod(time.monotonic)
    sleep = staticmethod(time.sleep)


def disable_console_quick_edit() -> bool:
    """Wyłącz tryb zaznaczania, który potrafi zamrozić proces konsolowy.

    Dotyczy klasycznego hosta konsoli Windows. W Windows Terminal funkcja może
    nie mieć zastosowania i wtedy bezpiecznie zwraca ``False``.
    """

    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        stdin_handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(stdin_handle, ctypes.byref(mode)):
            return False
        enable_quick_edit = 0x0040
        enable_extended_flags = 0x0080
        new_mode = (mode.value | enable_extended_flags) & ~enable_quick_edit
        return bool(kernel32.SetConsoleMode(stdin_handle, new_mode))
    except Exception:
        return False


class MSSScreen:
    def __init__(self) -> None:
        import mss

        self._mss = mss.MSS() if hasattr(mss, "MSS") else mss.mss()

    def grab(self, box: tuple[int, int, int, int]) -> Image.Image:
        x, y, width, height = box
        raw = self._mss.grab(
            {"left": x, "top": y, "width": width, "height": height}
        )
        return Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")


class PyAutoGUIInput:
    _SCAN_CODES = {
        "esc": 0x01,
        "escape": 0x01,
        "w": 0x11,
        "a": 0x1E,
        "s": 0x1F,
        "d": 0x20,
    }

    def __init__(self) -> None:
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.0
        self._api = pyautogui
        try:
            import win32api
            import win32con

            self._win32api = win32api
            self._win32con = win32con
        except ImportError:
            self._win32api = None
            self._win32con = None

    def _move_to_relative_legacy(
        self, x: int, y: int, duration: float = 0.0
    ) -> None:
        """Przesuń kursor względnymi zdarzeniami rozpoznawanymi przez Metin2."""

        target_x, target_y = int(x), int(y)
        if self._win32api is None or self._win32con is None:
            self._api.moveTo(target_x, target_y, duration=max(duration, 0.08))
            return

        start_x, start_y = self._win32api.GetCursorPos()
        distance = max(abs(target_x - start_x), abs(target_y - start_y))
        steps = max(8, min(24, int(distance / 35) + 1))
        previous_x, previous_y = start_x, start_y
        for index in range(1, steps + 1):
            next_x = round(start_x + (target_x - start_x) * index / steps)
            next_y = round(start_y + (target_y - start_y) * index / steps)
            self._win32api.mouse_event(
                self._win32con.MOUSEEVENTF_MOVE,
                int(next_x - previous_x),
                int(next_y - previous_y),
                0,
                0,
            )
            previous_x, previous_y = next_x, next_y
            time.sleep(0.012)
        # Akceleracja systemowa może zostawić 1–2 px błędu.
        for _ in range(3):
            try:
                self._win32api.SetCursorPos((target_x, target_y))
                break
            except Exception:
                time.sleep(0.02)

    def _move_to_absolute(
        self, x: int, y: int, duration: float = 0.0
    ) -> None:
        """Ustaw kursor absolutnie; DirectInput dostaje osobno mały nudge."""

        target_x, target_y = int(x), int(y)
        if self._win32api is not None:
            for _ in range(3):
                try:
                    self._win32api.SetCursorPos((target_x, target_y))
                    time.sleep(max(0.015, min(float(duration), 0.05)))
                    actual_x, actual_y = self._win32api.GetCursorPos()
                    if (
                        abs(actual_x - target_x) <= 2
                        and abs(actual_y - target_y) <= 2
                    ):
                        return
                except Exception:
                    time.sleep(0.02)
        for _ in range(2):
            try:
                self._api.moveTo(
                    target_x,
                    target_y,
                    duration=max(float(duration), 0.05),
                )
                return
            except Exception:
                time.sleep(0.02)

    def move_to(self, x: int, y: int, duration: float = 0.0) -> None:
        self._move_to_relative_legacy(x, y, duration)

    def nudge(self, pixels: int = 3) -> None:
        """Wyślij fizyczny ruch +/−, który wyzwala hover DirectInput."""

        amount = max(1, int(pixels))
        if self._win32api is not None and self._win32con is not None:
            self._win32api.mouse_event(
                self._win32con.MOUSEEVENTF_MOVE, amount, 0, 0, 0
            )
            time.sleep(0.03)
            self._win32api.mouse_event(
                self._win32con.MOUSEEVENTF_MOVE, -amount, 0, 0, 0
            )
            # Akceleracja myszy nie zawsze odwraca dwa względne ruchy idealnie.
            return
        self._api.moveRel(amount, 0, duration=0.02)
        self._api.moveRel(-amount, 0, duration=0.02)

    def position(self) -> tuple[int, int]:
        if self._win32api is not None:
            x, y = self._win32api.GetCursorPos()
            return int(x), int(y)
        point = self._api.position()
        return int(point.x), int(point.y)

    def click(self, x: int | None = None, y: int | None = None) -> None:
        self._api.click(x=x, y=y)

    def key_down(self, key: str) -> None:
        scan_code = self._SCAN_CODES.get(key.lower())
        if (
            scan_code is not None
            and self._win32api is not None
            and self._win32con is not None
            and hasattr(self._win32api, "keybd_event")
        ):
            self._win32api.keybd_event(
                0,
                scan_code,
                getattr(self._win32con, "KEYEVENTF_SCANCODE", 0x0008),
                0,
            )
            return
        self._api.keyDown(key)

    def key_up(self, key: str) -> None:
        scan_code = self._SCAN_CODES.get(key.lower())
        if (
            scan_code is not None
            and self._win32api is not None
            and self._win32con is not None
            and hasattr(self._win32api, "keybd_event")
        ):
            flags = getattr(self._win32con, "KEYEVENTF_SCANCODE", 0x0008)
            flags |= getattr(self._win32con, "KEYEVENTF_KEYUP", 0x0002)
            self._win32api.keybd_event(0, scan_code, flags, 0)
            return
        self._api.keyUp(key)

    def press(self, key: str) -> None:
        if key.lower() in self._SCAN_CODES:
            self.key_down(key)
            time.sleep(0.03)
            self.key_up(key)
            return
        self._api.press(key)


@dataclass(frozen=True, slots=True)
class WindowRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    @property
    def safe_cursor_point(self) -> tuple[int, int]:
        """Punkt wewnątrz klienta, z dala od środka rynku i okna sklepu."""

        return (
            self.x + min(32, max(1, self.width - 1)),
            self.y + min(48, max(1, self.height - 1)),
        )


class GameWindow:
    def __init__(self, title: str) -> None:
        self.title = title

    def locate(self) -> WindowRect:
        import pygetwindow as window_api

        matches = [
            window
            for window in window_api.getAllWindows()
            if (window.title or "").strip().lower() == self.title.lower()
            and window.width > 300
            and window.height > 300
        ]
        if not matches:
            matches = [
                window
                for window in window_api.getAllWindows()
                if self.title.lower() in (window.title or "").lower()
                and window.width > 300
                and window.height > 300
            ]
        if not matches:
            raise RuntimeError(f"nie znaleziono okna gry: {self.title!r}")
        window = matches[0]
        handle = getattr(window, "_hWnd", None)
        if handle:
            try:
                import win32gui

                left, top, right, bottom = win32gui.GetClientRect(handle)
                screen_x, screen_y = win32gui.ClientToScreen(handle, (left, top))
                return WindowRect(
                    screen_x, screen_y, right - left, bottom - top
                )
            except ImportError:
                pass
        return WindowRect(window.left, window.top, window.width, window.height)

    def handle(self) -> int | None:
        import pygetwindow as window_api

        matches = [
            window
            for window in window_api.getAllWindows()
            if (window.title or "").strip().lower() == self.title.lower()
            and window.width > 300
            and window.height > 300
        ]
        if not matches:
            matches = [
                window
                for window in window_api.getAllWindows()
                if self.title.lower() in (window.title or "").lower()
                and window.width > 300
                and window.height > 300
            ]
        return getattr(matches[0], "_hWnd", None) if matches else None

    def is_foreground(self) -> bool:
        try:
            import win32gui

            handle = self.handle()
            return bool(handle and win32gui.GetForegroundWindow() == handle)
        except Exception:
            return False

    def wait_until_foreground(self, timeout: float = 15.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.is_foreground():
                return True
            time.sleep(0.1)
        return self.is_foreground()

    def activate(self) -> bool:
        """Spróbuj aktywować klienta, ale nie przerywaj pracy przy blokadzie Windows.

        Windows może legalnie odmówić ``SetForegroundWindow`` procesowi, który
        nie jest aktualnie na pierwszym planie. PyGetWindow dodatkowo potrafi
        zgłosić wyjątek z kodem 0 mimo udanej operacji. Fokus jest więc
        ułatwieniem, nie warunkiem działania — użytkownik nadal może kliknąć
        klienta podczas odliczania.
        """

        import pygetwindow as window_api

        matches = [
            window
            for window in window_api.getAllWindows()
            if (window.title or "").strip().lower() == self.title.lower()
            and window.width > 300
            and window.height > 300
        ]
        if not matches:
            matches = [
                window
                for window in window_api.getAllWindows()
                if self.title.lower() in (window.title or "").lower()
                and window.width > 300
                and window.height > 300
            ]
        if not matches:
            raise RuntimeError(f"nie znaleziono okna gry: {self.title!r}")
        window = matches[0]
        if window.isMinimized:
            try:
                window.restore()
            except Exception:
                pass
        activated = False
        try:
            window.activate()
            activated = True
        except Exception:
            handle = getattr(window, "_hWnd", None)
            if handle:
                try:
                    import win32api
                    import win32con
                    import win32gui
                    import win32process

                    win32gui.ShowWindow(handle, win32con.SW_RESTORE)
                    foreground = win32gui.GetForegroundWindow()
                    current_thread = win32api.GetCurrentThreadId()
                    foreground_thread = (
                        win32process.GetWindowThreadProcessId(foreground)[0]
                        if foreground
                        else 0
                    )
                    target_thread = win32process.GetWindowThreadProcessId(handle)[0]
                    attached: list[int] = []
                    try:
                        for thread_id in {foreground_thread, target_thread}:
                            if thread_id and thread_id != current_thread:
                                win32process.AttachThreadInput(
                                    current_thread, thread_id, True
                                )
                                attached.append(thread_id)

                        # Krótkie naciśnięcie ALT odblokowuje ograniczenie
                        # SetForegroundWindow w większości wersji Windows.
                        win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
                        win32api.keybd_event(
                            win32con.VK_MENU,
                            0,
                            win32con.KEYEVENTF_KEYUP,
                            0,
                        )
                        win32gui.BringWindowToTop(handle)
                        win32gui.SetWindowPos(
                            handle,
                            win32con.HWND_TOPMOST,
                            0,
                            0,
                            0,
                            0,
                            win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
                        )
                        win32gui.SetWindowPos(
                            handle,
                            win32con.HWND_NOTOPMOST,
                            0,
                            0,
                            0,
                            0,
                            win32con.SWP_NOMOVE
                            | win32con.SWP_NOSIZE
                            | win32con.SWP_SHOWWINDOW,
                        )
                        win32gui.SetForegroundWindow(handle)
                        try:
                            win32gui.SetFocus(handle)
                        except Exception:
                            pass
                    finally:
                        for thread_id in reversed(attached):
                            try:
                                win32process.AttachThreadInput(
                                    current_thread, thread_id, False
                                )
                            except Exception:
                                pass
                    activated = win32gui.GetForegroundWindow() == handle
                except Exception:
                    # Odmowa Windows nie może wywrócić całego skanowania.
                    activated = False
        time.sleep(0.2)
        return activated
