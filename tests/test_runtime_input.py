from __future__ import annotations

import unittest
from unittest.mock import patch

from scanner.runtime import PyAutoGUIInput


class _Win32Api:
    def __init__(self) -> None:
        self.position = (100, 200)
        self.relative_events: list[tuple[int, int]] = []
        self.absolute_events: list[tuple[int, int]] = []
        self.keyboard_events: list[tuple[int, int, int, int]] = []

    def GetCursorPos(self) -> tuple[int, int]:
        return self.position

    def mouse_event(
        self, flag: int, dx: int, dy: int, data: int, extra: int
    ) -> None:
        self.relative_events.append((dx, dy))
        self.position = (self.position[0] + dx, self.position[1] + dy)

    def SetCursorPos(self, point: tuple[int, int]) -> None:
        self.absolute_events.append(point)
        self.position = point

    def keybd_event(self, vk: int, scan: int, flags: int, extra: int) -> None:
        self.keyboard_events.append((vk, scan, flags, extra))


class _Win32Con:
    MOUSEEVENTF_MOVE = 1
    KEYEVENTF_KEYUP = 0x0002
    KEYEVENTF_SCANCODE = 0x0008


class RuntimeInputTests(unittest.TestCase):
    def test_move_to_uses_legacy_relative_events_after_capture(self) -> None:
        backend = object.__new__(PyAutoGUIInput)
        backend._api = None
        backend._win32api = _Win32Api()
        backend._win32con = _Win32Con()

        with patch("scanner.runtime.time.sleep"):
            backend.move_to(340, 425, duration=0.1)

        self.assertGreaterEqual(len(backend._win32api.relative_events), 8)
        self.assertEqual(backend._win32api.absolute_events[-1], (340, 425))
        self.assertEqual(backend.position(), (340, 425))

    def test_nudge_emits_opposite_relative_movements(self) -> None:
        backend = object.__new__(PyAutoGUIInput)
        backend._api = None
        backend._win32api = _Win32Api()
        backend._win32con = _Win32Con()

        with patch("scanner.runtime.time.sleep"):
            backend.nudge(5)

        self.assertEqual(
            backend._win32api.relative_events[-2:], [(5, 0), (-5, 0)]
        )
        self.assertEqual(backend.position(), (100, 200))
        self.assertEqual(backend._win32api.absolute_events, [])

    def test_wasd_uses_directinput_scan_codes(self) -> None:
        backend = object.__new__(PyAutoGUIInput)
        backend._api = None
        backend._win32api = _Win32Api()
        backend._win32con = _Win32Con()

        backend.key_down("d")
        backend.key_up("d")

        self.assertEqual(
            backend._win32api.keyboard_events,
            [
                (0, 0x20, _Win32Con.KEYEVENTF_SCANCODE, 0),
                (
                    0,
                    0x20,
                    _Win32Con.KEYEVENTF_SCANCODE | _Win32Con.KEYEVENTF_KEYUP,
                    0,
                ),
            ],
        )

    def test_escape_press_uses_directinput_scan_code(self) -> None:
        backend = object.__new__(PyAutoGUIInput)
        backend._api = None
        backend._win32api = _Win32Api()
        backend._win32con = _Win32Con()

        with patch("scanner.runtime.time.sleep"):
            backend.press("esc")

        self.assertEqual(
            backend._win32api.keyboard_events,
            [
                (0, 0x01, _Win32Con.KEYEVENTF_SCANCODE, 0),
                (
                    0,
                    0x01,
                    _Win32Con.KEYEVENTF_SCANCODE | _Win32Con.KEYEVENTF_KEYUP,
                    0,
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
