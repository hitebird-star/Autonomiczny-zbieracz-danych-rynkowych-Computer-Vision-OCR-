from __future__ import annotations

import threading
import unittest

import win_ocr


class WinOcrLoopTests(unittest.TestCase):
    def test_event_loop_is_thread_local(self) -> None:
        main_loop = win_ocr._get_loop()
        loops = []

        thread = threading.Thread(target=lambda: loops.append(win_ocr._get_loop()))
        thread.start()
        thread.join()

        self.assertIs(win_ocr._get_loop(), main_loop)
        self.assertEqual(len(loops), 1)
        self.assertIsNot(loops[0], main_loop)


if __name__ == "__main__":
    unittest.main()
