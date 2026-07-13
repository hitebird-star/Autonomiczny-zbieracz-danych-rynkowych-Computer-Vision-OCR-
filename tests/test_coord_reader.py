from __future__ import annotations

import unittest

from scanner.analysis.coord_reader import (
    DEFAULT_ATTEMPTS,
    STARTUP_FALLBACK_ATTEMPTS,
    parse_coord_text,
    plausible_jump,
)


class ParseCoordTextTests(unittest.TestCase):
    def test_reads_coord_in_parens(self) -> None:
        p = parse_coord_text("Glevia2 Farm, CH1 (458, 720) 20:06:26")
        self.assertIsNotNone(p)
        self.assertEqual((p.x, p.y), (458, 720))
        self.assertEqual(p.channel, 1)

    def test_rejects_clock_without_parens(self) -> None:
        # zegar 20:06:26 ma dwukropki, nie nawiasy -> brak falszywego (20,6).
        self.assertIsNone(parse_coord_text("EXP 1.34% 20:06:26 21:37:24"))

    def test_rejects_out_of_bounds(self) -> None:
        self.assertIsNone(parse_coord_text("(99999, 5)"))

    def test_channel_optional(self) -> None:
        p = parse_coord_text("(100, 200)")
        self.assertIsNotNone(p)
        self.assertIsNone(p.channel)

    def test_map_name_best_effort(self) -> None:
        p = parse_coord_text("Glevia2 Farm CH2 (12, 34)")
        self.assertEqual(p.channel, 2)
        self.assertIn("Glevia2", p.map_name or "")

    def test_semicolon_separator_tolerated(self) -> None:
        # OCR czasem myli przecinek ze srednikiem.
        p = parse_coord_text("(458; 720)")
        self.assertEqual((p.x, p.y), (458, 720))

    def test_ocr_digit_confusions_inside_coord_are_corrected(self) -> None:
        p = parse_coord_text("Glevia2 Farm CH1 (454* 72B)")
        self.assertIsNotNone(p)
        self.assertEqual((p.x, p.y), (454, 728))

    def test_space_separator_and_missing_closing_paren_tolerated(self) -> None:
        p = parse_coord_text("Glevia2 Farm CH1 (456 729")
        self.assertIsNotNone(p)
        self.assertEqual((p.x, p.y), (456, 729))

    def test_opening_paren_still_required_after_parser_relaxation(self) -> None:
        self.assertIsNone(parse_coord_text("Glevia2 Farm CH1 456 729) 20:06:26"))

    def test_default_attempts_include_190_threshold_and_white_key(self) -> None:
        attempts = [(roi, scale, mode) for roi, scale, mode in DEFAULT_ATTEMPTS]
        self.assertIn((attempts[1][0], 5, 190), attempts)
        self.assertIn((attempts[1][0], 5, "white_outline"), attempts)
        self.assertIn((attempts[1][0], 5, "white"), attempts)

    def test_startup_fallback_uses_wider_left_edge_but_stays_out_of_default(self) -> None:
        fallback_roi, fallback_scale, fallback_mode = STARTUP_FALLBACK_ATTEMPTS[0]
        self.assertLess(fallback_roi[0], DEFAULT_ATTEMPTS[0][0][0])
        self.assertEqual(fallback_scale, 4)
        self.assertIsNone(fallback_mode)
        self.assertNotIn(STARTUP_FALLBACK_ATTEMPTS[0], DEFAULT_ATTEMPTS)
        self.assertIn("white_outline", [attempt[2] for attempt in STARTUP_FALLBACK_ATTEMPTS])


class PlausibleJumpTests(unittest.TestCase):
    def test_first_read_always_accepted(self) -> None:
        self.assertTrue(plausible_jump(None, (500, 500)))

    def test_small_step_accepted(self) -> None:
        self.assertTrue(plausible_jump((458, 720), (458, 715)))

    def test_large_jump_rejected(self) -> None:
        # skok o 200 w x = nieprawdopodobny (prog 60) -> odrzut bledu OCR.
        self.assertFalse(plausible_jump((458, 720), (258, 720)))

    def test_jump_on_boundary(self) -> None:
        self.assertTrue(plausible_jump((100, 100), (160, 100), max_jump=60))
        self.assertFalse(plausible_jump((100, 100), (161, 100), max_jump=60))


if __name__ == "__main__":
    unittest.main()
