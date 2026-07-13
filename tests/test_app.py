from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scanner.app import (
    _analysis_ready,
    _await_initial_position,
    _countdown,
    main,
    parser,
    settings_from_walk_config,
)
from scanner.config import load_settings
from tests.fakes import FakeClock, FakeInput


class AppTests(unittest.TestCase):
    def test_cli_parses_capture_and_auto(self) -> None:
        self.assertEqual(parser().parse_args(["probe"]).command, "probe")
        self.assertEqual(parser().parse_args(["coords"]).command, "coords")
        reset_map = parser().parse_args(["reset-map"])
        self.assertEqual(reset_map.command, "reset-map")
        self.assertFalse(reset_map.no_backup)
        hover = parser().parse_args(
            ["hover-bench", "--slot", "12", "--attempts", "8"]
        )
        self.assertEqual(hover.command, "hover-bench")
        self.assertEqual(hover.slot, 12)
        self.assertEqual(hover.attempts, 8)
        window_calibrate = parser().parse_args(
            [
                "window-calibrate",
                "--key",
                "f8",
                "--output",
                "dbg/custom.png",
                "--no-save",
            ]
        )
        self.assertEqual(window_calibrate.command, "window-calibrate")
        self.assertEqual(window_calibrate.key, "f8")
        self.assertEqual(window_calibrate.output, "dbg/custom.png")
        self.assertTrue(window_calibrate.no_save)
        grid_calibrate = parser().parse_args(
            [
                "grid-calibrate",
                "--cell",
                "32",
                "--grid-dx",
                "7",
                "--grid-dy",
                "32",
                "--save",
            ]
        )
        self.assertEqual(grid_calibrate.command, "grid-calibrate")
        self.assertEqual(grid_calibrate.cell, 32)
        self.assertEqual(grid_calibrate.grid_dx, 7)
        self.assertEqual(grid_calibrate.grid_dy, 32)
        self.assertTrue(grid_calibrate.save)
        self.assertEqual(parser().parse_args(["capture-open"]).command, "capture-open")
        args = parser().parse_args(
            [
                "auto",
                "--walk",
                "--max-shops",
                "3",
                "--analyze",
                "--csv",
                "out.csv",
                "--debug-live",
                "--lanes",
                "4",
                "--steps-per-lane",
                "6",
                "--step-hold",
                "0.5",
                "--settle",
                "0.7",
            ]
        )
        self.assertTrue(args.walk)
        self.assertTrue(args.analyze)
        self.assertEqual(args.max_shops, 3)
        self.assertEqual(args.csv, "out.csv")
        self.assertTrue(args.debug_live)
        self.assertEqual(args.lanes, 4)
        self.assertEqual(args.steps_per_lane, 6)
        self.assertEqual(args.step_hold, 0.5)
        self.assertEqual(args.settle, 0.7)

    def test_auto_parser_defines_flags_command_auto_consumes(self) -> None:
        # Regresja v10.52: command_auto czyta args.phase_b/args.zone, a parser
        # ich nie tworzyl -> AttributeError nawet na `auto --walk`. Test pilnuje
        # kontraktu parser() <-> command_auto.
        ns = parser().parse_args(["auto", "--walk"])
        self.assertFalse(ns.phase_b)
        self.assertFalse(ns.zone)
        self.assertTrue(parser().parse_args(["auto", "--phase-b"]).phase_b)
        self.assertTrue(parser().parse_args(["auto", "--zone"]).zone)
        self.assertFalse(parser().parse_args(["auto", "--walk"]).auto_grow_boundary)
        self.assertTrue(
            parser().parse_args(["auto", "--auto-grow-boundary"]).auto_grow_boundary
        )
        self.assertFalse(parser().parse_args(["auto", "--walk"]).fresh_map)
        self.assertTrue(parser().parse_args(["auto", "--fresh-map"]).fresh_map)
        self.assertEqual(
            parser().parse_args(["auto", "--coverage-drive"]).coverage_passes,
            3,
        )
        self.assertEqual(
            parser().parse_args(["auto", "--coverage-drive"]).coverage_cell_size,
            30.0,
        )
        self.assertEqual(
            parser().parse_args(["auto", "--coverage-drive"]).popup_budget,
            40,
        )
        self.assertEqual(
            parser().parse_args(
                ["auto", "--coverage-drive", "--popup-budget", "0"]
            ).popup_budget,
            0,
        )
        self.assertEqual(
            parser().parse_args(
                ["auto", "--coverage-drive", "--coverage-passes", "5"]
            ).coverage_passes,
            5,
        )
        self.assertEqual(
            parser().parse_args(
                ["auto", "--coverage-drive", "--coverage-cell-size", "40"]
            ).coverage_cell_size,
            40.0,
        )


    def test_zone_hook_seam_api_matches_command_auto(self) -> None:
        # P0b: _zone_hook w command_auto wola te nazwy. Test pilnuje, ze szew
        # nawigatora/rejestru ich nie zgubi (regresja v10.50: hook wolal
        # nieistniejace update_position/record_shop_open/has_fingerprint).
        from scanner.analysis.shop_registry import ShopRegistry
        from scanner.navigation.map_navigator import MapSynchronizedNavigator

        self.assertTrue(hasattr(MapSynchronizedNavigator, "stamp_position"))
        self.assertTrue(hasattr(MapSynchronizedNavigator, "record_shop"))
        self.assertTrue(hasattr(MapSynchronizedNavigator, "ingest_manifest"))
        self.assertIsInstance(
            MapSynchronizedNavigator.current_zone_id, property
        )
        self.assertTrue(hasattr(ShopRegistry, "by_fingerprint"))


    def test_walk_settings_are_loaded_from_legacy_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(
                '{"walk":{"key_left":"q","key_right":"e","drop_key":"s",'
                '"step_hold":0.4,"steps_per_lane":2,"lanes":2,"settle":0.1}}',
                encoding="utf-8",
            )
            route = settings_from_walk_config(path).steps()
            self.assertEqual(route[0].key, "e")
            self.assertEqual(route[3].key, "q")

    def test_walk_cli_overrides_do_not_modify_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            original = (
                '{"walk":{"key_left":"a","key_right":"d","drop_key":"s",'
                '"step_hold":0.6,"steps_per_lane":4,"lanes":3,"settle":0.9}}'
            )
            path.write_text(original, encoding="utf-8")

            planner = settings_from_walk_config(
                path,
                lanes=4,
                steps_per_lane=6,
                step_duration=0.5,
                settle=0.7,
            )

            self.assertEqual(planner.lanes, 4)
            self.assertEqual(planner.steps_per_lane, 6)
            self.assertEqual(len(planner.steps()), 27)
            self.assertEqual(planner.step_duration, 0.5)
            self.assertEqual(planner.settle, 0.7)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_walk_cli_rejects_invalid_route_size(self) -> None:
        with self.assertRaises(SystemExit):
            parser().parse_args(["auto", "--walk", "--lanes", "0"])

    def test_glevia_grid_always_uses_32_pixel_cells(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(
                '{"window_title":"Glevia2","shop_origin":[10,20],'
                '"grid":{"cell":30,"grid_dx":7,"grid_dy":32}}',
                encoding="utf-8",
            )
            self.assertEqual(load_settings(path).grid.cell, 32)

    def test_retry_open_timeout_defaults_to_one_second(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text("{}", encoding="utf-8")

            self.assertEqual(load_settings(path).retry_open_timeout, 1.0)

    def test_first_pass_hover_attempts_defaults_to_two(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text("{}", encoding="utf-8")

            settings = load_settings(path)

            self.assertEqual(settings.capture.first_pass_hover_attempts, 2)
            self.assertEqual(settings.capture.hover_attempts, 3)

    def test_analysis_gate_uses_ollama_availability(self) -> None:
        self.assertTrue(_analysis_ready(False))
        with patch(
            "scanner.analysis.ollama_reader.available", return_value=True
        ):
            self.assertTrue(_analysis_ready(True))
        with patch(
            "scanner.analysis.ollama_reader.available", return_value=False
        ):
            self.assertFalse(_analysis_ready(True))

    def test_initial_position_retries_ocr_without_input(self) -> None:
        class Pipeline:
            def __init__(self) -> None:
                self.calls = 0

            def read_current_position(self):
                self.calls += 1
                return None if self.calls < 3 else (458, 720)

        pipeline = Pipeline()
        result = _await_initial_position(
            pipeline,
            timeout=1.0,
            poll_interval=0.1,
            clock=FakeClock(),
        )

        self.assertEqual(result, (458, 720))
        self.assertEqual(pipeline.calls, 3)

    def test_initial_position_gives_up_without_movement(self) -> None:
        class Pipeline:
            def __init__(self) -> None:
                self.calls = 0

            def read_current_position(self):
                self.calls += 1
                return None

        pipeline = Pipeline()
        result = _await_initial_position(
            pipeline,
            timeout=0.2,
            poll_interval=0.1,
            clock=FakeClock(),
        )

        self.assertIsNone(result)
        self.assertGreaterEqual(pipeline.calls, 3)

    def test_start_gate_waits_for_stable_foreground_without_input(self) -> None:
        class Window:
            def is_foreground(self) -> bool:
                return True

        input_backend = FakeInput()
        result = _countdown(
            Window(),
            input_backend,
            clock=FakeClock(),
            stable_seconds=0.2,
            poll_interval=0.1,
        )

        self.assertIsNone(result)
        self.assertEqual(input_backend.actions, [])

    def test_start_gate_resets_when_focus_is_lost(self) -> None:
        class Window:
            states = iter((False, True, False, True, True, True, True))

            def is_foreground(self) -> bool:
                return next(self.states)

        _countdown(
            Window(),
            FakeInput(),
            clock=FakeClock(),
            stable_seconds=0.2,
            poll_interval=0.1,
        )

    def test_main_prints_runtime_error_without_traceback(self) -> None:
        args = parser().parse_args(["probe"])
        args.handler = lambda _: (_ for _ in ()).throw(RuntimeError("test focus"))
        fake_parser = Mock()
        fake_parser.parse_args.return_value = args
        with patch("scanner.app.parser", return_value=fake_parser):
            self.assertEqual(main([]), 1)

    def test_zone_hook_persists_registry_and_zone_map(self) -> None:
        # v10.56: _zone_hook musi ocalac rejestr i strefy po ingestedzie
        # (wczesniej market_map ginal co bieg).
        import tempfile
        from scanner.analysis.shop_registry import ShopRegistry
        from scanner.analysis.zone_map import ZoneMap
        from scanner.navigation.map_navigator import MapSynchronizedNavigator
        from scanner.models.shop_scan import ShopScan

        with tempfile.TemporaryDirectory() as tmp:
            market_dir = Path(tmp) / "glevia_market"
            market_dir.mkdir()
            registry = ShopRegistry.open(Path(tmp), partition="glevia_market")
            zmap = ZoneMap((348, 672, 501, 794), directory=market_dir)
            nav = MapSynchronizedNavigator(zmap, registry)

            scan = ShopScan(
                scan_id="p1",
                seller="Testowy",
                game_position=(400, 700),
                shop_fingerprint="fp_test_abc",
            )
            # Symulujemy cialo _zone_hook z command_auto
            gp = scan.game_position
            nav.stamp_position(gp)
            zid = nav.current_zone_id
            is_new = registry.by_fingerprint(scan.shop_fingerprint) is None
            nav.record_shop(zid, is_new_fingerprint=is_new)
            registry.ingest({
                "scan_id": scan.scan_id,
                "shop_fingerprint": scan.shop_fingerprint,
                "game_position": list(gp),
                "seller": scan.seller,
                "created_at": scan.created_at,
            })
            registry.save()
            zmap.save()

            # Asercje: pliki powstaly
            self.assertTrue(
                (market_dir / "shops.jsonl").exists(),
                "shops.jsonl powinien powstac po save()"
            )
            self.assertTrue(
                (market_dir / "zones.json").exists(),
                "zones.json powinien powstac po save()"
            )
            # Reload: rejestr przezywa
            reg2 = ShopRegistry.open(Path(tmp), partition="glevia_market")
            self.assertEqual(len(reg2), 1)
            self.assertIsNotNone(reg2.by_fingerprint("fp_test_abc"))
            # Reload: strefy przezywaja
            zmap2 = ZoneMap.load(market_dir)
            self.assertEqual(zmap2.saturation_k, zmap.saturation_k)


if __name__ == "__main__":
    unittest.main()
