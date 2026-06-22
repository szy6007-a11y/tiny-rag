#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_LANES = {
    "unit": ["tests/unit"],
    "integration": ["tests/integration"],
    "full": ["tests/unit", "tests/integration"],
}


def build_suite(lane: str) -> unittest.TestSuite:
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for rel_path in TEST_LANES[lane]:
        suite.addTests(
            loader.discover(
                start_dir=str(ROOT / rel_path),
                pattern="test_*.py",
                top_level_dir=str(ROOT),
            )
        )
    return suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run tiny-rag test lanes.")
    parser.add_argument(
        "lane",
        nargs="?",
        default="unit",
        choices=sorted(TEST_LANES),
        help="Test lane to run. Defaults to unit.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    suite = build_suite(args.lane)
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
