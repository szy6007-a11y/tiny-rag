"""Compatibility entrypoint for the historical chunking test command.

The canonical runner discovers ``tests/unit`` directly. When generic unittest
discovery walks the whole ``tests`` tree, this module returns an empty suite so
the chunking tests are not counted twice.
"""

import unittest


def load_tests(loader: unittest.TestLoader, tests, pattern):
    if pattern is not None:
        return unittest.TestSuite()
    return loader.loadTestsFromName("tests.unit.test_chunking")
