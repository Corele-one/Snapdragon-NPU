from __future__ import annotations

import importlib.util
import math
import struct
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fixture", PROJECT / "tools/generate_fixture.py")
fixture = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(fixture)


class FixtureTests(unittest.TestCase):
    def test_fixed_fixture_matches_stage2_prefix(self) -> None:
        q, k, v, mask = fixture.make_payload(1, 65, 12, 2, 64, "full", "figure8_fixed")
        self.assertAlmostEqual(struct.unpack_from("<f", q, 0)[0], -118 * 0.0078125)
        self.assertAlmostEqual(struct.unpack_from("<e", k, 0)[0], -125 * 0.00390625, places=3)
        self.assertAlmostEqual(struct.unpack_from("<e", v, 0)[0], -126 * 0.00390625, places=3)
        self.assertEqual(len(mask), math.ceil(65 / 64) * 64 * 2)
        self.assertEqual(struct.unpack_from("<e", mask, 64 * 2)[0], 0.0)
        self.assertLess(struct.unpack_from("<e", mask, 65 * 2)[0], -65000.0)

    def test_causal_and_padding_masks(self) -> None:
        _, _, _, causal = fixture.make_payload(4, 16, 12, 2, 64, "causal", "figure8_fixed")
        _, _, _, padding = fixture.make_payload(1, 16, 12, 2, 64, "padding", "figure8_fixed")
        self.assertLess(struct.unpack_from("<e", causal, 13 * 2)[0], -65000.0)
        self.assertEqual(struct.unpack_from("<e", causal, 12 * 2)[0], 0.0)
        self.assertLess(struct.unpack_from("<e", padding, 13 * 2)[0], -65000.0)
        self.assertEqual(struct.unpack_from("<e", padding, 12 * 2)[0], 0.0)

    def test_header_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "fixture.bin"
            q, k, v, mask = fixture.make_payload(4, 64, 12, 2, 128, "full", "20260810")
            header = fixture.HEADER.pack(fixture.MAGIC, 1, 4, 64, 64, 12, 2, 128, 0,
                                         20260810, 0, len(q), len(k), len(v), len(mask))
            output.write_bytes(header + q + k + v + mask)
            fields = fixture.HEADER.unpack_from(output.read_bytes())
            self.assertEqual(fields[0], fixture.MAGIC)
            self.assertEqual(fields[2:8], (4, 64, 64, 12, 2, 128))


if __name__ == "__main__":
    unittest.main()

