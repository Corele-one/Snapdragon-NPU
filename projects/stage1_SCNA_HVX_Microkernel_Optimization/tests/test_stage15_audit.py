import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]


class Stage15AuditTest(unittest.TestCase):
    def test_stage15_audit_reproduces_not_built_decisions(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = subprocess.run(
                [sys.executable, str(PROJECT / "tools/audit_stage15_layout_batching.py"),
                 "--project", str(PROJECT), "--json-out", str(output / "audit.json"),
                 "--report-out", str(output / "audit.md")],
                check=True, capture_output=True, text=True,
            )
            self.assertIn("NOT_BUILT_BOTH_CANDIDATES", result.stdout)
            audit = json.loads((output / "audit.json").read_text())
            self.assertEqual(audit["baseline"]["packets"], 36)
            self.assertEqual(audit["candidate_1_layout"]["packed_vectors_per_baseline_block"], 8)
            self.assertEqual(audit["candidate_1_layout"]["packed_affine_vector_steps_lower_bound"], 8)
            self.assertEqual(audit["candidate_2_batching"]["two_column_blocks_vectors"], 4)
            self.assertEqual(audit["candidate_2_batching"]["quad_caller_stack_delta"], 1920)
            self.assertIn("No new device candidate was built.", (output / "audit.md").read_text())
