# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import unittest
from pathlib import Path

import torch


class QuantContractTests(unittest.TestCase):
    def test_cpu_and_wrong_width_rejected_without_launch(self):
        path = Path(__file__).parents[1] / "src/flag_gems/fused/dsv41/group32_quant.py"
        spec = importlib.util.spec_from_file_location("isolated_quant32", path)
        quant = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(quant)
        for x in (torch.zeros(2, 288, dtype=torch.bfloat16), torch.zeros(2, 31)):
            with self.subTest(shape=x.shape):
                self.assertFalse(quant.supports(x))
                with self.assertRaises(ValueError):
                    quant.quantize_group32_ue8m0(x)


if __name__ == "__main__":
    unittest.main()
