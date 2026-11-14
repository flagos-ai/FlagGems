# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Public extracted-operator rejection contracts, without initializing CUDA."""

import importlib.util
import unittest
from pathlib import Path

import pytest
import torch


def module(name):
    path = Path(__file__).parents[1] / "src/flag_gems/fused/dsv41" / (name + ".py")
    spec = importlib.util.spec_from_file_location("isolated_" + name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


class PrefillContractTests(unittest.TestCase):
    def test_score_rejects_cpu_before_device_work(self):
        score = module("fp8_prefill_score")
        with self.assertRaises(ValueError):
            score.fp8_prefill_score(
                torch.zeros(2, 32, 128).to(torch.float8_e4m3fn),
                torch.zeros(2, 32, dtype=torch.bfloat16),
                torch.zeros(8, 128).to(torch.float8_e4m3fn),
                torch.tensor([0, 8]),
            )

    def test_stats_rejects_wrong_weight_width(self):
        stats = module("mhc_prefill_stats")
        with self.assertRaises(ValueError):
            stats.run_stats(
                torch.zeros(1, 256, dtype=torch.bfloat16), torch.zeros(24, 256), 1e-20
            )

    def test_swiglu_rejects_mismatched_physical_rows(self):
        quant = module("swiglu_row_quant")
        with self.assertRaisesRegex(ValueError, "physical row"):
            quant.swiglu_row_quant(
                torch.zeros(2, 16, dtype=torch.bfloat16),
                torch.tensor([0]),
                torch.zeros(2, 8).to(torch.float8_e4m3fn),
                torch.zeros(2, 1),
                num_experts=4,
                calc_scale=None,
                finalize_scale=None,
                quant_tensor=None,
            )

    def test_masked_quant_rejects_scale_byte_storage(self):
        quant = module("masked_row_quant")
        with self.assertRaises(TypeError):
            quant.masked_row_quant(
                torch.zeros(2, 4, 8, dtype=torch.bfloat16),
                torch.zeros(8, 8).to(torch.float8_e4m3fn),
                torch.zeros(8, 1, dtype=torch.uint8),
                torch.zeros(2, dtype=torch.int32),
                calc_scale=None,
                finalize_scale=None,
                quant_tensor=None,
            )


if __name__ == "__main__":
    unittest.main()


def test_migrated_casts_reject_cpu_before_dispatch(monkeypatch):
    import sys
    import types

    for name in ("flag_gems", "flag_gems.fused", "flag_gems.fused.dsv41"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    fp4 = types.ModuleType("flag_gems.fused.dsv41.fp4_indexer")
    fp4.PAYLOAD_BYTES, fp4.SCALE_BYTES = 64, 4
    fp4._e2m1_decode = lambda value: value
    monkeypatch.setitem(sys.modules, fp4.__name__, fp4)
    casts = module("fp4_prefill")
    with pytest.raises(ValueError, match="CUDA query"):
        casts.quantize_bf16_index_queries_fp8(
            torch.zeros(1, 1, 128, dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="CUDA physical slots"):
        casts.unpack_fp4_index_keys_to_fp8(
            torch.tensor([0]), torch.zeros(1, 68, dtype=torch.uint8), 1
        )


def test_migrated_masked_swiglu_rejects_invalid_output():
    activation = module("masked_swiglu")
    with pytest.raises(ValueError, match="geometry"):
        activation.masked_swiglu(
            torch.zeros(2, 4, 8, dtype=torch.bfloat16),
            torch.zeros(7, 4, dtype=torch.bfloat16),
            torch.tensor([1, 3], dtype=torch.int32),
            activate=None,
        )
