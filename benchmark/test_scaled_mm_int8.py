# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import torch

import flag_gems

from . import base, consts


class HopperINT8FallbackBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 6144, 1536),
            (64, 6144, 1536),
            (4096, 6144, 1536),
            (5089, 6144, 3072),
            (8192, 1024, 6144),
            (4096, 1536, 6144),
            (5089, 6144, 768),
        ]
        self.shape_desc = "M, K, N; row-major B"

    def get_input_iter(self, dtype):
        for m, k, n in self.shapes:
            yield (
                torch.randint(-32, 32, (m, k), device=self.device, dtype=torch.int8),
                torch.randint(-32, 32, (k, n), device=self.device, dtype=torch.int8),
                torch.rand((m, 1), device=self.device) * 0.01,
                torch.rand((1, n), device=self.device) * 0.01,
            )

    def get_tflops(self, op, a, b, *args, **kwargs):
        return 2 * a.shape[0] * a.shape[1] * b.shape[1]


@pytest.mark.scaled_mm
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("baseline", ["torch", "autotune"])
def test_int8_hopper_fallback(baseline, monkeypatch):
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("Hopper opt-in tiles")
    monkeypatch.setenv("FLAGGEMS_I8_SCALED_MM_SHAPE_TILES", "1")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

    def reference(a, b, sa, sb):
        if baseline == "autotune":
            os.environ["FLAGGEMS_I8_SCALED_MM_SHAPE_TILES"] = "0"
            return flag_gems.scaled_mm_int8(a, b, sa, sb, out_dtype=torch.bfloat16)
        return ((a.float() @ b.float()) * sa * sb).bfloat16()

    def candidate(a, b, sa, sb):
        os.environ["FLAGGEMS_I8_SCALED_MM_SHAPE_TILES"] = "1"
        return flag_gems.scaled_mm_int8(a, b, sa, sb, out_dtype=torch.bfloat16)

    bench = HopperINT8FallbackBenchmark(
        op_name=f"int8_hopper_fallback_{baseline}",
        torch_op=reference,
        gems_op=candidate,
        dtypes=[torch.int8],
    )
    bench.run()
