# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for the RMSNorm dW backward kernel, comparing the baseline
Triton kernel (``rms_norm_grad_dw_kernel``) against the TLE-optimized
variant (``rms_norm_grad_dw_kernel_tle``) that applies ``tle.gpu.set_layout``.

The two implementations being compared are low-level Triton kernels, not a
``torch`` op vs. a ``flag_gems`` op, and each needs its own grid / extra
kwargs (``TARGET_LAYOUT`` for the TLE variant). Rather than forcing this
through ``GenericBenchmark2DOnly`` (which assumes plain callables taking
positional/keyword tensor args), this subclasses ``base.Benchmark`` directly
and overrides ``get_latency`` so the kernel launch (``kernel[grid](...)``)
is handled explicitly, while still reusing the standard run loop,
warmup/rep handling, metrics, and result reporting (``BenchmarkResult`` /
``update_result`` / ``emit_record_logger``).
"""

import pytest
import torch
import triton

from flag_gems.utils.triton_version_utils import HAS_TLE

from . import base, consts
from .conftest import Config

if HAS_TLE:
    from flag_gems.ops.rms_norm import (
        _DW_COL_BLOCK_SIZE,
        _DW_ROW_BLOCK_SIZE,
        _DW_TARGET_LAYOUT,
        _DW_TLE_NUM_WARPS,
        rms_norm_grad_dw_kernel,
        rms_norm_grad_dw_kernel_tle,
    )
else:
    rms_norm_grad_dw_kernel = None
    rms_norm_grad_dw_kernel_tle = None


def rms_norm_dw_input_fn(shape, dtype, device):
    """Yields (X, DY, INV_RMS) for the dW kernel given a (M, N) shape.

    The dW kernel only operates on 2D (M, N) inputs, but ``core_shapes.yaml``
    (loaded by the base ``Benchmark`` via ``set_shapes``) contains shapes of
    other ranks shared across all operators. Non-2D shapes are skipped here
    rather than yielding nothing, so callers see an empty-but-valid iterator
    for those shapes.
    """
    if len(shape) != 2:
        return
    M, N = shape
    x = torch.randn(shape, dtype=dtype, device=device)
    dy = torch.randn(shape, dtype=dtype, device=device)
    inv_rms = torch.rand(M, dtype=torch.float32, device=device) + 0.5
    yield x, dy, inv_rms


class RmsNormDwTleBenchmark(base.Benchmark):
    """Benchmark comparing the baseline dW kernel vs. the TLE set_layout
    variant. ``torch_op`` holds the baseline kernel, ``gems_op`` (set via
    ``set_gems``) holds the TLE-optimized kernel. Both are launched via
    ``get_latency``, which is overridden to use Triton's ``kernel[grid]``
    launch syntax instead of a plain function call.
    """

    DEFAULT_SHAPES = [
        (1024, 4096),
        (2048, 4096),
        (4096, 4096),
    ]
    DEFAULT_SHAPE_DESC = "M, N"

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_shapes(self, shape_file_path=None):
        # This kernel only accepts 2D (M, N) inputs and isn't registered in
        # core_shapes.yaml, so skip the generic yaml-driven shape loading
        # (used by other operators) and always use our own fixed 2D shapes.
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def unpack_to_args_kwargs(self, input_tuple):
        x, dy, inv_rms = input_tuple
        return [x, dy, inv_rms], {}

    def record_shapes(self, x, dy, inv_rms):
        return [tuple(x.shape), tuple(dy.shape), tuple(inv_rms.shape)]

    def _launch(self, kernel, x, dy, inv_rms, extra_kwargs):
        M, N = x.shape
        row_block_size = _DW_ROW_BLOCK_SIZE
        col_block_size = _DW_COL_BLOCK_SIZE
        row_block_num = triton.cdiv(M, row_block_size)
        col_block_num = triton.cdiv(N, col_block_size)
        dw = torch.empty(row_block_num, N, dtype=torch.float32, device=x.device)
        grid = (row_block_num, col_block_num)

        kernel[grid](
            x,
            dy,
            inv_rms,
            dw,
            N,
            1,
            N,
            1,
            M,
            N,
            row_block_size,
            col_block_size,
            num_warps=_DW_TLE_NUM_WARPS,
            **extra_kwargs,
        )
        return dw

    def get_latency(self, op, *args, **kwargs):
        x, dy, inv_rms = args
        extra_kwargs = (
            {"TARGET_LAYOUT": _DW_TARGET_LAYOUT}
            if op is rms_norm_grad_dw_kernel_tle
            else {}
        )
        fn = lambda: self._launch(op, x, dy, inv_rms, extra_kwargs)

        do_bench = triton.testing.do_bench
        latency = do_bench(
            fn,
            warmup=Config.warm_up,
            rep=Config.repetition,
            return_mode="median",
        )
        return latency


@pytest.mark.skipif(not HAS_TLE, reason="requires triton.experimental.tle")
@pytest.mark.rms_norm_dw_tle
def test_rms_norm_dw_tle():
    bench = RmsNormDwTleBenchmark(
        op_name="rms_norm_dw_tle",
        torch_op=rms_norm_grad_dw_kernel,
        input_fn=rms_norm_dw_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(rms_norm_grad_dw_kernel_tle)
    bench.run()
