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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts

fp64_is_supported = flag_gems.runtime.device.support_fp64

_IS_KUNLUNXIN = flag_gems.vendor_name == "kunlunxin"

# Kunlunxin's native xdnn cast kernel lacks a direct bf16<->int16 path
# (cast.cpp:163 "scalar type ... combined is unsupported"), which crashes the
# torch baseline and takes down the whole benchmark. The gems kernel handles
# these casts directly, so only the torch *reference* needs rerouting: bf16 is
# a subset of float32 and int16 fits exactly in float32, so going through
# float32 is a numerically identical two-step cast (verified bit-exact).
_KLX_CAST_VIA_FP32 = {
    (torch.bfloat16, torch.int16),
    (torch.int16, torch.bfloat16),
}


def _to_copy_ref(inp, *, dtype):
    if (inp.dtype, dtype) in _KLX_CAST_VIA_FP32:
        return torch.ops.aten._to_copy(inp.to(torch.float32), dtype=dtype)
    return torch.ops.aten._to_copy(inp, dtype=dtype)


_TO_COPY_TORCH_OP = _to_copy_ref if _IS_KUNLUNXIN else torch.ops.aten._to_copy
# Candidate calls the gems kernel directly (no per-iteration use_gems context,
# which would otherwise dominate the measured latency).
_TO_COPY_GEMS_OP = flag_gems.to_copy if _IS_KUNLUNXIN else None


class ToCopyBenchmark(base.Benchmark):
    # The tflops metric divides by candidate latency with NO EPS guard
    # (base.py:791-797), unlike speedup/gbps which add self.EPS. On Kunlunxin the
    # on-demand CI path runs the benchmark WITHOUT XPU_EVENT_KL3_ENABLE=1, so
    # do_bench silently returns a median latency of 0 on the tiny (1024,1) cast
    # (warm_up=1/rep=1), making that bare division raise ZeroDivisionError and
    # fail the whole test. tflops is a meaningless metric for a pointwise cast
    # anyway, so drop it on Kunlunxin only; other backends keep it unchanged.
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ([] if _IS_KUNLUNXIN else ["tflops"])

    def __init__(self, *args, src_dtype=torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self.src_dtype = src_dtype

    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            if self.src_dtype in [
                torch.float32,
                torch.float16,
                torch.bfloat16,
                torch.float64,
            ]:
                inp = torch.randn(shape, dtype=self.src_dtype, device=self.device)
            elif self.src_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
                if flag_gems.vendor_name == "cambricon":
                    # Cambricon torch.randint currently does not support int8/int16 generation.
                    inp = torch.randint(
                        -100, 100, shape, dtype=self.src_dtype, device="cpu"
                    ).to(self.device)
                else:
                    inp = torch.randint(
                        -100, 100, shape, dtype=self.src_dtype, device=self.device
                    )
            elif self.src_dtype == torch.uint8:
                if flag_gems.vendor_name == "cambricon":
                    # Cambricon torch.randint currently does not support int8/int16 generation.
                    inp = torch.randint(
                        0, 255, shape, dtype=self.src_dtype, device="cpu"
                    ).to(self.device)
                else:
                    inp = torch.randint(
                        0, 255, shape, dtype=self.src_dtype, device=self.device
                    )
            else:
                inp = torch.randn(shape, dtype=self.src_dtype, device=self.device)
            yield inp, {"dtype": dtype}

    def get_tflops(self, op, *args, **kwargs):
        shape = list(args[0].shape)
        return torch.tensor(shape).prod().item()


@pytest.mark.to_copy
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_to_copy():
    base_dtypes = [torch.float16, torch.bfloat16]
    if fp64_is_supported:
        base_dtypes.append(torch.float64)

    float_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    # if fp64_is_supported:
    #     float_dtypes.append(torch.float64)

    int_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]
    uint_dtypes = [torch.uint8]

    for src_dtype in float_dtypes:
        for dst_dtype in base_dtypes:
            if src_dtype == dst_dtype:
                continue

            bench = ToCopyBenchmark(
                op_name="to_copy",
                torch_op=_TO_COPY_TORCH_OP,
                gems_op=_TO_COPY_GEMS_OP,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in float_dtypes:
        for dst_dtype in int_dtypes:
            bench = ToCopyBenchmark(
                op_name="to_copy",
                torch_op=_TO_COPY_TORCH_OP,
                gems_op=_TO_COPY_GEMS_OP,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in float_dtypes:
        for dst_dtype in uint_dtypes:
            bench = ToCopyBenchmark(
                op_name="to_copy",
                torch_op=_TO_COPY_TORCH_OP,
                gems_op=_TO_COPY_GEMS_OP,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in int_dtypes:
        for dst_dtype in float_dtypes:
            bench = ToCopyBenchmark(
                op_name="to_copy",
                torch_op=_TO_COPY_TORCH_OP,
                gems_op=_TO_COPY_GEMS_OP,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in int_dtypes:
        for dst_dtype in int_dtypes:
            if src_dtype == dst_dtype:
                continue
            bench = ToCopyBenchmark(
                op_name="to_copy",
                torch_op=_TO_COPY_TORCH_OP,
                gems_op=_TO_COPY_GEMS_OP,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()
