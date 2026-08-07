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


def _to_copy_dtype_pairs():
    base_dtypes = [torch.float16, torch.bfloat16]
    if fp64_is_supported:
        base_dtypes.append(torch.float64)

    float_dtypes = [torch.float32, torch.float16, torch.bfloat16]
    int_dtypes = [torch.int8, torch.int16, torch.int32, torch.int64]

    pairs = [
        (src_dtype, dst_dtype)
        for src_dtype in float_dtypes
        for dst_dtype in base_dtypes
        if src_dtype != dst_dtype
    ]
    pairs.extend(
        (src_dtype, dst_dtype)
        for src_dtype in float_dtypes
        for dst_dtype in int_dtypes + [torch.uint8]
    )
    pairs.extend(
        (src_dtype, dst_dtype) for src_dtype in int_dtypes for dst_dtype in float_dtypes
    )
    pairs.extend(
        (src_dtype, dst_dtype)
        for src_dtype in int_dtypes
        for dst_dtype in int_dtypes
        if src_dtype != dst_dtype
    )

    # XDNN cannot execute these two reference operations.  Put them first so
    # the final result recorded for the shared marker comes from a real run.
    unsupported_baselines = {
        (torch.bfloat16, torch.int16),
        (torch.int16, torch.bfloat16),
    }
    pairs.sort(key=lambda pair: pair not in unsupported_baselines)

    return [
        pytest.param(
            src_dtype,
            dst_dtype,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name == "kunlunxin"
                and (src_dtype, dst_dtype) in unsupported_baselines,
                reason="Kunlunxin XDNN baseline does not implement this dtype cast",
            ),
            id=f"{src_dtype}-{dst_dtype}",
        )
        for src_dtype, dst_dtype in pairs
    ]


class ToCopyBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

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
@pytest.mark.parametrize("src_dtype,dst_dtype", _to_copy_dtype_pairs())
def test_to_copy(src_dtype, dst_dtype):
    bench = ToCopyBenchmark(
        op_name=f"to_copy_{src_dtype}_to_{dst_dtype}",
        torch_op=torch.ops.aten._to_copy,
        dtypes=[dst_dtype],
        src_dtype=src_dtype,
    )
    bench.run()
