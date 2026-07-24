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
                inp = torch.randint(
                    -100, 100, shape, dtype=self.src_dtype, device=self.device
                )
            elif self.src_dtype == torch.uint8:
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
                op_name=f"to_copy_{src_dtype}_to_{dst_dtype}",
                torch_op=torch.ops.aten._to_copy,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in float_dtypes:
        for dst_dtype in int_dtypes:
            bench = ToCopyBenchmark(
                op_name=f"to_copy_{src_dtype}_to_{dst_dtype}",
                torch_op=torch.ops.aten._to_copy,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in float_dtypes:
        for dst_dtype in uint_dtypes:
            bench = ToCopyBenchmark(
                op_name=f"to_copy_{src_dtype}_to_{dst_dtype}",
                torch_op=torch.ops.aten._to_copy,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in int_dtypes:
        for dst_dtype in float_dtypes:
            bench = ToCopyBenchmark(
                op_name=f"to_copy_{src_dtype}_to_{dst_dtype}",
                torch_op=torch.ops.aten._to_copy,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()

    for src_dtype in int_dtypes:
        for dst_dtype in int_dtypes:
            if src_dtype == dst_dtype:
                continue
            bench = ToCopyBenchmark(
                op_name=f"to_copy_{src_dtype}_to_{dst_dtype}",
                torch_op=torch.ops.aten._to_copy,
                dtypes=[dst_dtype],
                src_dtype=src_dtype,
            )
            bench.run()
