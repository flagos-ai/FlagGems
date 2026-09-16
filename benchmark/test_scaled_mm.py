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

import pytest
import torch

import flag_gems

from . import base, consts

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name in ["ascend"],
    reason="Issue #3387: Not supported on Ascend",
)


def _cuda_fp8_available():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    if not flag_gems.runtime.device.support_bf16:
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89 and hasattr(torch, "float8_e4m3fn")


def _benchmark_cases():
    if _cuda_fp8_available() and hasattr(torch, "_scaled_mm"):
        return [
            (torch.float8_e4m3fn, "scalar", torch.float16, True),
            (torch.float8_e4m3fn, "scalar", torch.bfloat16, True),
            (torch.float8_e4m3fn, "scalar", torch.float32, False),
            (torch.float8_e4m3fn, "rowwise", torch.bfloat16, True),
        ]

    return []


def _case_id(case):
    dtype, scale_mode, out_dtype, use_bias = case
    return (
        f"{str(dtype).split('.')[-1]}-{scale_mode}-"
        f"{str(out_dtype).split('.')[-1]}-bias_{use_bias}"
    )


def _benchmark_case_params():
    return [pytest.param(case, id=_case_id(case)) for case in _benchmark_cases()]


class ScaledMMBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, op_name, torch_op, gems_op, case, use_out=False):
        dtype = case[0]
        super().__init__(op_name, torch_op, dtypes=[dtype])
        self.set_gems(gems_op)
        self.case = case
        self.use_out = use_out

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (16, 16, 16),
            (128, 128, 128),
            (512, 512, 512),
        ]
        self.shape_desc = "M, N, K"

    def set_more_shapes(self):
        return [
            (1024, 1024, 1024),
            (256, 4096, 4096),
        ]

    def get_input_iter(self, dtype):
        _, scale_mode, out_dtype, use_bias = self.case
        for M, N, K in self.shapes:
            mat1 = torch.randn((M, K), dtype=torch.float32, device=flag_gems.device)
            mat2 = torch.randn((K, N), dtype=torch.float32, device=flag_gems.device)
            mat1 = (mat1 * 0.25).to(dtype)
            mat2 = (mat2 * 0.25).to(dtype).t().contiguous().t()

            if scale_mode == "scalar":
                scale_a = torch.tensor([0.75], device=flag_gems.device)
                scale_b = torch.tensor([1.25], device=flag_gems.device)
            else:
                scale_a = torch.linspace(
                    0.75, 1.25, M, device=flag_gems.device
                ).reshape(M, 1)
                scale_b = torch.linspace(
                    1.25, 0.75, N, device=flag_gems.device
                ).reshape(1, N)

            bias = None
            if use_bias:
                bias = torch.randn((N,), dtype=out_dtype, device=flag_gems.device)
            if self.use_out:
                out = torch.empty((M, N), dtype=out_dtype, device=flag_gems.device)
                yield mat1, mat2, scale_a, scale_b, bias, None, out_dtype, False, {
                    "out": out
                }
            else:
                yield mat1, mat2, scale_a, scale_b, bias, None, out_dtype, False

    def get_tflops(self, op, *args, **kwargs):
        M, K = args[0].shape
        N = args[1].shape[1]
        return 2 * M * N * K


@pytest.mark.scaled_mm
@pytest.mark.parametrize("case", _benchmark_case_params())
def test_scaled_mm_benchmark(case):
    bench = ScaledMMBenchmark("scaled_mm", torch._scaled_mm, flag_gems.scaled_mm, case)
    bench.run()


@pytest.mark.scaled_mm_out
@pytest.mark.parametrize("case", _benchmark_case_params())
def test_scaled_mm_out_benchmark(case):
    bench = ScaledMMBenchmark(
        "scaled_mm_out",
        torch.ops.aten._scaled_mm.out,
        flag_gems.scaled_mm_out,
        case,
        use_out=True,
    )
    bench.run()
