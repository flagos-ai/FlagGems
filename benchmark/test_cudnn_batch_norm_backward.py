# Copyright 2026, The FlagOS Contributors.
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
#
import pytest
import torch

import flag_gems

from . import base, consts

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "mthreads", "hygon"),
    reason=(
        "native cudnn_batch_norm forward is required for saved statistics and "
        "reserve; Ascend falls back to CPU, MUSA has no native device kernel, "
        "and Hygon is not compiled with cuDNN support"
    ),
)

DTYPES = [torch.float16, torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)


class CudnnBatchNormBackwardBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (16, 16, 64),
        (16, 16, 1024),
        (16, 16, 4098),
        (1, 8, 4, 4),
        (16, 8, 128, 128),
    ]

    def set_shapes(self, shape_file_path=None):
        self.shapes = list(self.DEFAULT_SHAPES)
        self.shape_desc = "N, C, spatial dimensions"


def _inputs(shape, dtype, device):
    param_dtype = torch.float32 if dtype == torch.float16 else dtype
    with_running_stats = [False]
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        with_running_stats.append(True)
    for use_running_stats in with_running_stats:
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn(shape[1], dtype=param_dtype, device=device)
        bias = torch.randn_like(weight)
        running_mean = torch.zeros_like(weight) if use_running_stats else None
        running_var = torch.ones_like(weight) if use_running_stats else None
        grad_output = torch.randn_like(inp)
        momentum = 0.1
        eps = 1e-5
        _, save_mean, save_invstd, reserve = torch.ops.aten.cudnn_batch_norm.default(
            inp, weight, bias, running_mean, running_var, True, momentum, eps
        )
        yield (
            inp,
            grad_output,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_invstd,
            eps,
            reserve,
        )


@pytest.mark.cudnn_batch_norm_backward
def test_cudnn_batch_norm_backward():
    bench = CudnnBatchNormBackwardBenchmark(
        input_fn=_inputs,
        op_name="cudnn_batch_norm_backward",
        torch_op=torch.ops.aten.cudnn_batch_norm_backward.default,
        dtypes=DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_batch_norm_backward)
    bench.run()
