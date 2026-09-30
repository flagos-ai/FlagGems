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
from .generated_operator_utils import OperatorBenchmark

# aten::mkldnn_max_pool3d reads MkldnnCPU blocking-layout storage, so reference and
# candidate both work on host tensors and the reported speedup is host-vs-host (prefer
# --mode wrapper, since event timing cannot observe work that never reaches an
# accelerator). The NCDHW volumes below are the original pooling workloads; kernel,
# stride and padding stay cubic and dilation is always 1, the only value the native
# primitive accepts.
MKL_SHAPES = [
    (4, 3, 16, 56, 56),
    (8, 64, 8, 28, 28),
    (16, 128, 4, 14, 14),
    (32, 256, 2, 7, 7),
    (2, 512, 4, 7, 7),
]

DILATION = [1, 1, 1]
CONFIGS = [
    ([2, 2, 2], [2, 2, 2], [0, 0, 0], False),
    ([3, 3, 3], [1, 1, 1], [1, 1, 1], True),
]


def _case_fn(shape, dtype):
    del dtype
    for kernel_size, stride, padding, ceil_mode in CONFIGS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "dilation": DILATION,
                "ceil_mode": ceil_mode,
            },
            builder_args=(shape, kernel_size, stride, padding, ceil_mode),
        )


def _build_inputs_fn(plan, dtype, device):
    del device
    shape, kernel_size, stride, padding, ceil_mode = plan.builder_args
    # The trailing dict is unpacked as call kwargs by Benchmark.unpack_to_args_kwargs.
    return (
        torch.empty(shape, dtype=dtype).to_mkldnn(),
        {
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": DILATION,
            "ceil_mode": ceil_mode,
        },
    )


class MkldnnMaxPool3dBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Ordinary shared loading keeps an explicit --shape-file entry authoritative; a
        # 3-D pooling primitive indexes a 5-D NCDHW volume, so only rank-5 descriptors
        # can run and the recorded NCDHW volumes are unioned in afterwards.
        super().set_shapes(shape_file_path)
        self.shapes = list(
            dict.fromkeys(
                [tuple(shape) for shape in self.shapes if len(shape) == 5] + MKL_SHAPES
            )
        )


@pytest.mark.mkldnn_max_pool3d
def test_mkldnn_max_pool3d():
    bench = MkldnnMaxPool3dBenchmark(
        op_name="mkldnn_max_pool3d",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_max_pool3d,
        gems_op=getattr(flag_gems, "mkldnn_max_pool3d", None),
        dtypes=consts.FLOAT_DTYPES + [torch.int8, torch.uint8],
    )
    bench.run()
