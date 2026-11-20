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

# Fixed plans keep the denominator stable across collection and timing. Shapes
# describe the returned input, including the noncontiguous layout variants.
FUSED_DROPOUT_CASES = [
    ((1,), "contiguous", 0.5),
    ((4097,), "contiguous", 0.5),
    ((96000,), "contiguous", 0.2),
    ((96000,), "contiguous", 0.5),
    ((96000,), "contiguous", 0.8),
    ((2**20,), "contiguous", 0.5),
    ((2**24,), "contiguous", 0.5),
    ((64, 257), "transpose", 0.5),
    ((4, 16, 32, 33), "channels_last", 0.5),
    ((128, 129), "slice", 0.5),
    ((33, 65), "expanded", 0.5),
]
FLOAT_DTYPES = consts.FLOAT_DTYPES + (
    [torch.float64]
    if flag_gems.runtime.device.support_fp64
    and flag_gems.vendor_name not in ("ascend", "iluvatar", "mthreads")
    else []
)


def _case_fn(case, dtype):
    del dtype
    shape, layout, p = case
    yield consts.BenchmarkCasePlan(
        shape={"input": shape},
        params={"layout": layout, "p": p},
        builder_args=(shape,),
    )


def _build_inputs(plan, dtype, device):
    shape = plan.builder_args[0]
    layout = plan.params["layout"]
    if layout == "transpose":
        inp = torch.randn(tuple(reversed(shape)), dtype=dtype, device=device).T
    elif layout == "slice":
        inp = torch.randn((shape[0], shape[1] * 2), dtype=dtype, device=device)[:, 1::2]
    elif layout == "expanded":
        inp = torch.randn((1, shape[1]), dtype=dtype, device=device).expand(shape)
    else:
        inp = torch.randn(shape, dtype=dtype, device=device)
        if layout == "channels_last":
            inp = inp.to(memory_format=torch.channels_last)
    return inp, plan.params["p"]


class FusedDropoutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(FUSED_DROPOUT_CASES)
        self.shape_desc = "input, layout, keep probability"


@pytest.mark.fused_dropout
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Native aten::_fused_dropout falls back to CPU on CANN850 and CANN900",
)
@pytest.mark.skipif(
    base.Config.mode == consts.BenchMode.CUDAGRAPH,
    reason="_fused_dropout does not support graph capture; use --mode kernel or operator",
)
def test_fused_dropout():
    bench = FusedDropoutBenchmark(
        op_name="_fused_dropout",
        torch_op=torch.ops.aten._fused_dropout.default,
        gems_op=flag_gems._fused_dropout,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
