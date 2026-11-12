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

from . import base, consts, utils


def _avg_pool3d_kwargs(shape):
    """Enumerate the avg_pool3d kwargs variants for a shape without allocating."""
    variants = [
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "ceil_mode": False,
            "count_include_pad": True,
            "divisor_override": None,
        }
    ]
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # With count_include_pad=False
        variants.append(
            {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": False,
                "count_include_pad": False,
                "divisor_override": None,
            }
        )
        # With ceil_mode
        variants.append(
            {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "ceil_mode": True,
                "count_include_pad": True,
                "divisor_override": None,
            }
        )
        # With divisor_override
        if shape[-3] >= 2 and shape[-2] >= 2 and shape[-1] >= 2:
            variants.append(
                {
                    "kernel_size": 2,
                    "stride": 1,
                    "padding": 0,
                    "ceil_mode": False,
                    "count_include_pad": True,
                    "divisor_override": 3,
                }
            )
    return variants


def avg_pool3d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    for kwargs in _avg_pool3d_kwargs(shape):
        yield inp, kwargs


_AVG_POOL3D_SHAPES_5D = [
    (4, 3, 16, 56, 56),
    (8, 64, 8, 28, 28),
    (16, 128, 4, 14, 14),
    (32, 256, 4, 7, 7),
]


class AvgPool3dBenchmark(base.GenericBenchmark):
    def get_case_iter(self, dtype) -> Generator:
        ordinal = 0
        for shape in _AVG_POOL3D_SHAPES_5D:
            for variant, kwargs in enumerate(_avg_pool3d_kwargs(shape)):
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": list(shape)},
                        params={"variant": variant},
                        builder_args=(shape, variant),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, variant = plan.builder_args
        items = list(self.input_fn(shape, case.dtype, self.device))
        return items[variant]


@pytest.mark.avg_pool3d
def test_perf_avg_pool3d():
    bench = AvgPool3dBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d",
        torch_op=torch.ops.aten.avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class AvgPool3dBackwardBenchmark(AvgPool3dBenchmark):
    def build_inputs(self, case):
        if base.vendor_name == "enflame" and case.dtype in (torch.float16, torch.bfloat16):
            pytest.skip(f"Enflame native avg_pool3d_backward does not support {case.dtype}")
        inp, kwargs = super().build_inputs(case)
        inp = inp.clone().requires_grad_(True)
        def triple(value):
            return (value,) * 3 if isinstance(value, int) else tuple(value)
        kernel = triple(kwargs["kernel_size"])
        stride = triple(kwargs["stride"])
        padding = triple(kwargs["padding"])
        spatial = []
        for length, k, step, pad in zip(inp.shape[-3:], kernel, stride, padding):
            rounding = step - 1 if kwargs["ceil_mode"] else 0
            output = (length + 2 * pad - k + rounding) // step + 1
            if kwargs["ceil_mode"] and (output - 1) * step >= length + pad:
                output -= 1
            spatial.append(output)
        grad_output = torch.randn(
            (*inp.shape[:-3], *spatial), dtype=inp.dtype, device=inp.device
        )
        return (
            grad_output, inp, list(kernel), list(stride), list(padding),
            kwargs["ceil_mode"], kwargs["count_include_pad"], kwargs["divisor_override"],
        )


@pytest.mark.avg_pool3d_backward
def test_perf_avg_pool3d_backward():
    bench = AvgPool3dBackwardBenchmark(
        input_fn=avg_pool3d_input_fn,
        op_name="avg_pool3d_backward",
        torch_op=torch.ops.aten.avg_pool3d_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
