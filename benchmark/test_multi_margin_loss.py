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

from typing import ClassVar

import pytest
import torch

import flag_gems

from . import base, consts


class MultiMarginLossBenchmark(base.Benchmark):
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"
    DEFAULT_SHAPES: ClassVar[list[tuple[int, int]]] = [
        (1, 64),
        (32, 64),
        (128, 256),
        (1024, 1000),
        (4096, 1000),
    ]
    DEFAULT_SHAPE_DESC = "N, C"

    def __init__(self, *, p, weighted, reduction, **kwargs):
        super().__init__(**kwargs)
        self.p = p
        self.weighted = weighted
        self.reduction = reduction

    def init_user_config(self):
        # This benchmark needs loss-specific [N, C] shapes; the generic
        # Benchmark entry in core_shapes.yaml is intentionally not applicable.
        self.mode = base.Config.mode
        self.set_dtypes(base.Config.user_desired_dtypes)
        self.set_metrics(base.Config.user_desired_metrics)
        self.shapes = self.DEFAULT_SHAPES
        self.shapes = [
            shape
            for shape in self.shapes
            if len(shape) == 2 and shape[0] > 0 and shape[1] > 0
        ]
        if not self.shapes:
            pytest.skip("multi_margin_loss benchmark requires positive [N, C] shapes")

    def get_input_iter(self, dtype):
        for N, C in self.shapes:
            input = torch.randn((N, C), dtype=dtype, device=self.device)
            target = torch.randint(0, C, (N,), dtype=torch.int64, device=self.device)
            weight = (
                torch.randn((C,), dtype=dtype, device=self.device)
                if self.weighted
                else None
            )
            yield input, target, self.p, 1.0, weight, self.reduction


class MultiMarginLossOutBenchmark(MultiMarginLossBenchmark):
    def get_input_iter(self, dtype):
        for args in super().get_input_iter(dtype):
            input = args[0]
            output_shape = (input.shape[0],) if self.reduction == 0 else ()
            out = torch.empty(output_shape, dtype=dtype, device=self.device)
            yield (*args, {"out": out})


class MultiMarginLossBackwardBenchmark(MultiMarginLossBenchmark):
    def get_input_iter(self, dtype):
        for args in super().get_input_iter(dtype):
            input = args[0]
            grad_output_shape = (input.shape[0],) if self.reduction == 0 else ()
            grad_output = torch.randn(
                grad_output_shape,
                dtype=dtype,
                device=self.device,
            )
            yield (grad_output, *args)


class MultiMarginLossBackwardOutBenchmark(MultiMarginLossBackwardBenchmark):
    def get_input_iter(self, dtype):
        for args in super().get_input_iter(dtype):
            grad_input = torch.empty_like(args[1])
            yield (*args, {"grad_input": grad_input})


_DTYPES = consts.FLOAT_DTYPES + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)

_ASCEND_NATIVE_FALLBACK = pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name == "ascend",
    reason="Ascend native multi_margin_loss family falls back to an invalid CPU baseline",
)


@pytest.mark.multi_margin_loss
@_ASCEND_NATIVE_FALLBACK
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multi_margin_loss(p, weighted, reduction):
    benchmark = MultiMarginLossBenchmark(
        p=p,
        weighted=weighted,
        reduction=reduction,
        op_name="multi_margin_loss",
        torch_op=torch.ops.aten.multi_margin_loss,
        gems_op=flag_gems.multi_margin_loss,
        dtypes=_DTYPES,
    )
    benchmark.run()


@pytest.mark.multi_margin_loss_out
@_ASCEND_NATIVE_FALLBACK
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multi_margin_loss_out(p, weighted, reduction):
    benchmark = MultiMarginLossOutBenchmark(
        p=p,
        weighted=weighted,
        reduction=reduction,
        op_name="multi_margin_loss_out",
        torch_op=torch.ops.aten.multi_margin_loss.out,
        gems_op=flag_gems.multi_margin_loss_out,
        dtypes=_DTYPES,
    )
    benchmark.run()


@pytest.mark.multi_margin_loss_backward
@_ASCEND_NATIVE_FALLBACK
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multi_margin_loss_backward(p, weighted, reduction):
    benchmark = MultiMarginLossBackwardBenchmark(
        p=p,
        weighted=weighted,
        reduction=reduction,
        op_name="multi_margin_loss_backward",
        torch_op=torch.ops.aten.multi_margin_loss_backward,
        gems_op=flag_gems.multi_margin_loss_backward,
        dtypes=_DTYPES,
    )
    benchmark.run()


@pytest.mark.multi_margin_loss_backward_out
@_ASCEND_NATIVE_FALLBACK
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multi_margin_loss_backward_out(p, weighted, reduction):
    benchmark = MultiMarginLossBackwardOutBenchmark(
        p=p,
        weighted=weighted,
        reduction=reduction,
        op_name="multi_margin_loss_backward_out",
        torch_op=torch.ops.aten.multi_margin_loss_backward.grad_input,
        gems_op=flag_gems.multi_margin_loss_backward_out,
        dtypes=_DTYPES,
    )
    benchmark.run()
