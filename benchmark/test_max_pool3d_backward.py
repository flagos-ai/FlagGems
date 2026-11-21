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

"""Backward performance for ``max_pool3d`` on the default core shapes."""

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils


def _pool3d_params(shape, level):
    """Window parameters used for one ``max_pool3d`` default shape."""
    params = [
        {
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
            "dilation": 1,
            "ceil_mode": False,
        }
    ]
    if level == consts.BenchLevel.COMPREHENSIVE:
        # Non-cubic kernel/stride/padding.
        if shape[-3] > 5 and shape[-2] > 5 and shape[-1] > 5:
            params.append(
                {
                    "kernel_size": (2, 3, 3),
                    "stride": (1, 2, 2),
                    "padding": (0, 1, 1),
                    "dilation": 1,
                    "ceil_mode": False,
                }
            )
        # Dilation: effective kernel (3-1)*2+1 = 5.
        if shape[-3] >= 4 and shape[-2] >= 4 and shape[-1] >= 4:
            params.append(
                {
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 2,
                    "ceil_mode": False,
                }
            )
        # ceil_mode rounds the output shape up instead of down.
        params.append(
            {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "dilation": 1,
                "ceil_mode": True,
            }
        )
    return params


class MaxPool3dBackwardBenchmark(base.GenericBenchmark):
    """Backward benchmark for the value-only ``max_pool3d`` forward.

    The default core shapes are the ones ``MaxPool3dBenchmark`` already uses;
    every case builds the forward graph outside the timed closure and times
    ``torch.autograd.grad`` with ``retain_graph=True``.
    """

    def get_input_iter(self, dtype) -> Generator:
        shapes_5d = [
            (4, 3, 16, 56, 56),
            (8, 64, 8, 28, 28),
            (16, 128, 4, 14, 14),
            (32, 256, 2, 7, 7),
        ]
        for shape in shapes_5d:
            for params in _pool3d_params(shape, base.Config.bench_level):
                yield _max_pool3d_backward_input(shape, dtype, self.device, params)


def _max_pool3d_backward_input(shape, dtype, device, params):
    """Build one backward case: input tensor plus the pooling window.

    The forward graph is created by the timed operator itself, so the closure
    only contains the autograd call; ``requires_grad`` is set here because
    ``unpack_to_args_kwargs`` also sets it for every floating-point tensor.
    """
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, params


@pytest.mark.max_pool3d_backward
def test_max_pool3d_backward():
    bench = MaxPool3dBackwardBenchmark(
        op_name="max_pool3d",
        input_fn=_max_pool3d_backward_input,
        torch_op=torch.max_pool3d,
        gems_op=flag_gems.max_pool3d,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
