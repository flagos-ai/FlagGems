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


class NestedSumBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        NESTED_SUM_BACKWARD_SHAPES = (
            (128, 256),
            (1024, 1024),
            (512, 1024, 512),
            (16, 8192, 4096),
            (8, 4096, 11008),
            (4, 32, 4096, 128),
            (32, 256, 256, 128),
        )

        self.shapes = NESTED_SUM_BACKWARD_SHAPES
        return None


def _torch_nested_sum_backward(grad, self, dim, keepdim):
    if not keepdim:
        d = dim[0] if isinstance(dim, (list, tuple)) else dim
        grad = grad.unsqueeze(d)
    return grad.expand_as(self).contiguous()


def _get_gbps(args, latency):
    grad, self, _, _ = args

    bytes_per_element = grad.element_size()
    total_bytes = (grad.numel() + self.numel()) * bytes_per_element

    return total_bytes / latency / 1e9


def _input_fn(shape, dtype, device):
    ndim = len(shape)

    for dim in [0, ndim // 2, -1]:
        actual_dim = dim + ndim if dim < 0 else dim

        if actual_dim >= ndim:
            continue

        for keepdim in (True, False):
            if keepdim:
                grad_shape = list(shape)
                grad_shape[actual_dim] = 1
            else:
                grad_shape = list(shape)
                grad_shape.pop(actual_dim)

            self_tensor = torch.randn(shape, dtype=dtype, device=device)
            grad = torch.randn(grad_shape, dtype=dtype, device=device)

            yield grad, self_tensor, [actual_dim], keepdim


@pytest.mark.nested_sum_backward
def test_nested_sum_backward():
    bench = NestedSumBackwardBenchmark(
        op_name="nested_sum_backward",
        torch_op=_torch_nested_sum_backward,
        gems_op=flag_gems._nested_sum_backward,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )

    bench.run()
