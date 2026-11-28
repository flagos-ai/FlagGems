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


def _aten_index_put_impl_backward(grad_output, self, indices, values, accumulate=False):
    """ATen formula of the ``_index_put_impl_`` backward (derivatives.yaml)."""
    if accumulate:
        grad_self = grad_output
    else:
        grad_self = torch.ops.aten.index_put(
            grad_output, indices, torch.zeros_like(values), False
        )
    grad_values = (
        grad_output[indices] if len(indices) == 1 else grad_output[tuple(indices)]
    )
    return grad_self, grad_values


def _input_fn(input_shape, idx_len, dtype, device):
    dev = flag_gems.device
    grad_output = torch.randn(input_shape, dtype=dtype, device=dev, requires_grad=False)
    values = torch.randn((idx_len,) + tuple(input_shape[1:]), dtype=dtype, device=dev)
    indices = [torch.randint(0, input_shape[0], (idx_len,), device=dev)]
    # The op signature is (grad_output, self, indices, values, accumulate):
    # ``self`` is unused by the backward formula.
    yield grad_output, None, indices, values, False


INDEX_PUT_IMPL_BWD_SHAPE = (
    ((1024, 1024), 1024),
    ((2048, 2048), 2048),
    ((512, 512, 512), 512),
    ((4096, 2048), 4096),
    ((4096, 128), 65536),
)


class IndexPutImplBackwardBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype):
        for input_shape, idx_len in INDEX_PUT_IMPL_BWD_SHAPE:
            yield from _input_fn(input_shape, idx_len, dtype, self.device)

    def set_more_shapes(self):
        return []


@pytest.mark.index_put_impl_backward_
def test_index_put_impl_backward():
    bench = IndexPutImplBackwardBenchmark(
        op_name="index_put_impl_backward_",
        torch_op=_aten_index_put_impl_backward,
        gems_op=flag_gems.index_put_impl_backward_,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
