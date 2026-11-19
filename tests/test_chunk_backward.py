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

from . import accuracy_utils as utils


@pytest.mark.chunk_backward
@pytest.mark.parametrize(
    "shape",
    [(64,), (128, 64), (1024, 1024), (64, 512, 512)],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("chunks", [2, 3, 7])
@pytest.mark.parametrize("dim", [0, -1])
def test_chunk_backward(shape, dtype, chunks, dim):
    """chunk's backward is a concatenation of the per-chunk gradients."""
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True).requires_grad_(True)
    res_inp = inp.clone().requires_grad_(True)

    ref_chunks = torch.chunk(ref_inp, chunks, dim)
    res_chunks = torch.chunk(res_inp, chunks, dim)
    assert len(ref_chunks) == len(res_chunks)

    res_grads = [
        torch.randn(c.shape, dtype=dtype, device=flag_gems.device) for c in res_chunks
    ]
    ref_grads = [utils.to_reference(g, True) for g in res_grads]

    (ref_in_grad,) = torch.autograd.grad(ref_chunks, ref_inp, ref_grads)
    res_in_grad = flag_gems.chunk_backward(res_grads, dim)

    utils.gems_assert_close(res_in_grad, ref_in_grad, dtype)
