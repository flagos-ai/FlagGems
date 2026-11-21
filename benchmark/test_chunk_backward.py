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


def chunk_backward_case_fn(shape, dtype):
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"chunks": 3, "dim": 0},
        builder_args=(shape,),
    )


def chunk_backward_build_inputs_fn(plan, dtype, device):
    (shape,) = plan.builder_args
    chunks = plan.params["chunks"]
    dim = plan.params["dim"]
    inp = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
    chunks_out = torch.chunk(inp, chunks, dim)
    grads = [torch.randn_like(c) for c in chunks_out]
    return grads, dim


def chunk_backward_torch_op(grads, dim):
    # Chunk's backward is ATen's SplitBackward, which concatenates the incoming
    # gradients. Route the reference through ``torch.split`` because
    # ``aten::chunk`` is an IMPL-mode FlagGems override here and therefore has
    # no autograd formula ("the derivative for 'chunk' is not implemented").
    d = dim if dim >= 0 else dim + grads[0].dim()
    sizes = [g.shape[d] for g in grads]
    inp = torch.cat(grads, dim).detach().requires_grad_(True)
    chunks_out = torch.split(inp, sizes, dim)
    (out,) = torch.autograd.grad(chunks_out, inp, grads)
    return out


@pytest.mark.chunk_backward
def test_chunk_backward():
    bench = base.GenericBenchmarkExcluse1D(
        case_fn=chunk_backward_case_fn,
        build_inputs_fn=chunk_backward_build_inputs_fn,
        op_name="chunk_backward",
        torch_op=chunk_backward_torch_op,
        gems_op=flag_gems.chunk_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
