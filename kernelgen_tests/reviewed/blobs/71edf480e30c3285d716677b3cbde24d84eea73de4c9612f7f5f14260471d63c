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

from . import base, consts


SHAPES = [
    ((64,), (64,), (64,), (64,)),
    ((4096,), (4096,), (4096,), (4096,)),
    ((8, 128), (8, 128), (8, 128), (8, 128)),
    ((2, 1024, 64), (2, 1024, 64), (2, 1024, 64), (2, 1024, 64)),
]


def _flat_to_per_dim_indices(flat_indices, inp_shape):
    strides, prod = [], 1
    for size in reversed(inp_shape):
        strides.append(prod)
        prod *= size
    return tuple((flat_indices // stride) % size for stride, size in zip(reversed(strides), inp_shape))


class UnsafeMaskedIndexPutAccumulateBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = SHAPES


def _case_fn(shape, dtype):
    del dtype
    inp_shape, mask_shape, indices_shape, values_shape = shape
    yield base.BenchmarkCasePlan(
        shape={"self": list(inp_shape), "mask": list(mask_shape), "indices": list(indices_shape), "values": list(values_shape)},
        params={"accumulate": True},
        builder_args=tuple(tuple(x) for x in shape),
    )


def _build_inputs_fn(plan, dtype, device):
    torch.manual_seed(0)
    inp_shape, mask_shape, indices_shape, values_shape = plan.builder_args
    inp = torch.randn(inp_shape, dtype=dtype, device=device)
    mask = torch.randint(0, 2, mask_shape, dtype=torch.int32, device=device)
    flat = torch.randint(0, max(inp.numel(), 1), indices_shape, device=device)
    values = torch.randn(values_shape, dtype=dtype, device=device)
    return inp, mask, _flat_to_per_dim_indices(flat, inp_shape), values


@pytest.mark.unsafe_masked_index_put_accumulate
def test_unsafe_masked_index_put_accumulate():
    bench = UnsafeMaskedIndexPutAccumulateBenchmark(
        op_name="unsafe_masked_index_put_accumulate",
        torch_op=torch._unsafe_masked_index_put_accumulate,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[d for d in consts.FLOAT_DTYPES if d != torch.bfloat16],
    )
    bench.run()
