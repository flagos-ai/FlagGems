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


@pytest.mark.all
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_all():
    bench = base.UnaryReductionBenchmark(
        op_name="all", torch_op=torch.all, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.all_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_all_dim():
    bench = base.UnaryReductionBenchmark(
        op_name="all_dim", torch_op=torch.all, dtypes=consts.FLOAT_DTYPES
    )
    bench.run()


@pytest.mark.all_dims
def test_all_dims():
    def all_dims_input_fn(shape, dtype, device):
        inp = torch.randn(shape, dtype=dtype, device=device)
        yield inp, {"dim": [0, 1]}

    bench = base.GenericBenchmarkExcluse1D(
        input_fn=all_dims_input_fn,
        op_name="all_dims",
        torch_op=torch.all,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# Frozen API signatures. Input generation/transfers, compilation and warmup
# precede steady-state timing; each call still includes output allocation.
@pytest.mark.all_dim
@pytest.mark.skipif(flag_gems.vendor_name != "hygon", reason="Hygon vendor override")
@pytest.mark.parametrize(
    "shape,dim,keepdim,layout",
    [
        ((256, 128), -1, False, "contiguous"),
        ((257, 129), -1, True, "contiguous"),
        ((1024, 256), 1, False, "contiguous"),
        ((8, 128, 513), -1, False, "contiguous"),
        ((1024, 1024), -1, False, "contiguous"),
        ((65536, 128), -1, False, "contiguous"),
        ((256, 127), -1, False, "contiguous"),
        ((256, 1025), -1, False, "contiguous"),
        ((256, 128), 0, False, "contiguous"),
        ((128, 256), -1, False, "transpose"),
    ],
)
def test_hygon_all_dim_contiguous(shape, dim, keepdim, layout):
    class FixedBenchmark(base.GenericBenchmark):
        def set_shapes(self, shape_file=None):
            self.shapes = [shape]

    def input_fn(shape, dtype, device):
        generator = torch.Generator().manual_seed(20261104)
        cpu = torch.randint(1, 9, shape, generator=generator).to(dtype)
        cpu.reshape(-1, shape[-1])[::4, -1] = 0
        inp = cpu.to(device)
        if layout == "transpose":
            inp = inp.T
        yield inp, {"dim": dim, "keepdim": keepdim}

    bench = FixedBenchmark(
        input_fn=input_fn, op_name="all_dim", torch_op=torch.all, dtypes=[torch.float32]
    )
    bench.set_gems(flag_gems.all_dim)
    bench.run()
