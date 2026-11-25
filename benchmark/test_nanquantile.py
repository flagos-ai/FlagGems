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


class NanquantileBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1,),
            (17,),
            (1024,),
            (1025,),
            (128, 128),
            (512, 512),
            (1024, 1024),
        ]
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes += [(33, 65), (4096,), (256, 1024), (4096, 1024)]


def _make_input(shape, dtype, device):
    inp = torch.randn(shape, dtype=dtype, device=device)
    inp.reshape(-1)[::5] = float("nan")
    return inp


def _tensor_input(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    q = torch.tensor([0.1, 0.5, 0.9], dtype=dtype, device=device)
    yield inp, q, -1


def _scalar_input(shape, dtype, device):
    yield _make_input(shape, dtype, device), 0.37, -1


def _tensor_out_input(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    q = torch.tensor([0.1, 0.5, 0.9], dtype=dtype, device=device)
    out_shape = (q.numel(), *shape[:-1])
    yield inp, q, -1, False, {"out": torch.empty(out_shape, dtype=dtype, device=device)}


def _scalar_out_input(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    yield inp, 0.37, -1, False, {
        "out": torch.empty(shape[:-1], dtype=dtype, device=device)
    }


def _run(op_name, input_fn, torch_op, gems_op):
    bench = NanquantileBenchmark(
        op_name=op_name,
        input_fn=input_fn,
        torch_op=torch_op,
        gems_op=gems_op,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.nanquantile
def test_nanquantile():
    _run(
        "nanquantile",
        _tensor_input,
        torch.ops.aten.nanquantile.default,
        flag_gems.nanquantile,
    )


@pytest.mark.nanquantile_scalar
def test_nanquantile_scalar():
    _run(
        "nanquantile_scalar",
        _scalar_input,
        torch.ops.aten.nanquantile.scalar,
        flag_gems.nanquantile_scalar,
    )


@pytest.mark.nanquantile_out
def test_nanquantile_out():
    _run(
        "nanquantile_out",
        _tensor_out_input,
        torch.ops.aten.nanquantile.out,
        flag_gems.nanquantile_out,
    )


@pytest.mark.nanquantile_scalar_out
def test_nanquantile_scalar_out():
    _run(
        "nanquantile_scalar_out",
        _scalar_out_input,
        torch.ops.aten.nanquantile.scalar_out,
        flag_gems.nanquantile_scalar_out,
    )
