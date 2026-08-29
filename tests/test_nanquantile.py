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
from . import conftest as cfg

DTYPES = [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float64]
INTERPOLATIONS = (
    ["linear"]
    if cfg.QUICK_MODE
    else [
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    ]
)


def _input(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    flat = inp.reshape(-1)
    flat[::5] = float("nan")
    return inp


def _assert_close(result, reference, dtype, reduction_size):
    utils.gems_assert_close(
        result,
        reference,
        dtype,
        reduce_dim=reduction_size,
        equal_nan=True,
    )


@pytest.mark.nanquantile
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("dim, keepdim", [(None, False), (1, False), (-1, True)])
def test_nanquantile_tensor(dtype, interpolation, dim, keepdim):
    inp = _input((7, 33), dtype).T
    q = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=dtype, device=inp.device)
    ref_inp, ref_q = utils.to_reference(inp), utils.to_reference(q)
    reference = torch.ops.aten.nanquantile.default(
        ref_inp, ref_q, dim, keepdim, interpolation=interpolation
    )
    result = flag_gems.nanquantile(
        inp, q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    reduction_size = inp.numel() if dim is None else inp.shape[dim]
    _assert_close(result, reference, dtype, reduction_size)


@pytest.mark.nanquantile_scalar
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("dim, keepdim", [(None, False), (0, True), (-1, False)])
def test_nanquantile_scalar(dtype, interpolation, dim, keepdim):
    inp = _input((17, 19), dtype)
    ref_inp = utils.to_reference(inp)
    reference = torch.ops.aten.nanquantile.scalar(
        ref_inp, 0.37, dim, keepdim, interpolation=interpolation
    )
    result = flag_gems.nanquantile_scalar(
        inp, 0.37, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    reduction_size = inp.numel() if dim is None else inp.shape[dim]
    _assert_close(result, reference, dtype, reduction_size)


@pytest.mark.nanquantile_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("q_shape", [(), (3,), (0,)])
def test_nanquantile_out(dtype, q_shape):
    inp = _input((5, 13), dtype)
    q = torch.empty(q_shape, dtype=dtype, device=inp.device)
    if q.numel():
        values = torch.linspace(0.1, 0.9, q.numel(), dtype=dtype, device=inp.device)
        q.copy_(values.reshape(q_shape))
    ref_inp, ref_q = utils.to_reference(inp), utils.to_reference(q)
    ref_out = torch.empty(0, dtype=dtype, device=ref_inp.device)
    reference = torch.ops.aten.nanquantile.out(
        ref_inp, ref_q, 1, True, interpolation="linear", out=ref_out
    )
    out = torch.empty(0, dtype=dtype, device=inp.device)
    result = flag_gems.nanquantile_out(
        inp, q, dim=1, keepdim=True, interpolation="linear", out=out
    )
    assert result is out
    _assert_close(result, reference, dtype, inp.shape[1])


@pytest.mark.nanquantile_scalar_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_nanquantile_scalar_out(dtype, interpolation):
    inp = _input((11, 15), dtype)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.empty(0, dtype=dtype, device=ref_inp.device)
    reference = torch.ops.aten.nanquantile.scalar_out(
        ref_inp, 0.73, 0, False, interpolation=interpolation, out=ref_out
    )
    out = torch.empty((1,), dtype=dtype, device=inp.device)
    result = flag_gems.nanquantile_scalar_out(
        inp, 0.73, dim=0, interpolation=interpolation, out=out
    )
    assert result is out
    _assert_close(result, reference, dtype, inp.shape[0])


@pytest.mark.nanquantile
@pytest.mark.parametrize("size", [1, 1024, 1025, 4097])
def test_nanquantile_all_nan_and_path_boundaries(size):
    inp = torch.full((3, size), float("nan"), device=flag_gems.device)
    inp[1, 0] = -float("inf")
    inp[1, -1] = float("inf")
    q = torch.tensor([0.0, 0.5, 1.0], device=inp.device)
    reference = torch.nanquantile(utils.to_reference(inp), utils.to_reference(q), dim=1)
    result = flag_gems.nanquantile(inp, q, dim=1)
    _assert_close(result, reference, torch.float32, size)


@pytest.mark.nanquantile
def test_nanquantile_special_values_and_scalar_tensor_q():
    inp = torch.tensor(
        [[float("nan"), -0.0, 0.0, -float("inf"), float("inf"), 2.0]],
        device=flag_gems.device,
    )
    q = torch.tensor(0.5, device=inp.device)
    reference = torch.nanquantile(utils.to_reference(inp), utils.to_reference(q), dim=1)
    result = flag_gems.nanquantile(inp, q, dim=1)
    _assert_close(result, reference, torch.float32, inp.shape[1])
    result_valid = ~torch.isnan(result)
    reference_valid = ~torch.isnan(reference)
    utils.gems_assert_equal(
        torch.signbit(result)[result_valid],
        torch.signbit(reference)[reference_valid],
    )


@pytest.mark.nanquantile
def test_nanquantile_errors():
    inp = torch.randn((3, 5), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(
            inp.to(torch.float16), torch.tensor(0.5, device=inp.device)
        )
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(inp, torch.ones((1, 1), device=inp.device))
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(inp, torch.tensor(1.1, device=inp.device))
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(
            inp, torch.tensor(0.5, device=inp.device), interpolation="bad"
        )
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(torch.empty(0, device=inp.device), 0.5)
