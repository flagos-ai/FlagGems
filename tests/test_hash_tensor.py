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

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def _reference_hash(inp, dims, *, keepdim):
    if flag_gems.vendor_name != "iluvatar":
        return torch.ops.aten.hash_tensor(
            utils.to_reference(inp), dims, keepdim=keepdim, mode=0
        )

    # Iluvatar's CPU and device builds return uint32. For ATen mode 0,
    # XOR-reduce the 64-bit element representations on CPU without that truncation.
    inp = inp.cpu()
    bits = (
        inp.double().numpy().view(np.int64)
        if inp.is_floating_point()
        else inp.long().numpy()
    )
    hashed = np.bitwise_xor.reduce(bits, axis=tuple(dims) or None, keepdims=keepdim)
    return torch.from_numpy(np.asarray(hashed)).view(torch.uint64)


def _assert_equal(res_out, ref_out, input_device):
    """Compare two hash_tensor results, which are always uint64.

    Some device runtimes do not implement uint64 elementwise comparison
    (torch_xmlir's eq/equal reject it), so reinterpret the same bits as int64.
    The dtype view keeps the comparison bit-exact.
    """
    if flag_gems.vendor_name == "iluvatar":
        assert res_out.device == input_device
    if ref_out.device != res_out.device:
        res_out = res_out.cpu()
    utils.gems_assert_equal(res_out.view(torch.int64), ref_out.view(torch.int64))


@pytest.mark.hash_tensor
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [None, 0, -1])
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor(shape, dtype, dim, keepdim):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    # hash_tensor is bit-exact: never upcast the reference, or the hashed
    # bit patterns (and therefore the result) would differ.

    dim_arg = [] if dim is None else [dim]

    ref_out = _reference_hash(inp, dim_arg, keepdim=keepdim)
    res_out = flag_gems.hash_tensor(inp, dim_arg, keepdim=keepdim, mode=0)

    _assert_equal(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES + [torch.bool])
@pytest.mark.parametrize("dim", [None, -1])
def test_hash_tensor_int_dtypes(shape, dtype, dim):
    if dtype == torch.bool:
        inp = torch.randint(0, 2, shape, device=flag_gems.device).to(dtype)
    else:
        inp = torch.randint(-1000, 1000, shape, device=flag_gems.device).to(dtype)

    dim_arg = [] if dim is None else [dim]

    ref_out = _reference_hash(inp, dim_arg, keepdim=False)
    res_out = flag_gems.hash_tensor(inp, dim_arg, keepdim=False, mode=0)

    _assert_equal(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
@pytest.mark.parametrize(
    "shape,dim",
    [
        # Multi-axis reductions: trailing-contiguous, leading, and a
        # non-collapsible mix that forces the wrapper's materializing path.
        ((64, 128, 256), [1, 2]),
        ((16, 32, 64), [0, 1]),
        ((8, 16, 32, 64), [0, 2]),
    ],
)
@pytest.mark.parametrize("keepdim", [False, True])
def test_hash_tensor_multidim(shape, dim, keepdim):
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)

    ref_out = _reference_hash(inp, dim, keepdim=keepdim)
    res_out = flag_gems.hash_tensor(inp, dim, keepdim=keepdim, mode=0)

    _assert_equal(res_out, ref_out, inp.device)


@pytest.mark.hash_tensor
def test_hash_tensor_scalar():
    inp = torch.tensor(42, dtype=torch.int64, device=flag_gems.device)

    ref_out = _reference_hash(inp, [], keepdim=False)
    res_out = flag_gems.hash_tensor(inp, [], keepdim=False, mode=0)

    _assert_equal(res_out, ref_out, inp.device)
