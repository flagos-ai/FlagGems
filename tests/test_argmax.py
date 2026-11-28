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

if cfg.QUICK_MODE:
    DIM_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
else:
    DIM_LIST = [0, 1]
    FLOAT_DTYPES = utils.FLOAT_DTYPES

EMPTY_SHAPES = [(0, 5), (3, 0, 4), (2, 5, 0), (0,)]


# TODO: There are some bugs in argmax with large size.
@pytest.mark.argmax
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES + EMPTY_SHAPES)
@pytest.mark.parametrize("dim", DIM_LIST + [None])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_argmax(shape, dim, keepdim, dtype):
    rank = len(shape)
    is_empty_tensor = any(d == 0 for d in shape)

    if dim is not None:
        if rank == 0 or dim >= rank or dim < -rank:
            # Skip invalid input combination - dimension out of bound for shape
            return

    if is_empty_tensor:
        if dim is None:
            # The dim parameter must be specified for empty tensor for PyTorch
            return

        dim_index = dim % rank
        if shape[dim_index] == 0:
            # Zero-sized dimension is invalid input for PyTorch
            return

    if is_empty_tensor:
        inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)

    ref_inp = utils.to_reference(inp)

    ref_out = torch.argmax(ref_inp, dim=dim, keepdim=keepdim)
    res_out = flag_gems.argmax(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out, ref_out)


# --- Kunlunxin P800 NaN index regression -----------------------------------
_P800_ONLY = pytest.mark.skipif(
    flag_gems.vendor_name != "kunlunxin",
    reason="P800/Kunlunxin-specific NaN index regression",
)
_NAN = float("nan")
_FULL_VOCAB = 248320


@pytest.mark.argmax
@_P800_ONLY
@pytest.mark.parametrize("M", [1, 2, 3, 4, 7, 8])
def test_argmax_all_nan_full_vocab(M):
    inp = torch.randn(M, _FULL_VOCAB, dtype=torch.float32, device=flag_gems.device)
    inp = inp.to(torch.bfloat16).float()
    inp[0].fill_(_NAN)
    if M > 2:
        inp[2].fill_(_NAN)

    out = flag_gems.argmax(inp, dim=-1)

    assert bool(((out >= 0) & (out < _FULL_VOCAB)).all()), f"OOV index: {out}"
    assert out[0].item() == 0
    if M > 2:
        assert out[2].item() == 0

    ref_inp = utils.to_reference(inp).cpu()
    ref = torch.argmax(ref_inp, dim=-1)
    finite = ~torch.isnan(ref_inp).any(dim=-1)
    utils.gems_assert_equal(out[finite.to(out.device)], ref[finite])


@pytest.mark.argmax
@_P800_ONLY
@pytest.mark.parametrize("N", [8193, 131072])
def test_argmax_all_nan_mid_vocab(N):
    inp = torch.full((4, N), _NAN, dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.argmax(inp, dim=-1)
    assert bool(((out >= 0) & (out < N)).all())
    assert out[0].item() == 0


@pytest.mark.argmax
@_P800_ONLY
def test_argmax_partial_nan_is_in_range():
    inp = torch.randn(4, 8193, dtype=torch.float32, device=flag_gems.device)
    inp[0, 4000] = _NAN
    out = flag_gems.argmax(inp, dim=-1)
    assert bool(((out >= 0) & (out < 8193)).all())


@pytest.mark.argmax
@_P800_ONLY
def test_argmax_small_n_all_nan_is_in_range():
    inp = torch.full((4, 255), _NAN, dtype=torch.float32, device=flag_gems.device)
    out = flag_gems.argmax(inp, dim=-1)
    assert bool(((out >= 0) & (out < 255)).all())
