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
from .test_linalg_polar import (
    _assert_output_metadata,
    _empty_with_dtype,
    _reference_linalg_polar,
)

POLAR_SHAPES = (
    [(8, 4)]
    if cfg.QUICK_MODE
    else [
        (3, 2),
        (8, 4),
        (16, 8),
        (2, 8, 4),
        (8, 16, 8),
    ]
)
ATOL = 1e-2


def _assert_polar_properties(inp, polar_u, polar_h):
    inp = inp.detach().cpu()
    polar_u = polar_u.detach().cpu()
    polar_h = polar_h.detach().cpu()
    utils.gems_assert_close(polar_u @ polar_h, inp, torch.float32, atol=ATOL)
    utils.gems_assert_close(polar_h, polar_h.mT, torch.float32, atol=ATOL)


@pytest.mark.linalg_polar_out
@pytest.mark.parametrize("shape", POLAR_SHAPES)
def test_linalg_polar_out_resize(shape):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = inp.detach().cpu().to(torch.float64)
    _, ref_h = _reference_linalg_polar(ref_inp)
    out_u = torch.empty(0, dtype=inp.dtype, device=inp.device)
    out_h = torch.empty(0, dtype=inp.dtype, device=inp.device)

    result_u, result_h = flag_gems.linalg_polar_out(inp, U=out_u, H=out_h)
    _assert_output_metadata(inp, result_u, result_h)

    assert result_u is out_u
    assert result_h is out_h
    assert result_u.shape == inp.shape
    assert result_h.shape == (*inp.shape[:-2], inp.shape[-1], inp.shape[-1])
    utils.gems_assert_close(result_h.cpu(), ref_h, torch.float32, atol=ATOL)
    _assert_polar_properties(ref_inp, result_u, result_h)


@pytest.mark.linalg_polar_out
def test_linalg_polar_out_noncontiguous_buffers():
    torch.manual_seed(0)
    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    ref_inp = inp.detach().cpu().to(torch.float64)
    out_u = torch.empty((4, 8), device=flag_gems.device).mT
    out_h = torch.empty((4, 4), device=flag_gems.device).mT
    u_stride = out_u.stride()
    h_stride = out_h.stride()
    u_pointer = out_u.data_ptr()
    h_pointer = out_h.data_ptr()

    result_u, result_h = flag_gems.linalg_polar_out(inp, U=out_u, H=out_h)
    _assert_output_metadata(inp, result_u, result_h)

    assert result_u is out_u
    assert result_h is out_h
    assert result_u.data_ptr() == u_pointer
    assert result_h.data_ptr() == h_pointer
    assert result_u.stride() == u_stride
    assert result_h.stride() == h_stride
    _assert_polar_properties(ref_inp, result_u, result_h)


@pytest.mark.linalg_polar_out
def test_linalg_polar_out_rejects_wrong_dtype():
    torch.manual_seed(0)
    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    out_u = _empty_with_dtype(inp.shape, torch.float64, inp.device)
    out_h = torch.empty((4, 4), dtype=torch.float32, device=inp.device)

    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems.linalg_polar_out(inp, U=out_u, H=out_h)


@pytest.mark.linalg_polar_out
def test_linalg_polar_out_rejects_autograd():
    torch.manual_seed(0)
    inp = torch.randn((8, 4), dtype=torch.float32, device=flag_gems.device)
    inp.requires_grad_(True)
    out_u = torch.empty_like(inp)
    out_h = torch.empty((4, 4), dtype=inp.dtype, device=inp.device)

    with pytest.raises(RuntimeError, match="out="):
        flag_gems.linalg_polar_out(inp, U=out_u, H=out_h)
