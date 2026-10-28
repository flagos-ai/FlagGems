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

# view_as_complex_copy maps a real dtype to the complex dtype one size up:
# float16 -> complex32, float32 -> complex64, float64 -> complex128. bfloat16
# and every non-float dtype are rejected, so the real domains here are the
# three supported float types.
VIEW_AS_COMPLEX_FLOAT_DTYPES = [torch.float16, torch.float32]
if not cfg.QUICK_MODE and utils.fp64_is_supported:
    VIEW_AS_COMPLEX_FLOAT_DTYPES.append(torch.float64)
VIEW_AS_COMPLEX_DTYPES = VIEW_AS_COMPLEX_FLOAT_DTYPES

if cfg.QUICK_MODE:
    # (input_shape,) tuples whose last dimension is 2.
    VIEW_AS_COMPLEX_SHAPES = [
        (4, 2),
        (2, 5, 2),
    ]
else:
    # (input_shape,) tuples; the last dimension is always 2. Covers 0-dim
    # scalar output ([2] -> ()), 1-D/2-D/3-D/5-D inputs. Empty tensors are
    # covered by the dedicated test_accuracy_view_as_complex_copy_empty, and
    # non-contiguous / high-rank inputs that need the strided path by the
    # dedicated tests below (the contig path is a flat rank-independent copy).
    VIEW_AS_COMPLEX_SHAPES = [
        (2,),
        (4, 2),
        (10, 8, 2),
        (2, 5, 2),
        (2, 3, 4, 2),
        (2, 2, 2, 3, 2),
        (1, 2),
    ]


def _as_complex64(t):
    # complex32 has no CPU ``equal`` kernel, so comparisons of the float16
    # case go through a lossless complex64 upcast; everything else is left
    # untouched.
    if t.dtype == torch.complex32:
        return t.to(torch.complex64)
    return t


def _assert_matches(res, ref, inp):
    # dtype -- the reinterpret must produce the mapping complex32/64/128 ->
    # float16/32/64; a wrong out dtype would fail the value check only after a
    # lossy comparison, so it is pinned explicitly.
    assert res.dtype == ref.dtype, f"dtype {res.dtype} != {ref.dtype}"
    # shape -- size[:-1] with the last dim folded away; a wrong rank/extent is
    # a wrong reinterpret even if values match.
    assert tuple(res.shape) == tuple(ref.shape)
    # is_contiguous -- the output must be a fresh contiguous tensor like aten.
    assert res.is_contiguous()
    # _is_alias_of -- the copy must never share storage with the input;
    # gems_assert_equal below compares values only and would pass for an
    # aliased view, so this is not redundant with it.
    assert not torch._C._is_alias_of(res, inp)
    # values -- full elementwise equality against the reference.
    utils.gems_assert_equal(_as_complex64(res), _as_complex64(ref))


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("input_shape", VIEW_AS_COMPLEX_SHAPES)
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy(input_shape, dtype):
    inp = torch.randn(input_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)

    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_dtype_mapping(dtype):
    inp = torch.randn(6, 2, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.view_as_complex_copy(inp)

    expected = {
        torch.float16: torch.complex32,
        torch.float32: torch.complex64,
        torch.float64: torch.complex128,
    }[dtype]
    assert res_out.dtype == expected
    assert res_out.shape == (6,)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_values(dtype):
    # The reinterpret must be bit-exact: real part = last dim index 0, imag
    # part = last dim index 1.
    inp = torch.randn(5, 4, 2, dtype=dtype, device=flag_gems.device)
    res = flag_gems.view_as_complex_copy(inp)
    real = inp[..., 0].to(torch.float32)
    imag = inp[..., 1].to(torch.float32)
    assert torch.equal(res.real.to(torch.float32), real)
    assert torch.equal(res.imag.to(torch.float32), imag)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_noncontiguous_batch(dtype):
    # Non-contiguous batch dims are allowed as long as the last dim keeps
    # stride 1; the strided gather path must honor the real strides.
    pool = torch.randn(8, 6, 2, dtype=dtype, device=flag_gems.device)
    inp = pool[::2]
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_transposed(dtype):
    # Transpose keeps the last dimension's stride 1 while scrambling the rest.
    base = torch.randn(3, 5, 2, dtype=dtype, device=flag_gems.device)
    inp = base.transpose(0, 1)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_high_rank_noncontiguous(dtype):
    # 4-D/5-D/6-D non-contiguous inputs: only non-contiguous inputs reach
    # _view_as_complex_copy_strided_kernel, whose leading-dim stride slots are
    # the rank-dependent code path. Every VIEW_AS_COMPLEX_SHAPES entry is
    # contiguous and takes the flat, rank-independent contig kernel, so this
    # is the only coverage of the strided decomposition for rank > 2 (the
    # highest rank accepted is 6, i.e. rank == _MAX_LEAD_RANK + 1).
    base4 = torch.randn(4, 8, 6, 2, dtype=dtype, device=flag_gems.device)
    base5 = torch.randn(4, 6, 8, 6, 2, dtype=dtype, device=flag_gems.device)
    base6 = torch.randn(2, 4, 6, 3, 5, 2, dtype=dtype, device=flag_gems.device)
    inputs = [
        base4[:, ::2, :, :],  # 4-D stride-2 batch
        base4.transpose(1, 2),  # 4-D transposed
        base5[:, ::2, :, ::2, :],  # 5-D strided
        base5.transpose(2, 3),  # 5-D transposed
        base6.transpose(1, 3),  # 6-D transposed (highest accepted rank)
    ]
    for inp in inputs:
        assert not inp.is_contiguous()
        assert inp.stride(-1) == 1
        ref_inp = utils.to_reference(inp)
        ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
        res_out = flag_gems.view_as_complex_copy(inp)
        _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_expanded(dtype):
    # Zero-stride batch dims must be materialized into concrete copies.
    base = torch.randn(1, 2, dtype=dtype, device=flag_gems.device)
    inp = base.expand(4, 2)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_storage_offset(dtype):
    base = torch.randn(20, dtype=dtype, device=flag_gems.device)
    inp = base[4:12].view(4, 2)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten.view_as_complex_copy(ref_inp)
    res_out = flag_gems.view_as_complex_copy(inp)
    _assert_matches(res_out, ref_out, inp)


@pytest.mark.view_as_complex_copy
@pytest.mark.parametrize("dtype", VIEW_AS_COMPLEX_DTYPES)
def test_accuracy_view_as_complex_copy_independent_storage(dtype):
    inp = torch.randn(8, 2, dtype=dtype, device=flag_gems.device)
    res_out = flag_gems.view_as_complex_copy(inp)
    ref = utils.to_reference(res_out.clone())

    inp.add_(1.0)
    # The copy must not change when the source is mutated in place.
    utils.gems_assert_equal(_as_complex64(res_out), _as_complex64(ref))


@pytest.mark.view_as_complex_copy
def test_accuracy_view_as_complex_copy_empty():
    inp = torch.randn(0, 2, device=flag_gems.device)
    res_out = flag_gems.view_as_complex_copy(inp)
    assert res_out.shape == (0,)
    assert res_out.dtype == torch.complex64
    ref_out = torch.ops.aten.view_as_complex_copy(utils.to_reference(inp))
    _assert_matches(res_out, ref_out, inp)

    inp2 = torch.randn(3, 0, 2, device=flag_gems.device)
    res2 = flag_gems.view_as_complex_copy(inp2)
    assert res2.shape == (3, 0)
    ref2 = torch.ops.aten.view_as_complex_copy(utils.to_reference(inp2))
    _assert_matches(res2, ref2, inp2)


@pytest.mark.view_as_complex_copy
def test_accuracy_view_as_complex_copy_error_paths():
    dev = flag_gems.device

    # 0-dim input.
    with pytest.raises(RuntimeError):
        torch.ops.aten.view_as_complex_copy(torch.tensor(1.0, device=dev))
    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(torch.tensor(1.0, device=dev))

    # Last dimension different from 2.
    with pytest.raises(RuntimeError):
        torch.ops.aten.view_as_complex_copy(torch.randn(4, 3, device=dev))
    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(torch.randn(4, 3, device=dev))

    # Last dimension stride different from 1.
    bad = torch.randn(4, 4, device=dev)[:, ::2]
    with pytest.raises(RuntimeError):
        torch.ops.aten.view_as_complex_copy(bad)
    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(bad)

    # Unsupported dtypes: bfloat16 and integers.
    with pytest.raises(RuntimeError):
        torch.ops.aten.view_as_complex_copy(
            torch.randn(4, 2, device=dev, dtype=torch.bfloat16)
        )
    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(
            torch.randn(4, 2, device=dev, dtype=torch.bfloat16)
        )
    with pytest.raises(RuntimeError):
        flag_gems.view_as_complex_copy(torch.randint(0, 4, (4, 2), device=dev))

    # rank - 1 > _MAX_LEAD_RANK (5): a 7-D input. The guard lives on the
    # non-contiguous branch, so use a non-contiguous 7-D input; 6-D is the
    # highest accepted rank (covered by the high-rank strided test).
    rank7 = torch.randn(*([2] * 7), device=dev).transpose(1, 2)
    assert not rank7.is_contiguous()
    with pytest.raises(RuntimeError, match="at most 6 dimensions"):
        flag_gems.view_as_complex_copy(rank7)
