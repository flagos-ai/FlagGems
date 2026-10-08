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

# Shape configurations for slice_copy testing: 1D, 2D, 3D and higher-rank tensors.
SLICE_COPY_SHAPES = [
    (16, 32, 64),
    (32, 64),
    (64,),
    (4, 8, 12),
    (2, 19, 7),
]

# Every dtype aten::slice_copy accepts on CUDA. The kernel is a pure bit copy, so
# bool / float8 go through a uint8 reinterpretation and complex through
# view_as_real; both paths are covered here.
_FLOAT8_DTYPES = [
    getattr(torch, name)
    for name in ("float8_e4m3fn", "float8_e5m2")
    if hasattr(torch, name)
]
_BYTE_DTYPES = [torch.bool, torch.uint8, torch.int8]
_ALL_NON_COMPLEX_DTYPES = list(
    dict.fromkeys(
        utils.ALL_FLOAT_DTYPES + utils.ALL_INT_DTYPES + [torch.int16] + _BYTE_DTYPES
    )
)
_COMPLEX_DTYPES = (
    [torch.complex64]
    if cfg.QUICK_MODE
    else [torch.complex32, torch.complex64, torch.complex128]
)


def _make_input(shape, dtype, device=None):
    """Build a deterministic, dtype-appropriate input.

    For float8 the values are constructed from raw bytes so that every exponent
    and the NaN encodings are exercised; the copy must be bit exact.
    """
    device = device or flag_gems.device
    numel = 1
    for s in shape:
        numel *= s
    if dtype == torch.bool:
        return (torch.arange(numel, device=device) % 3 == 0).reshape(shape)
    if dtype in _FLOAT8_DTYPES:
        raw = (torch.arange(numel, device=device) % 256).to(torch.uint8)
        return raw.reshape(shape).view(dtype)
    if dtype.is_complex:
        real = torch.randn(shape, device=device, dtype=torch.float32)
        imag = torch.randn(shape, device=device, dtype=torch.float32)
        return torch.complex(real, imag).to(dtype)
    if dtype.is_floating_point:
        return torch.randn(shape, device=device, dtype=torch.float32).to(dtype)
    iinfo = torch.iinfo(dtype)
    lo = max(iinfo.min, -100)
    hi = min(iinfo.max, 100)
    return torch.randint(lo, hi, shape, device=device, dtype=dtype)


def _aten_slice_copy(inp, dim=0, start=None, end=None, step=1):
    return torch.ops.aten.slice_copy.Tensor(inp, dim, start, end, step)


def _aten_slice_copy_out(inp, dim, start, end, step, out):
    return torch.ops.aten.slice_copy.Tensor_out(inp, dim, start, end, step, out=out)


def _assert_bitwise_equal(res, ref):
    """Compare exactly, reinterpreting float8 as uint8 (torch has no fp8 sub)."""
    assert res.shape == ref.shape, f"{res.shape} != {ref.shape}"
    if res.dtype in _FLOAT8_DTYPES:
        res, ref = res.view(torch.uint8), ref.view(torch.uint8)
    utils.gems_assert_equal(res, ref)


@pytest.mark.slice_copy
@pytest.mark.parametrize("shape", SLICE_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_basic(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        dim_size = inp.size(dim)
        for start in [0, dim_size // 4, dim_size // 2]:
            for end in [dim_size // 2 + 1, dim_size]:
                for step in [1, 2, 3]:
                    if start >= end:
                        continue
                    ref_out = torch.slice_copy(ref_inp, dim, start, end, step)
                    res_out = flag_gems.slice_copy(inp, dim, start, end, step)
                    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_default_args(dtype):
    # When start/end are None and step defaults to 1, slice_copy reproduces the input.
    shape = (4, 8, 12)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        ref_out = torch.slice_copy(ref_inp, dim)
        res_out = flag_gems.slice_copy(inp, dim)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_negative_start_end(dtype):
    # 3-D shape so negative dim indices (-1, -2) resolve to distinct axes.
    shape = (8, 16, 32)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    cases = [
        (1, -8, None, 1),
        (1, -8, -1, 1),
        (2, -10, -2, 2),
        (0, -4, None, 1),
        # start/end below -dim_size clamp to 0 (measured on ATen).
        (0, -100, 5, 1),
        (1, 2, -100, 1),
        (2, -100, -50, 1),
    ]
    for dim, start, end, step in cases:
        ref_out = torch.slice_copy(ref_inp, dim, start, end, step)
        res_out = flag_gems.slice_copy(inp, dim, start, end, step)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_step(dtype):
    # 2-D shape: vary step along both axes to exercise the strided kernel path.
    shape = (32, 64)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        dim_size = inp.size(dim)
        for start in [0, 1, dim_size // 4]:
            for step in [2, 3, 5, 7]:
                ref_out = torch.slice_copy(ref_inp, dim, start, dim_size, step)
                res_out = flag_gems.slice_copy(inp, dim, start, dim_size, step)
                utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_edge_cases(dtype):
    device = flag_gems.device

    # Empty slice: start >= end after clamping.
    inp = torch.randn((4, 8), dtype=dtype, device=device)
    ref_inp = utils.to_reference(inp)
    ref_out = torch.slice_copy(ref_inp, 1, 5, 3, 1)
    res_out = flag_gems.slice_copy(inp, 1, 5, 3, 1)
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)

    # Out-of-bounds start and end get clamped.
    ref_out = torch.slice_copy(ref_inp, 0, 100, 200, 1)
    res_out = flag_gems.slice_copy(inp, 0, 100, 200, 1)
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)

    # End clamped to the dimension size.
    ref_out = torch.slice_copy(ref_inp, 1, 2, 1000, 1)
    res_out = flag_gems.slice_copy(inp, 1, 2, 1000, 1)
    utils.gems_assert_equal(res_out, ref_out)

    # start == end produces an empty slice.
    ref_out = torch.slice_copy(ref_inp, 0, 4, 4, 1)
    res_out = flag_gems.slice_copy(inp, 0, 4, 4, 1)
    assert res_out.numel() == 0
    utils.gems_assert_equal(res_out, ref_out)

    # 1D tensor.
    inp1d = torch.randn(64, dtype=dtype, device=device)
    ref_inp1d = utils.to_reference(inp1d)
    ref_out = torch.slice_copy(ref_inp1d, 0, 10, 50, 2)
    res_out = flag_gems.slice_copy(inp1d, 0, 10, 50, 2)
    utils.gems_assert_equal(res_out, ref_out)

    # Zero-sized dimensions: shape propagates, slicing a zero-size dim stays zero.
    inp0 = torch.randn((4, 0, 3), dtype=dtype, device=device)
    for dim, start, end, step in [
        (1, 0, 0, 1),
        (1, 1, 3, 1),
        (0, 1, 3, 1),
        (2, 0, 2, 1),
    ]:
        ref_out = _aten_slice_copy(utils.to_reference(inp0), dim, start, end, step)
        res_out = flag_gems.slice_copy(inp0, dim, start, end, step)
        assert tuple(res_out.shape) == tuple(ref_out.shape)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_negative_dim(dtype):
    # 3-D shape so negative dims -1/-2/-3 map to distinct axes.
    shape = (4, 8, 12)
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in [-1, -2, -3]:
        ref_out = torch.slice_copy(ref_inp, dim, 1, 5, 2)
        res_out = flag_gems.slice_copy(inp, dim, 1, 5, 2)
        utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_slice_copy_int_dtype(dtype):
    inp = torch.randint(-100, 100, (4, 16), dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in [0, 1]:
        for start in [0, 2, -8]:
            for end in [8, None]:
                for step in [1, 2]:
                    ref_out = torch.slice_copy(ref_inp, dim, start, end, step)
                    res_out = flag_gems.slice_copy(inp, dim, start, end, step)
                    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", _ALL_NON_COMPLEX_DTYPES)
def test_slice_copy_all_dtypes(dtype):
    # Compare against aten::slice_copy.Tensor for every natively supported
    # non-complex dtype, including fp64, bool and the 1-byte integer types.
    inp = _make_input((5, 9), dtype)
    ref_inp = utils.to_reference(inp)

    for dim, start, end, step in [
        (0, 1, 5, 2),
        (1, 2, 8, 3),
        (1, 0, 9, 1),
        (0, -3, None, 1),
    ]:
        ref_out = _aten_slice_copy(ref_inp, dim, start, end, step)
        res_out = flag_gems.slice_copy(inp, dim, start, end, step)
        _assert_bitwise_equal(res_out, ref_out)
        assert res_out.is_contiguous()
        assert res_out.dtype == dtype


@pytest.mark.slice_copy
@pytest.mark.skipif(not _FLOAT8_DTYPES, reason="PyTorch build has no float8 dtypes")
@pytest.mark.parametrize("dtype", _FLOAT8_DTYPES)
def test_slice_copy_float8_bitwise(dtype):
    # Triton has no fp8 scalar type; the kernel copies through uint8. Sweep all
    # 256 byte patterns (incl. NaN encodings) and require a bit-exact copy.
    inp = _make_input((256,), dtype)
    ref_out = _aten_slice_copy(inp, 0, 0, 256, 3)
    res_out = flag_gems.slice_copy(inp, 0, 0, 256, 3)
    assert bool((res_out.view(torch.uint8) == ref_out.view(torch.uint8)).all())


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", _COMPLEX_DTYPES)
def test_slice_copy_complex(dtype):
    # Triton exposes no complex scalar type, so the kernel copies complex data
    # through view_as_real; check against ATen on every kernel path.
    inp = _make_input((4, 6, 5), dtype)

    cases = [
        (0, 1, 4, 1),  # step1 path, inner > 1
        (1, 1, 6, 2),  # general path, inner > 1
        (2, 0, 5, 2),  # inner1 path
        (2, 1, None, 1),
    ]
    for dim, start, end, step in cases:
        ref_out = _aten_slice_copy(inp, dim, start, end, step)
        res_out = flag_gems.slice_copy(inp, dim, start, end, step)
        assert res_out.dtype == dtype
        assert bool((res_out == ref_out).all())


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_slice_copy_lazy_conj_neg(dtype):
    # A conj/neg view carries the flag lazily: the raw bytes are un-negated, so
    # a plain bit copy would return the wrong values. ATen resolves them and
    # returns a tensor with the flag cleared.
    inp = _make_input((4, 6), dtype)
    lazy = inp.conj() if dtype.is_complex else torch.ops.aten._neg_view(inp)
    ref_out = _aten_slice_copy(lazy, 1, 1, 5, 2)
    res_out = flag_gems.slice_copy(lazy, 1, 1, 5, 2)
    assert bool((res_out == ref_out).all())
    assert res_out.is_conj() == ref_out.is_conj()
    assert res_out.is_neg() == ref_out.is_neg()


@pytest.mark.slice_copy
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_non_contiguous_input(dtype):
    # Transposed / strided / expanded inputs must be gathered by value, not by
    # flat offset.
    base = torch.randn((6, 8, 4), dtype=dtype, device=flag_gems.device)
    # Rebuild the same views on the reference tensor rather than moving each view,
    # so the stride patterns match on both sides in quick-cpu mode (--ref=cpu).
    ref_base = utils.to_reference(base)

    def _views(t):
        return [
            t.transpose(0, 1),
            t.permute(2, 0, 1),
            t[:, ::2, :],
            t.flip(0),
            t[1:5, 2:7, :],
        ]

    for view, ref_view in zip(_views(base), _views(ref_base)):
        for dim in range(view.ndim):
            ref_out = _aten_slice_copy(ref_view, dim, 0, ref_view.size(dim), 2)
            res_out = flag_gems.slice_copy(view, dim, 0, view.size(dim), 2)
            assert res_out.is_contiguous()
            utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.slice_copy
def test_slice_copy_invalid_args():
    # Error type, text and precedence all measured on aten::slice_copy.
    inp = torch.randn((4, 6), device=flag_gems.device)

    with pytest.raises(RuntimeError, match="slice step must be positive"):
        flag_gems.slice_copy(inp, 1, 0, 4, 0)
    with pytest.raises(RuntimeError, match="slice step must be positive"):
        flag_gems.slice_copy(inp, 1, 0, 4, -1)

    for dim in [2, 9, -3, -100]:
        with pytest.raises(IndexError, match=r"Dimension out of range"):
            flag_gems.slice_copy(inp, dim, 0, 4, 1)

    scalar = torch.tensor(3.0, device=flag_gems.device)
    with pytest.raises(IndexError, match="0-dim tensor"):
        flag_gems.slice_copy(scalar, 0, 0, 1, 1)

    # A bad dim is reported before a bad step, matching ATen.
    with pytest.raises(IndexError, match=r"Dimension out of range"):
        flag_gems.slice_copy(inp, 9, 0, 4, 0)


@pytest.mark.slice_copy_out
@pytest.mark.parametrize("shape", SLICE_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_slice_copy_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(inp.ndim):
        dim_size = inp.size(dim)
        start, end, step = dim_size // 4, dim_size - dim_size // 4, 2
        ref_out = torch.slice_copy(ref_inp, dim, start, end, step)

        out_shape = list(inp.shape)
        out_shape[dim] = (end - start + step - 1) // step
        res_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        res_r = flag_gems.slice_copy_out(inp, dim, start, end, step, out=res_out)

        assert res_r is res_out
        utils.gems_assert_equal(res_r, ref_out)


@pytest.mark.slice_copy_out
@pytest.mark.parametrize("dtype", _ALL_NON_COMPLEX_DTYPES + _COMPLEX_DTYPES)
def test_slice_copy_out_vs_aten_all_dtypes(dtype):
    # Compare slice_copy.Tensor_out against ATen directly for every dtype. The
    # reference runs on the reference device (CPU under --ref=cpu), so allocate
    # its out= there too.
    inp = _make_input((5, 9), dtype)
    ref_inp = utils.to_reference(inp)
    for dim, start, end, step in [(0, 1, 5, 2), (1, 2, 8, 3), (1, 0, 9, 1)]:
        target = list(inp.shape)
        target[dim] = (end - start + step - 1) // step
        ref_out = torch.empty(target, dtype=dtype, device=ref_inp.device)
        res_out = torch.empty(target, dtype=dtype, device=flag_gems.device)
        _aten_slice_copy_out(ref_inp, dim, start, end, step, ref_out)
        r = flag_gems.slice_copy_out(inp, dim, start, end, step, out=res_out)
        assert r is res_out
        _assert_bitwise_equal(res_out, ref_out)


@pytest.mark.slice_copy_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_slice_copy_out_mismatched_size(dtype):
    # ATen resizes a mismatched non-empty out (with a deprecation warning) rather
    # than raising; the result must land in the resized tensor.
    inp = _make_input((4, 6), dtype)
    ref_slice = _aten_slice_copy(inp, 1, 1, 5, 2)

    for out_shape in [(4, 1), (4, 5), (8, 2), (2,), (4, 2, 1), (0,)]:
        out = torch.zeros(out_shape, dtype=dtype, device=flag_gems.device)
        r = flag_gems.slice_copy_out(inp, 1, 1, 5, 2, out=out)
        assert r is out
        assert tuple(out.shape) == tuple(ref_slice.shape)
        assert bool((out == ref_slice).all())


@pytest.mark.slice_copy_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_slice_copy_out_non_contiguous(dtype):
    # The kernels store at flat offsets, which is only valid for a contiguous
    # out. A transposed / strided / permuted out must still be filled correctly
    # and must not touch storage outside the view.
    inp = _make_input((4, 6), dtype)
    ref_slice = _aten_slice_copy(inp, 1, 1, 5, 2)  # (4, 2)

    # Transposed out.
    buf = torch.zeros((2, 4), dtype=dtype, device=flag_gems.device)
    out = buf.t()
    assert not out.is_contiguous()
    r = flag_gems.slice_copy_out(inp, 1, 1, 5, 2, out=out)
    assert r is out
    assert out.stride() == (1, 4)
    assert bool((out == ref_slice).all())

    # Strided view into a larger buffer: the untouched elements must stay zero.
    big = torch.zeros((4, 10), dtype=dtype, device=flag_gems.device)
    out = big[:, 2:6:2]
    flag_gems.slice_copy_out(inp, 1, 1, 5, 2, out=out)
    assert bool((out == ref_slice).all())
    ref_big = torch.zeros((4, 10), dtype=dtype, device=flag_gems.device)
    _aten_slice_copy_out(inp, 1, 1, 5, 2, ref_big[:, 2:6:2])
    assert bool((big == ref_big).all())

    # Reverse-strided out.
    buf = torch.zeros((4, 2), dtype=dtype, device=flag_gems.device)
    out = buf.flip(0)
    flag_gems.slice_copy_out(inp, 1, 1, 5, 2, out=out)
    assert bool((out == ref_slice).all())

    # 3-D permuted out.
    inp3 = _make_input((4, 6, 3), dtype)
    ref3 = _aten_slice_copy(inp3, 1, 1, 5, 2)  # (4, 2, 3)
    buf3 = torch.zeros((3, 4, 2), dtype=dtype, device=flag_gems.device)
    out3 = buf3.permute(1, 2, 0)
    flag_gems.slice_copy_out(inp3, 1, 1, 5, 2, out=out3)
    assert bool((out3 == ref3).all())


@pytest.mark.slice_copy_out
def test_slice_copy_out_dtype_mismatch():
    inp = torch.randn((4, 6), device=flag_gems.device)
    for out_dtype in [torch.float16, torch.float64, torch.int32]:
        out = torch.zeros((4, 2), dtype=out_dtype, device=flag_gems.device)
        ref_msg = None
        try:
            _aten_slice_copy_out(inp, 1, 1, 5, 2, out.clone())
        except RuntimeError as exc:
            ref_msg = str(exc)
        assert ref_msg is not None
        with pytest.raises(RuntimeError) as excinfo:
            flag_gems.slice_copy_out(inp, 1, 1, 5, 2, out=out)
        assert str(excinfo.value) == ref_msg


@pytest.mark.slice_copy_out
@pytest.mark.skipif(
    flag_gems.device == "cpu", reason="device-mismatch test needs a non-CPU device"
)
def test_slice_copy_out_device_mismatch():
    inp = torch.randn((4, 6), device=flag_gems.device)
    out = torch.zeros((4, 2), dtype=torch.float32, device="cpu")
    ref_msg = None
    try:
        _aten_slice_copy_out(inp, 1, 1, 5, 2, out.clone())
    except RuntimeError as exc:
        ref_msg = str(exc)
    assert ref_msg is not None
    with pytest.raises(RuntimeError) as excinfo:
        flag_gems.slice_copy_out(inp, 1, 1, 5, 2, out=out)
    assert str(excinfo.value) == ref_msg


@pytest.mark.slice_copy_out
@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
def test_slice_copy_out_overlapping(dtype):
    # out aliasing the input storage: slice_copy reads the whole slice before
    # writing, so a forward-overlapping out must see the original values.
    for out_lo, dim, start, end, step in [
        (1, 1, 0, 2, 1),  # out shifted forward by one column
        (1, 1, 1, 5, 2),  # out is exactly the sliced region
        (0, 1, 2, 6, 2),  # out shifted backwards
    ]:
        base = torch.arange(24, device=flag_gems.device, dtype=torch.float32).reshape(
            4, 6
        )
        if dtype.is_complex:
            base = torch.complex(base, base * 0.5).to(dtype)
        slice_len = (end - start + step - 1) // step
        ref_t = base.clone()
        res_t = base.clone()
        _aten_slice_copy_out(
            ref_t, dim, start, end, step, ref_t[:, out_lo : out_lo + slice_len]
        )
        flag_gems.slice_copy_out(
            res_t, dim, start, end, step, out=res_t[:, out_lo : out_lo + slice_len]
        )
        assert bool((res_t == ref_t).all()), f"out_lo={out_lo} {start}:{end}:{step}"


@pytest.mark.slice_copy_out
def test_slice_copy_out_internal_overlap():
    # An expanded out addresses the same element twice; ATen refuses it.
    inp = torch.randn((4, 6), device=flag_gems.device)
    out = torch.zeros(2, device=flag_gems.device).expand(4, 2)
    ref_msg = None
    try:
        _aten_slice_copy_out(inp, 1, 1, 5, 2, out)
    except RuntimeError as exc:
        ref_msg = str(exc)
    assert ref_msg is not None and "single memory location" in ref_msg
    with pytest.raises(RuntimeError, match="single memory location"):
        flag_gems.slice_copy_out(inp, 1, 1, 5, 2, out=out)


@pytest.mark.slice_copy_out
def test_slice_copy_out_non_contiguous_input():
    base = torch.randn((6, 8, 4), device=flag_gems.device)
    inp = base.transpose(0, 1)
    # Transpose the reference on its own device so both sides share the layout.
    ref_out = _aten_slice_copy(utils.to_reference(base).transpose(0, 1), 1, 1, 6, 2)
    buf = torch.zeros(tuple(reversed(ref_out.shape)), device=flag_gems.device)
    out = buf.permute(2, 1, 0)
    flag_gems.slice_copy_out(inp, 1, 1, 6, 2, out=out)
    utils.gems_assert_equal(out, ref_out)


@pytest.mark.slice_copy
@pytest.mark.skipif(cfg.QUICK_MODE, reason="allocates ~2 GiB")
@pytest.mark.skipif(
    flag_gems.device == "cpu", reason="int64-index path is a device kernel concern"
)
def test_slice_copy_int64_index():
    # Flat offsets above 2**31 - 1 must not be computed in int32. Two cases:
    # a small slice out of a huge input, and an output larger than int32 max.
    n = 2**31 + 1024
    inp = torch.ones(n, dtype=torch.int8, device=flag_gems.device)
    tail = torch.tensor(
        [1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int8, device=flag_gems.device
    )
    inp[-8:] = tail
    try:
        res_out = flag_gems.slice_copy(inp, 0, n - 8, n, 1)
        assert res_out.tolist() == tail.tolist()

        res_out = flag_gems.slice_copy(inp, 0, 0, n, 1)
        assert res_out.numel() == n
        # every element copied: the tail sums to 36, the rest are ones
        expected = (n - 8) + 36
        assert res_out.to(torch.int64).sum().item() == expected
    finally:
        del inp
        torch.cuda.empty_cache() if flag_gems.device != "cpu" else None
