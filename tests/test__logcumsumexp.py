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

import math

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

LOGCUMSUMEXP_SHAPES = (
    [(2, 32)]
    if utils.QUICK_MODE
    else utils.REDUCTION_SHAPES + [(2637,), (16, 1025, 255)]
)

# Shapes exercising the scalar/zero-sized paths across dimension positions.
EMPTY_SHAPES = [(0,), (2, 0), (0, 3), (3, 0), (2, 0, 4), (2, 4, 0)]


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("shape", LOGCUMSUMEXP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__logcumsumexp(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)

    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch._logcumsumexp(ref_inp, dim)
    res_out = flag_gems._logcumsumexp(inp, dim)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


@pytest.mark.underscore_logcumsumexp_out
@pytest.mark.parametrize("shape", LOGCUMSUMEXP_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__logcumsumexp_out(shape, dtype):
    dim = 1 if shape == utils.REDUCTION_SHAPES[-1] else -1
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out_buf = torch.empty_like(ref_inp)
    ref_out = torch.ops.aten._logcumsumexp.out(ref_inp, dim, out=ref_out_buf)

    res_out_buf = torch.empty_like(inp)
    res_out = flag_gems._logcumsumexp_out(inp, dim, out=res_out_buf)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=shape[dim])


# ---------------------------------------------------------------------------
# Numerical stability: large dynamic range, NaN and +-Inf (review comment 1).
# ---------------------------------------------------------------------------


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "data",
    [
        [0.0, 1000.0],
        [-1000.0, 0.0],
        [0.0, 1000.0, 2000.0],
        [-1000.0, -2000.0, 5.0],
    ],
)
def test__logcumsumexp_large_dynamic_range(data):
    """A prefix far below the row maximum must not underflow to -inf."""
    inp = torch.tensor(data, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, 0)
    res_out = flag_gems._logcumsumexp(inp, 0)

    utils.gems_assert_equal(res_out, ref_out)
    # Explicit guard against the previous whole-row-maximum shift, which
    # produced -inf for every prefix preceding a large value.
    assert not torch.isinf(res_out).any()


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "data",
    [
        [float("-inf"), float("inf"), 1.0],
        [float("inf"), float("-inf")],
        [float("-inf"), 5.0, float("-inf")],
        [float("-inf"), float("inf")],
    ],
)
def test__logcumsumexp_mixed_infinities(data):
    """+-Inf rows match ATen exactly and never collapse to NaN."""
    inp = torch.tensor(data, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, 0)
    res_out = flag_gems._logcumsumexp(inp, 0)

    utils.gems_assert_equal(res_out, ref_out)
    assert not torch.isnan(res_out).any()


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "data",
    [
        [float("nan"), 1.0, 3.0],
        [1.0, float("nan"), 3.0],
        [1.0, 3.0, float("nan")],
        [float("-inf"), float("nan")],
        [float("inf"), float("nan")],
    ],
)
def test__logcumsumexp_nan_propagation(data):
    """NaN is infectious: once seen it stays for every later prefix."""
    inp = torch.tensor(data, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, 0)
    res_out = flag_gems._logcumsumexp(inp, 0)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)
    first_nan = next(i for i, v in enumerate(data) if v != v)
    assert torch.isnan(res_out[first_nan:]).all()


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "data,expected",
    [
        ([float("-inf")] * 4, [float("-inf")] * 4),
        ([float("inf")] * 4, [float("inf")] * 4),
        (
            [float("-inf"), float("inf"), float("-inf")],
            [float("-inf"), float("inf"), float("inf")],
        ),
    ],
)
def test__logcumsumexp_infinite_inputs(data, expected):
    """All -inf / all +inf rows must not turn into NaN."""
    inp = torch.tensor(data, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, 0)
    res_out = flag_gems._logcumsumexp(inp, 0)

    utils.gems_assert_equal(res_out, ref_out)
    assert res_out.tolist() == expected


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test__logcumsumexp_wide_dynamic_range_rows(dtype):
    """Long rows spanning ~e^60 must stay accurate at every prefix."""
    if dtype == torch.float16 and not utils.bf16_is_supported:
        pytest.skip("fp16 not supported on this device")
    rows, cols = 8, 1024
    scale = torch.exp(torch.linspace(0, 60, cols, device=flag_gems.device))
    inp = (torch.randn(rows, cols, dtype=dtype, device=flag_gems.device) * scale).to(
        dtype
    )
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, 1)
    res_out = flag_gems._logcumsumexp(inp, 1)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=cols)
    assert not torch.isnan(res_out).any()


# ---------------------------------------------------------------------------
# FP64 and complex parity (review comment 2).
# ---------------------------------------------------------------------------


@pytest.mark.underscore_logcumsumexp
@pytest.mark.skipif(
    not utils.fp64_is_supported, reason="float64 not supported on this device"
)
@pytest.mark.parametrize("shape,dim", [((4, 1024), 1), ((512,), 0), ((8, 64, 32), 1)])
def test__logcumsumexp_fp64_precision(shape, dim):
    """FP64 accumulation must keep FP64 accuracy (no silent fp32 cast)."""
    dtype = torch.float64
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device) * 3
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
    res_out = flag_gems._logcumsumexp(inp, dim)

    assert res_out.dtype == torch.float64
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)
    # RESOLUTION[float64] is 1e-7; a fp32 accumulation loses ~1e-6 and fails.
    # Both sides are normalized to the reference device first: under --ref=cpu
    # the reference lives on the host while the result stays on the device.
    max_err = (
        (res_out.double() - ref_out.double().to(res_out.device)).abs().max().item()
    )
    assert max_err < 1e-9, f"fp32 accumulation leaked into fp64 path: {max_err}"


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize(
    "shape,dim", [((3, 4), 1), ((3, 4), 0), ((2, 3, 4), 2), ((64,), 0)]
)
def test__logcumsumexp_complex(shape, dim, dtype):
    dtype_is_fp64 = dtype == torch.complex128
    if dtype_is_fp64 and not utils.fp64_is_supported:
        pytest.skip("complex128 not supported on this device")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
    res_out = flag_gems._logcumsumexp(inp, dim)

    assert res_out.dtype == dtype
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "data",
    [
        [complex(float("-inf"), 0), complex(0, 0), complex(1, 2)],
        [complex(float("inf"), 0), complex(1, 0), complex(0, 0)],
        [complex(float("inf"), 0), complex(float("inf"), 0)],
        [complex(float("-inf"), 0), complex(float("-inf"), 0)],
        [complex(float("nan"), 0), complex(1, 0), complex(0, 0)],
        [complex(1, 0), complex(float("nan"), 0)],
    ],
)
def test__logcumsumexp_complex_special_values(data):
    """Parity with ATen's complex NaN/+-Inf branches.

    Finite prefixes only need fp32 rounding agreement, but the NaN/+-Inf
    pattern is asserted exactly so a wrong branch cannot hide behind the
    tolerance.
    """
    inp = torch.tensor(data, dtype=torch.complex64, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, 0)
    res_out = flag_gems._logcumsumexp(inp, 0)

    utils.gems_assert_close(res_out, ref_out, torch.complex64, equal_nan=True)
    # Compare the branch structure on one device: under --ref=cpu the reference
    # is materialized on the host, and Tensor.equal requires matching devices.
    ref_on_dev = ref_out.to(res_out.device)
    assert torch.isnan(res_out).equal(torch.isnan(ref_on_dev))
    assert torch.isinf(res_out.real).equal(torch.isinf(ref_on_dev.real))
    assert torch.isinf(res_out.imag).equal(torch.isinf(ref_on_dev.imag))


# ---------------------------------------------------------------------------
# Scalar and zero-sized inputs in every dimension position (comment 3).
# ---------------------------------------------------------------------------


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("dim", [0, -1])
def test__logcumsumexp_scalar(dim, dtype):
    inp = torch.tensor(3.5, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
    res_out = flag_gems._logcumsumexp(inp, dim)

    assert res_out.shape == ref_out.shape == torch.Size([])
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=1)


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("dim", [1, -2])
def test__logcumsumexp_scalar_invalid_dim(dim):
    """A 0-dim tensor behaves as 1-dim, so dim=1 / dim=-2 are out of range."""
    inp = torch.tensor(3.5, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(IndexError):
        torch.ops.aten._logcumsumexp(ref_inp, dim)
    with pytest.raises(IndexError):
        flag_gems._logcumsumexp(inp, dim)


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("dim", [2, -3, 5, -5])
def test__logcumsumexp_invalid_dim(dim):
    """Out-of-range dims raise before any kernel launch."""
    inp = torch.randn(3, 4, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    with pytest.raises(IndexError):
        torch.ops.aten._logcumsumexp(ref_inp, dim)
    with pytest.raises(IndexError):
        flag_gems._logcumsumexp(inp, dim)


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("shape", EMPTY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__logcumsumexp_empty(shape, dtype):
    """Zero-sized inputs must return an empty tensor, not divide by zero."""
    inp = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    for dim in range(len(shape)):
        ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
        res_out = flag_gems._logcumsumexp(inp, dim)
        assert res_out.shape == ref_out.shape == torch.Size(shape)


@pytest.mark.underscore_logcumsumexp_out
@pytest.mark.parametrize("shape", [(0,), (2, 0), (0, 3), (3, 0)])
def test__logcumsumexp_out_empty(shape):
    inp = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
    out = torch.empty(shape, dtype=torch.float32, device=flag_gems.device)

    for dim in range(len(shape)):
        res_out = flag_gems._logcumsumexp_out(inp, dim, out=out)
        assert res_out is out
        assert res_out.shape == torch.Size(shape)


# ---------------------------------------------------------------------------
# The out= contract (review comment 4).
# ---------------------------------------------------------------------------


@pytest.mark.underscore_logcumsumexp_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__logcumsumexp_out_resizes(dtype):
    """A wrongly sized out buffer is resized, not written out of bounds."""
    inp = torch.randn(3, 4, dtype=dtype, device=flag_gems.device)
    out = torch.empty(17, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out_buf = torch.empty(17, dtype=ref_inp.dtype, device=ref_inp.device)
    ref_out = torch.ops.aten._logcumsumexp.out(ref_inp, 1, out=ref_out_buf)
    res_out = flag_gems._logcumsumexp_out(inp, 1, out=out)

    assert res_out is out
    assert out.shape == inp.shape
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=4)


@pytest.mark.underscore_logcumsumexp_out
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test__logcumsumexp_out_noncontiguous(dtype):
    """A non-contiguous out keeps its layout and its buffer identity."""
    inp = torch.randn(3, 4, dtype=dtype, device=flag_gems.device)
    out = torch.empty(4, 3, dtype=dtype, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp, True)

    ref_out_buf = torch.empty(4, 3, dtype=ref_inp.dtype, device=ref_inp.device).t()
    ref_out = torch.ops.aten._logcumsumexp.out(ref_inp, 1, out=ref_out_buf)
    res_out = flag_gems._logcumsumexp_out(inp, 1, out=out)

    assert res_out is out
    assert res_out.stride() == out.stride()
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=4)


@pytest.mark.underscore_logcumsumexp_out
def test__logcumsumexp_out_rejects_mismatched_dtype():
    inp = torch.randn(3, 4, device=flag_gems.device)
    out = torch.empty(3, 4, dtype=torch.float64, device=flag_gems.device)

    # The reference must stay on the input's device: ATen's CUDA kernel rejects
    # a mismatched dtype, while the CPU kernel silently accepts it (measured),
    # so normalizing the reference to CPU would assert the wrong contract.
    with pytest.raises(RuntimeError):
        torch.ops.aten._logcumsumexp.out(inp, 1, out=out)
    with pytest.raises(RuntimeError):
        flag_gems._logcumsumexp_out(inp, 1, out=out)


@pytest.mark.underscore_logcumsumexp_out
def test__logcumsumexp_out_rejects_mismatched_device():
    inp = torch.randn(3, 4, device=flag_gems.device)
    out = torch.empty(3, 4, dtype=inp.dtype, device="cpu")

    with pytest.raises(RuntimeError):
        flag_gems._logcumsumexp_out(inp, 1, out=out)


@pytest.mark.underscore_logcumsumexp_out
def test__logcumsumexp_out_invalid_dim_does_not_touch_out():
    """A bad dim must raise before the output buffer is written."""
    inp = torch.randn(3, 4, device=flag_gems.device)
    out = torch.full((3, 4), 5.0, device=flag_gems.device)

    with pytest.raises(IndexError):
        flag_gems._logcumsumexp_out(inp, 99, out=out)
    assert (out == 5.0).all()


@pytest.mark.underscore_logcumsumexp
def test__logcumsumexp_noncontiguous_input():
    """A transposed input is handled like ATen's contiguous result."""
    inp = torch.randn(4, 3, device=flag_gems.device).t()
    ref_inp = utils.to_reference(inp, True)

    for dim in (0, 1):
        ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
        res_out = flag_gems._logcumsumexp(inp, dim)
        assert res_out.shape == ref_out.shape
        utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=4)


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
def test__logcumsumexp_all_dims(dim):
    """Every dimension position must agree with ATen."""
    inp = torch.randn(4, 8, 16, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
    res_out = flag_gems._logcumsumexp(inp, dim)

    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=inp.shape[dim])


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((1, 8192), 1),  # just past the single-block limit
        ((1, 16384), 1),  # several chunks
        ((2, 4097, 3), 1),  # K > 1 and a tail chunk
        ((3, 12000), 1),
    ],
    ids=["2x-chunk", "long", "k-gt-1", "tail-chunk"],
)
def test__logcumsumexp_long_rows(shape, dim):
    """Rows past the single-block limit must scan in chunks and still match ATen.

    A whole-row block is what makes ``tl.associative_scan`` expensive to compile
    as N grows, so long rows are split into fixed-size chunks with a two-pass
    fan-in. The result must be identical to ATen's (within the usual fp32
    tolerance) for every row length, chunk boundary and K.
    """
    inp = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
    res_out = flag_gems._logcumsumexp(inp, dim)

    assert res_out.shape == ref_out.shape
    assert not torch.isnan(res_out).any()
    utils.gems_assert_close(res_out, ref_out, torch.float32, reduce_dim=shape[dim])


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "data",
    [
        [float("-inf"), float("inf"), 0.0],
        [0.0, 1000.0],
        [float("inf"), float("inf"), 1.0],
        [float("-inf"), float("-inf"), 1.0],
        [float("nan"), 1.0, 2.0],
        [1.0, float("nan"), 2.0],
    ],
)
def test__logcumsumexp_long_rows_special_values(data):
    """NaN/+-Inf state must survive the chunk boundary.

    The chunk total is what propagates a NaN (or the all-+-inf branch) into the
    following chunks, so it is taken as the inclusive scan's last lane. A
    ``tl.max`` reduction would drop the NaN there and silently un-poison the
    elements after a chunk boundary.
    """
    n = 8192
    x = torch.tensor(
        (data * (n // len(data) + 1))[:n],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_x = utils.to_reference(x)

    ref_out = torch.ops.aten._logcumsumexp(ref_x, 0)
    res_out = flag_gems._logcumsumexp(x, 0)
    ref_on_dev = ref_out.to(res_out.device)

    # The branch structure is compared exactly; the finite prefixes only need
    # fp32 rounding agreement (and the CPU/CUDA backends round differently, so
    # the values are compared against the same-device reference only).
    assert torch.isnan(res_out).equal(torch.isnan(ref_on_dev))
    assert torch.isinf(res_out).equal(torch.isinf(ref_on_dev))
    if not utils.TO_CPU:
        utils.gems_assert_close(
            res_out, ref_out, torch.float32, equal_nan=True, reduce_dim=1
        )


@pytest.mark.underscore_logcumsumexp
def test__logcumsumexp_long_row_fp64():
    """FP64 accumulation must stay FP64 on the chunked path."""
    if not utils.fp64_is_supported:
        pytest.skip("float64 not supported on this device")
    inp = torch.randn(1, 8192, dtype=torch.float64, device=flag_gems.device) * 3
    ref_inp = utils.to_reference(inp)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, 1)
    res_out = flag_gems._logcumsumexp(inp, 1)

    assert res_out.dtype == torch.float64
    max_err = (
        (res_out.double() - ref_out.double().to(res_out.device)).abs().max().item()
    )
    assert (
        max_err < 1e-9
    ), f"fp32 accumulation leaked into the chunked fp64 path: {max_err}"


# ---------------------------------------------------------------------------
# Long complex rows must take the chunked *complex* scan (review comment).
# ---------------------------------------------------------------------------


def _wrap_to_pi(delta):
    """Map an angular difference into (-pi, pi].

    A complex ``log`` returns its argument in (-pi, pi], so two implementations
    that agree on the value may still print arguments that differ by exactly
    ``2*pi`` when the true value sits on the branch cut (measured: gem returns
    ``0.6952714920043945-3.006592273712158j`` where ATen returns the same
    modulus with ``+3.2765932083129883j``). Comparing the wrapped difference
    checks the value rather than which side of the cut each rounding chose. This
    is a property of the ATen contract, not of the chunked path: the identical
    discrepancy reproduces on the untouched single-block complex kernel.
    """
    two_pi = 2.0 * math.pi
    return (delta + math.pi) % two_pi - math.pi


def _assert_complex_scan_close(res_out, ref_out):
    """Compare a complex scan to ATen, tolerating only the 2*pi branch cut.

    NaNs are compared as a mask (they must appear in the same places); only the
    entries where both sides are *finite* are checked for value agreement, so an
    ``inf - inf`` in the difference cannot turn into a spurious NaN.
    """
    assert torch.isnan(res_out).equal(torch.isnan(ref_out))
    both_finite = torch.isfinite(res_out) & torch.isfinite(ref_out)
    if bool(both_finite.any()):
        dr = (res_out.real - ref_out.real)[both_finite]
        di = _wrap_to_pi((res_out.imag - ref_out.imag)[both_finite])
        # Finite results must agree to fp32 rounding, not merely in branch
        # structure: 1e-3 still catches the ~9.6 / ~5.5e5 disagreements the real
        # chunk kernels produce while leaving the fp32 ulp noise an order of
        # magnitude of room.
        assert dr.abs().max().item() < 1e-3, "real part diverges from ATen"
        assert di.abs().max().item() < 1e-3, "imaginary part diverges beyond 2*pi"
    # Non-finite entries must agree in sign too, which the wrapped difference
    # above deliberately skips.
    assert torch.isinf(res_out.real).equal(torch.isinf(ref_out.real))
    assert torch.isinf(res_out.imag).equal(torch.isinf(ref_out.imag))
    assert (res_out.real[torch.isinf(res_out.real)] > 0).equal(
        ref_out.real[torch.isinf(ref_out.real)] > 0
    )
    assert (res_out.imag[torch.isinf(res_out.imag)] > 0).equal(
        ref_out.imag[torch.isinf(ref_out.imag)] > 0
    )


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "shape,dim",
    [
        ((5000,), 0),  # just past the single-block limit
        ((8192,), 0),  # exact chunk boundary
        ((1, 12288), 1),  # several chunks
        ((2, 5000, 3), 1),  # K > 1 and a tail chunk
    ],
    ids=["past-limit", "chunk-boundary", "several-chunks", "k-gt-1-tail"],
)
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test__logcumsumexp_long_complex_rows(shape, dim, dtype):
    """Complex rows past the single-block limit must scan as complex pairs.

    ``out`` is the interleaved ``(..., 2)`` real view on the chunked path, so
    reusing the *real* chunk kernels there scans each re/im part as an
    independent real number instead of combining complex pairs -- measured
    disagreement with ATen was up to 6.283 in the imaginary part (a full 2*pi
    branch wrap, since the interleaved parts are scanned as separate reals), and
    values like ``~5.5e5`` for complex128. The chunked path therefore needs its
    own complex kernels.
    """
    if dtype == torch.complex128 and not utils.fp64_is_supported:
        pytest.skip("complex128 not supported on this device")

    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)

    ref_out = torch.ops.aten._logcumsumexp(ref_inp, dim)
    res_out = flag_gems._logcumsumexp(inp, dim)

    assert res_out.shape == ref_out.shape
    _assert_complex_scan_close(res_out, ref_out.to(res_out.device))


@pytest.mark.underscore_logcumsumexp
@pytest.mark.parametrize(
    "data",
    [
        [complex(float("nan"), 0), complex(1, 0), complex(0, 0)],
        [complex(float("inf"), 0), complex(float("inf"), 0), complex(0, 0)],
        [complex(float("-inf"), 0), complex(float("-inf"), 0), complex(0, 0)],
        [complex(float("-inf"), 0), complex(float("inf"), 0), complex(1, 2)],
    ],
)
def test__logcumsumexp_long_complex_rows_special_values(data):
    """NaN/+-Inf state must survive the chunked complex path's chunk boundary.

    The chunk totals are taken as the inclusive scan's last lane on both the real
    and imaginary parts, so a NaN or a branch-selecting infinity has to carry
    across chunks exactly as it does on the single-block complex path.
    """
    n = 8192
    x = torch.tensor(
        (data * (n // len(data) + 1))[:n],
        dtype=torch.complex64,
        device=flag_gems.device,
    )
    ref_x = utils.to_reference(x)

    ref_out = torch.ops.aten._logcumsumexp(ref_x, 0)
    res_out = flag_gems._logcumsumexp(x, 0)
    ref_on_dev = ref_out.to(res_out.device)

    # The branch structure is compared exactly; finite prefixes only need fp32
    # rounding agreement.
    assert torch.isnan(res_out).equal(torch.isnan(ref_on_dev))
    assert torch.isinf(res_out.real).equal(torch.isinf(ref_on_dev.real))
    assert torch.isinf(res_out.imag).equal(torch.isinf(ref_on_dev.imag))
    if not utils.TO_CPU:
        _assert_complex_scan_close(res_out, ref_on_dev)
