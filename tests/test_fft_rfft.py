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
from . import test_utils as tu

# aten::fft_rfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None)
#   -> Tensor
# Turns a real signal of length n along dim (the trailing axis by default) into
# n // 2 + 1 complex bins. Measured with torch.ops.aten.fft_rfft on the active
# backend (cuda / nvidia):
#   * float32 -> complex64, float64 -> complex128, float16 -> complex32;
#     int8 / uint8 / int32 / int64 / bool are accepted and promoted through
#     float32 (complex64 result)
#   * float16 is a cuFFT capability (the CPU reference reports 'Unsupported
#     dtype Half') and cuFFT additionally restricts it to a power-of-two
#     *transformed* length: 'cuFFT only supports dimensions whose sizes are
#     powers of two when computing in half precision, but got a signal size
#     of[7]'. Non-transformed axes are not restricted - (2, 19, 8) with a
#     default dim and dim=0 on (4, 19) both transform fine. This implementation
#     exposes no half-precision capability flag (DeviceDetector reports only
#     fp64 / bf16 / int64 / fp8), so the half lane is scoped to the measured
#     NVIDIA/cuFFT backend through flag_gems.vendor_name and its restriction
#     cases are collected only there; other devices and other generations are
#     outside what was measured.
#   * bfloat16 / float8_e4m3fn / float8_e5m2 raise 'Unsupported dtype ...' on
#     this backend, and a complex tensor raises 'rfft expects a real input
#     tensor' (schema-wide: the real-signal contract rejects every complex
#     dtype)
#   * a 0-dim input raises IndexError ('Dimension specified as -1 but tensor has
#     no dimensions') on both the CPU and the CUDA reference: rfft always has to
#     reduce one axis, so the () entry of the spec shape grid is covered as an
#     invalid input instead of a value row
#   * an empty signal (shape (0,)) raises RuntimeError ('Invalid number of data
#     points (0) specified') on CPU and CUDA unless an explicit positive n pads
#     it, while a zero-length *batch* axis is rejected by the vendor FFT library
#     ('cuFFT error: CUFFT_INVALID_SIZE' on CUDA, 'MKL FFT error' on CPU); the
#     batch rows are therefore collected for the cuFFT backend only
# The operator has no second tensor operand, so no broadcast dimension exists.
_CUFFT_FFT_BACKEND = flag_gems.vendor_name == "nvidia"

_FLOAT_DTYPES = [torch.float32]
if _CUFFT_FFT_BACKEND:
    _FLOAT_DTYPES.append(torch.float16)
if utils.fp64_is_supported:
    _FLOAT_DTYPES.append(torch.float64)

_PROMOTED_DTYPES = [torch.int8, torch.uint8, torch.int32, torch.bool]
if utils.int64_is_supported:
    _PROMOTED_DTYPES.append(torch.int64)

_RFFT_DTYPES = _FLOAT_DTYPES + _PROMOTED_DTYPES

# Rejected inputs. A complex tensor violates the schema-wide real-signal
# contract (complex64 always, complex128 where the device can build it). The
# 'Unsupported dtype ...' rejections were measured on the cuFFT backend, so a row
# is only collected when that vendor scope applies *and* the device advertises the
# dtype at all - collecting bfloat16 or fp8 on a backend that reports the
# capability as absent would only assert input construction that cannot run.
_REJECTED_DTYPES = [torch.complex64]
if utils.fp64_is_supported:
    _REJECTED_DTYPES.append(torch.complex128)
if _CUFFT_FFT_BACKEND:
    if utils.bf16_is_supported:
        _REJECTED_DTYPES.append(torch.bfloat16)
    if utils.fp8_is_supported:
        _REJECTED_DTYPES += [torch.float8_e4m3fn, torch.float8_e5m2]


def _is_power_of_two(value):
    return value > 0 and (value & (value - 1)) == 0


def _output_dtype(dtype):
    # Measured output dtypes: float64 -> complex128, float16 -> complex32,
    # every other accepted real dtype -> complex64.
    return {
        torch.float64: torch.complex128,
        torch.float16: torch.complex32,
    }.get(dtype, torch.complex64)


def _numel(shape):
    total = 1
    for size in shape:
        total *= size
    return total


def _output_shape(shape, n, dim):
    # Metadata-only output geometry of one rfft call: only the transformed axis
    # changes size.
    axis = dim % len(shape)
    length = shape[axis] if n is None else n
    out = list(shape)
    out[axis] = length // 2 + 1
    return tuple(out)


_SPEC_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 1]

# The half lane needs a power-of-two transformed length, so it keeps the spec
# shapes whose trailing length already is one and uses adapted natural-length
# variants (that length rounded up to a power of two) for the rest; the original
# odd lengths stay covered by the explicit-n padding cases and by the
# non-power-of-two rejection test below. Quick mode keeps one adapted shape.
_HALF_SHAPES = tu.selected_cases(
    [shape for shape in _SPEC_SHAPES if _is_power_of_two(shape[-1])]
    + [(20, 320, 16), (16, 128, 64, 64), (16, 7, 57, 32, 32)],
    quick=[(2, 19, 8)],
)

_VALUE_CASES = [
    (dtype, shape, value_range)
    for dtype in _RFFT_DTYPES
    for shape in (_HALF_SHAPES if dtype is torch.float16 else _SPEC_SHAPES)
    for value_range in tu.selected_ranges()
]


@pytest.mark.fft_rfft
@pytest.mark.parametrize("dtype,shape,value_range", _VALUE_CASES)
def test_fft_rfft_value_ranges(dtype, shape, value_range):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp)
    res_out = flag_gems.fft_rfft(inp)

    tu.assert_result_close(res_out, ref_out)


# n, dim and norm are omitted, so the schema defaults (None, -1, None) have to
# come from the candidate's own public call signature. float64 / int64 are only
# rows when the backend advertises them.
_DEFAULT_ROWS = [(torch.float32, (20, 320, 15))]
if utils.int64_is_supported:
    _DEFAULT_ROWS.append((torch.int64, (1024, 1024)))
if utils.fp64_is_supported:
    _DEFAULT_ROWS.append((torch.float64, (256,)))

_DEFAULT_CASES = tu.selected_cases(_DEFAULT_ROWS, quick=[])


@pytest.mark.fft_rfft
@pytest.mark.parametrize("dtype,shape", _DEFAULT_CASES)
def test_fft_rfft_default_arguments(dtype, shape):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp)
    res_out = flag_gems.fft_rfft(inp)

    tu.assert_result_close(res_out, ref_out)


# n is an optional SymInt: 1 is the shortest valid transform, odd values
# truncate to n // 2 + 1 bins, larger values zero-pad, and an empty signal only
# becomes valid through an explicit positive n.
_N_CASES = tu.selected_cases(
    [
        ((1024, 1024), 1),
        ((1024, 1024), 3),
        ((1024, 1024), 17),
        ((1024, 1024), 2048),
        ((20, 320, 15), 16),
        ((20, 320, 15), 640),
        ((16, 128, 64, 60), 60),
        ((16, 7, 57, 32, 29), 5),
        ((0,), 8),
    ],
    quick=[],
)

# Parameter sweeps keep the supplement's prescribed shapes with a floating and
# a promoted integer dtype (quick mode uses the integer one).
_PARAM_DTYPES = tu.selected_cases([torch.float32, torch.int32], quick=[torch.int32])


@pytest.mark.fft_rfft
@pytest.mark.parametrize("shape,n", _N_CASES)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test_fft_rfft_signal_length(shape, n, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp, n, -1)
    res_out = flag_gems.fft_rfft(inp, n, -1)

    tu.assert_result_close(res_out, ref_out)


# dim selects the transformed axis: trailing, leading and negative indices.
_DIM_CASES = tu.selected_cases(
    [
        ((1024, 1024), 0),
        ((1024, 1024), 1),
        ((16, 128, 64, 60), 2),
        ((16, 128, 64, 60), -2),
        ((20, 320, 15), -3),
        ((16, 7, 57, 32, 29), 3),
    ],
    quick=[],
)


@pytest.mark.fft_rfft
@pytest.mark.parametrize("shape,dim", _DIM_CASES)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test_fft_rfft_transform_dim(shape, dim, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp, None, dim)
    res_out = flag_gems.fft_rfft(inp, None, dim)

    tu.assert_result_close(res_out, ref_out)


# norm spans the documented modes plus an explicit None; invalid values are
# negative-tested below.
_NORM_CASES = tu.selected_cases(
    [
        ((1024, 1024), None),
        ((1024, 1024), "backward"),
        ((1024, 1024), "forward"),
        ((20, 320, 15), "ortho"),
        ((16, 128, 64, 60), "ortho"),
        ((16, 7, 57, 32, 29), "forward"),
    ],
    quick=[],
)


@pytest.mark.fft_rfft
@pytest.mark.parametrize("shape,norm", _NORM_CASES)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test_fft_rfft_norm(shape, norm, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp, None, -1, norm)
    res_out = flag_gems.fft_rfft(inp, None, -1, norm)

    tu.assert_result_close(res_out, ref_out)


# Representative n / dim / norm interactions, including the explicit None forms
# of the two optional arguments that accept them.
_COMBINED_CASES = tu.selected_cases(
    [
        ((1024, 1024), None, -1, None),
        ((1024, 1024), 512, 0, "ortho"),
        ((20, 320, 15), None, 1, "forward"),
        ((16, 128, 64, 60), 64, 2, None),
        ((16, 7, 57, 32, 29), 29, -1, "ortho"),
    ],
    quick=[],
)


@pytest.mark.fft_rfft
@pytest.mark.parametrize("shape,n,dim,norm", _COMBINED_CASES)
@pytest.mark.parametrize("dtype", _PARAM_DTYPES)
def test_fft_rfft_combined_arguments(shape, n, dim, norm, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp, n, dim, norm)
    res_out = flag_gems.fft_rfft(inp, n, dim, norm)

    tu.assert_result_close(res_out, ref_out)


# aten::fft_rfft.out(self, n=None, dim=-1, norm=None, *, out) is callable on
# this backend (probed as torch.ops.aten.fft_rfft.out(inp, out=buf) and as
# torch.ops.aten.fft_rfft.out(inp, n, dim, norm, out=buf), both returning the
# caller's buffer). At most four positional arguments are accepted - passing
# out positionally as a fifth is rejected with 'aten::fft_rfft() takes 4
# positional argument(s) but 5 was/were given' - so out is always a keyword
# here. The buffer geometry comes from the call metadata; the offset / strided
# rows additionally check that the view layout survives the call and that the
# surrounding storage is left untouched.
_OUT_ROWS = [
    (torch.float32, (1024, 1024), None, -1, None, "fresh"),
    (torch.float32, (20, 320, 15), 16, -1, "ortho", "offset"),
    (torch.float32, (16, 128, 64, 60), 60, 2, "forward", "fresh"),
    (torch.float32, (1024, 1024), None, -1, None, "strided"),
    (torch.int32, (1024, 1024), None, -1, None, "fresh"),
]
if _CUFFT_FFT_BACKEND:
    _OUT_ROWS.append((torch.float16, (256,), None, -1, None, "fresh"))
if utils.fp64_is_supported:
    _OUT_ROWS.append((torch.float64, (256,), None, -1, None, "fresh"))

_OUT_CASES = tu.selected_cases(_OUT_ROWS, quick=[])


def _make_out_buffer(out_shape, dtype, geometry, device):
    """Build the caller's output view plus (region, snapshot) guard pairs.

    Each snapshot is taken before the call through tu.to_reference, so it has
    independent storage and follows the shared reference device policy while the
    shared exact comparison catches any write outside the requested output region.
    """
    if geometry == "offset":
        flat = torch.zeros(_numel(out_shape) + 16, dtype=dtype, device=device)
        view = flat[8 : 8 + _numel(out_shape)].view(out_shape)
        regions = [flat[:8], flat[8 + _numel(out_shape) :]]
    elif geometry == "strided":
        padded = list(out_shape)
        padded[-1] *= 2
        container = torch.zeros(padded, dtype=dtype, device=device)
        view = container[..., ::2]
        regions = [container[..., 1::2]]
    else:
        return torch.zeros(out_shape, dtype=dtype, device=device), []
    return view, [(region, tu.to_reference(region)) for region in regions]


@pytest.mark.fft_rfft
@pytest.mark.parametrize("dtype,shape,n,dim,norm,geometry", _OUT_CASES)
def test_fft_rfft_out(dtype, shape, n, dim, norm, geometry):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    out_shape = _output_shape(shape, n, dim)
    out_dtype = _output_dtype(dtype)
    ref_buf, _ = _make_out_buffer(out_shape, out_dtype, "fresh", ref_inp.device)
    ref_buf = torch.ops.aten.fft_rfft.out(ref_inp, n, dim, norm, out=ref_buf)

    buf, guards = _make_out_buffer(out_shape, out_dtype, geometry, flag_gems.device)
    # A correctly shaped view must keep its own layout: a resize_ or a
    # contiguous rewrite would silently discard the caller's buffer.
    layout = (buf.stride(), buf.storage_offset())

    res_buf = flag_gems.fft_rfft(inp, n, dim, norm, out=buf)

    assert res_buf is buf
    assert (buf.stride(), buf.storage_offset()) == layout
    tu.assert_result_close(buf, ref_buf)
    for region, snapshot in guards:
        tu.assert_result_equal(region, snapshot)


# Layout rows: the transformed axis stays non-contiguous (stride > 1) through a
# slice or a transpose, and one row carries a non-zero storage offset.
_LAYOUT_CASES = tu.selected_cases(
    [
        ((16, 128, 128), "slice"),
        ((1024, 2048), "slice"),
        ((64,), "offset"),
        ((17, 32), "transpose"),
    ],
    quick=[],
)


def _make_layout_input(base_shape, transform):
    base = tu.make_input(torch.float32, base_shape, ["-1", "1"])
    if transform == "slice":
        return base[:, :, 0:64:2] if base.dim() == 3 else base[:, 0:1024:4]
    if transform == "offset":
        return base[8:]
    return base.t()


@pytest.mark.fft_rfft
@pytest.mark.parametrize("base_shape,transform", _LAYOUT_CASES)
def test_fft_rfft_layouts(base_shape, transform):
    inp = _make_layout_input(base_shape, transform)

    ref_inp = tu.to_reference(inp)
    ref_out = torch.ops.aten.fft_rfft(ref_inp)
    res_out = flag_gems.fft_rfft(inp)

    tu.assert_result_close(res_out, ref_out)


# NaN / +-Inf / signed-zero payloads padded with an explicit n = 8, a valid
# transform length for every supported floating dtype including float16.
_SPECIAL_CASES = tu.selected_cases(
    tu.special_value_cases(_FLOAT_DTYPES),
    quick=[],
)


@pytest.mark.fft_rfft
@pytest.mark.parametrize("dtype,scenario", _SPECIAL_CASES)
def test_fft_rfft_special_values(dtype, scenario):
    inp = tu.make_special_input(dtype, scenario)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp, 8)
    res_out = flag_gems.fft_rfft(inp, 8)

    tu.assert_result_close(res_out, ref_out)


# (dtype, shape, n, dim, norm); the n / dim / norm rows exercise the backward
# paths of those arguments as well.
_BACKWARD_ROWS = [
    (torch.float32, (256,), None, -1, None),
    (torch.float32, (16, 128, 64), None, -1, None),
    (torch.float32, (1024, 1024), 17, 0, "ortho"),
    (torch.float32, (16, 7, 57, 32, 29), 29, -1, "forward"),
]
if _CUFFT_FFT_BACKEND:
    _BACKWARD_ROWS.append((torch.float16, (1024, 1024), None, -1, None))
    _BACKWARD_ROWS.append((torch.float16, (256,), 4, -1, "ortho"))
if utils.fp64_is_supported:
    _BACKWARD_ROWS.append((torch.float64, (1024, 1024), None, -1, None))
    _BACKWARD_ROWS.append((torch.float64, (1024, 1024), 256, 1, "ortho"))

_BACKWARD_CASES = tu.selected_cases(_BACKWARD_ROWS, quick=[])


@pytest.mark.fft_rfft
@pytest.mark.parametrize("dtype,shape,n,dim,norm", _BACKWARD_CASES)
def test_fft_rfft_backward(dtype, shape, n, dim, norm):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp, n, dim, norm)
    res_out = flag_gems.fft_rfft(inp, n, dim, norm)
    tu.assert_result_close(res_out, ref_out)

    # A non-constant complex upstream gradient of the output dtype: kept on the
    # candidate device for the candidate and transferred for the reference.
    grad_out = torch.randn(ref_out.shape, dtype=ref_out.dtype, device=flag_gems.device)
    ref_grad_out = tu.to_reference(grad_out)

    (ref_grad,) = torch.autograd.grad(ref_out, ref_inp, grad_outputs=ref_grad_out)
    (res_grad,) = torch.autograd.grad(res_out, inp, grad_outputs=grad_out)

    tu.assert_result_close(res_grad, ref_grad)


# Arguments that ATen itself rejects on every backend: a 0-dim input, an empty
# transform, and out-of-range n / dim / norm. The zero-length batch rows are
# appended below because their rejection comes from the vendor FFT library.
_NEGATIVE_SHAPE = (4, 8, 4)
_INVALID_ARGUMENT_ROWS = [
    ((), (), "scalar-input"),
    ((0,), (), "empty-transform"),
    (_NEGATIVE_SHAPE, (0,), "n=0"),
    (_NEGATIVE_SHAPE, (-8,), "n=-8"),
    (_NEGATIVE_SHAPE, (None, 3), "dim=3"),
    (_NEGATIVE_SHAPE, (None, -4), "dim=-4"),
    (_NEGATIVE_SHAPE, (None, -1, "circular"), "norm=circular"),
]
if _CUFFT_FFT_BACKEND:
    # Measured 'cuFFT error: CUFFT_INVALID_SIZE' (the CPU reference reports its
    # own 'MKL FFT error'), so this row is scoped to the cuFFT backend.
    _INVALID_ARGUMENT_ROWS += [
        ((0, 8), (), "empty-batch"),
        ((0, 8), (4,), "empty-batch-n4"),
    ]

_INVALID_ARGUMENT_CASES = [(shape, args) for shape, args, _ in _INVALID_ARGUMENT_ROWS]
_INVALID_ARGUMENT_IDS = [name for _, _, name in _INVALID_ARGUMENT_ROWS]


@pytest.mark.fft_rfft
@pytest.mark.parametrize(
    "shape,args",
    _INVALID_ARGUMENT_CASES,
    ids=_INVALID_ARGUMENT_IDS,
)
def test_fft_rfft_rejects_invalid_argument(shape, args):
    inp = tu.make_input(torch.float32, shape, ["-1", "1"])

    with pytest.raises((RuntimeError, ValueError, IndexError)):
        flag_gems.fft_rfft(inp, *args)


@pytest.mark.fft_rfft
@pytest.mark.parametrize(
    "dtype",
    _REJECTED_DTYPES,
    ids=[str(dtype) for dtype in _REJECTED_DTYPES],
)
def test_fft_rfft_rejects_unsupported_input(dtype):
    # Measured natively on the cuFFT backend: bfloat16 and the fp8 dtypes raise
    # 'Unsupported dtype ...', while a complex input raises 'rfft expects a real
    # input tensor' on every backend.
    inp = torch.zeros((4,), dtype=dtype, device=flag_gems.device)

    with pytest.raises((RuntimeError, TypeError, ValueError)):
        flag_gems.fft_rfft(inp)


# The cuFFT half-precision lane only accepts a power-of-two transformed length,
# so the spec shapes with other trailing lengths stay as explicit negative
# coverage instead of being padded away. The rows are built from the backend
# metadata at collection time (no skip decorator); each entry is one shape
# descriptor, since a single-argument parametrization passes every list element
# as one value.
_HALF_ODD_CASES = (
    [
        (2, 19, 7),
        (7,),
        (20, 320, 15),
        (16, 128, 64, 60),
        (16, 7, 57, 32, 29),
    ]
    if _CUFFT_FFT_BACKEND
    else []
)


@pytest.mark.fft_rfft
@pytest.mark.parametrize(
    "shape",
    _HALF_ODD_CASES,
    ids=[str(shape) for shape in _HALF_ODD_CASES],
)
def test_fft_rfft_half_rejects_non_power_of_two(shape):
    inp = tu.make_input(torch.float16, shape, ["-1", "1"])

    with pytest.raises((RuntimeError, ValueError)):
        flag_gems.fft_rfft(inp)


# An explicit power-of-two n is the half-precision valid form of an arbitrary
# natural length: the original odd spec shapes are preserved here rather than
# replaced by an adapted input.
_HALF_PAD_CASES = tu.selected_cases(
    (
        [
            ((20, 320, 15), 16),
            ((16, 7, 57, 32, 29), 8),
        ]
        if _CUFFT_FFT_BACKEND
        else []
    ),
    quick=[],
)


@pytest.mark.fft_rfft
@pytest.mark.parametrize("shape,n", _HALF_PAD_CASES)
def test_fft_rfft_half_padding(shape, n):
    inp = tu.make_input(torch.float16, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fft_rfft(ref_inp, n)
    res_out = flag_gems.fft_rfft(inp, n)

    tu.assert_result_close(res_out, ref_out)
