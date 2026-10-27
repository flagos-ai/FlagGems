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

from importlib.metadata import version

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

_MUSA_NATIVE_IS_NEAREST = False
if flag_gems.vendor_name == "mthreads":
    import torch_musa

    _MUSA_NATIVE_IS_NEAREST = torch_musa.__version__ == "2.9.1+a18d871"
_CPU_REFERENCE = utils.TO_CPU or _MUSA_NATIVE_IS_NEAREST
_ILUVATAR_NATIVE_LAYOUT_BUG = (
    flag_gems.vendor_name == "iluvatar" and version("torch") == "2.7.1+corex.4.4.0"
)

DTYPES = [
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name in ("ascend", "iluvatar")
            or (flag_gems.vendor_name == "mthreads" and not _CPU_REFERENCE),
            reason=(
                "Ascend rejects FP64; Iluvatar FP64 arithmetic is incorrect; "
                "MUSA FP64 kernel is tested against CPU because muDNN rejects DOUBLE"
            ),
        ),
    ),
    torch.float16,
    torch.bfloat16,
]
CASES = [
    ((2, 3, 8, 9), (8, 9), None, None),
    ((2, 3, 8, 9), (8, 9), 0.3, 2.1),
    ((2, 3, 7, 9), (14, 18), None, None),
    ((1, 5, 7, 9), (11, 15), None, None),
    ((2, 2, 13, 17), (5, 8), None, None),
    ((1, 3, 63, 65), (2, 3), None, None),
    ((2, 3, 7, 9), (1, 1), None, None),
    ((1, 3, 1, 1), (17, 23), None, None),
    ((2, 1, 1, 9), (5, 7), None, None),
    ((1, 2, 7, 1), (3, 13), None, None),
    ((1, 2, 3, 4), (13, 17), None, None),
    ((2, 3, 7, 9), (11, 15), 1.7, 1.8),
    ((2, 1, 7, 9), (11, 15), 0.3, 0.6),
    ((1, 2, 7, 9), (11, 15), 0.7, None),
    ((1, 2, 7, 9), (11, 15), None, 2.3),
    ((0, 3, 7, 9), (11, 15), None, None),
    ((2, 0, 7, 9), (11, 15), None, None),
]


def _make_grad(shape, dtype, layout):
    values = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if layout == "strided":
        backing = torch.empty(
            (*shape[:-1], shape[-1] * 2), dtype=dtype, device=flag_gems.device
        )
        result = backing[..., ::2]
        result.copy_(values)
        return result
    if layout == "channels_last":
        return values.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    if layout == "transposed":
        return values.transpose(-1, -2).contiguous().transpose(-1, -2)
    return values


def _check(go, result, case, align_corners, out_variant=False):
    shape, output_size, scales_h, scales_w = case
    reference_input = (
        go.cpu().to(torch.float64) if _CPU_REFERENCE else utils.to_reference(go)
    )
    # Validate the required FP32 accumulation for half/BF16. Native CUDA's
    # same-dtype atomic accumulation rounds after every contribution instead.
    # CPU mode uses FP64; device mode uses FP32 opmath for low-precision inputs.
    # TorchMUSA 2.9.1+a18d871 dispatches bicubic backward to nearest, so that
    # exact release always uses the independent CPU FP64 cubic reference.
    if not _CPU_REFERENCE and go.dtype in (torch.float16, torch.bfloat16):
        reference_input = reference_input.to(torch.float32)
    if (
        _ILUVATAR_NATIVE_LAYOUT_BUG
        and not _CPU_REFERENCE
        and reference_input.is_contiguous(memory_format=torch.channels_last)
    ):
        # This exact CoreX release misinterprets channels-last native inputs.
        # Canonicalize only the native reference; Gems receives the original view.
        reference_input = reference_input.contiguous()
    if out_variant:
        # Native CPU bicubic backward writes contiguous storage for its out tensor.
        # Keep the mathematical reference contiguous while testing result strides.
        reference_output = torch.empty(
            shape,
            dtype=reference_input.dtype,
            device=reference_input.device,
        )
        ref = torch.ops.aten.upsample_bicubic2d_backward.grad_input(
            reference_input,
            output_size,
            shape,
            align_corners,
            scales_h,
            scales_w,
            grad_input=reference_output,
        )
        assert ref is reference_output
    else:
        ref = torch.ops.aten.upsample_bicubic2d_backward.default(
            reference_input, output_size, shape, align_corners, scales_h, scales_w
        )
    assert result.shape == shape
    assert result.dtype == go.dtype
    assert result.device == go.device
    utils.gems_assert_close(result.cpu() if _CPU_REFERENCE else result, ref, go.dtype)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
def test_upsample_bicubic2d_backward(case, align_corners, dtype):
    shape, output_size, scales_h, scales_w = case
    go = _make_grad((*shape[:2], *output_size), dtype, "contiguous")
    result = flag_gems.ops.upsample_bicubic2d_backward(
        go, output_size, shape, align_corners, scales_h, scales_w
    )
    _check(go, result, case, align_corners)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
@pytest.mark.upsample_bicubic2d_backward_grad_input
def test_upsample_bicubic2d_backward_grad_input(case, align_corners, dtype):
    shape, output_size, scales_h, scales_w = case
    go = _make_grad((*shape[:2], *output_size), dtype, "contiguous")
    grad_input = torch.empty(shape, dtype=dtype, device=flag_gems.device)
    result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
        go, output_size, shape, align_corners, scales_h, scales_w, grad_input=grad_input
    )
    assert result is grad_input
    _check(go, result, case, align_corners, out_variant=True)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("layout", ["strided", "channels_last", "transposed"])
@pytest.mark.parametrize("output_size", [(11, 15), (7, 9)], ids=["resize", "copy"])
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
def test_upsample_bicubic2d_backward_layout(
    layout, output_size, align_corners, dtype, out_variant
):
    case = ((2, 3, 7, 9), output_size, 1.3, 1.7)
    go = _make_grad((2, 3, *output_size), dtype, layout)
    if out_variant:
        if layout == "strided":
            storage = torch.full(
                (2, 3, 7, 18), 987, device=flag_gems.device, dtype=dtype
            )
            out = storage[..., ::2]
        else:
            out = _make_grad(case[0], dtype, layout)
        strides = out.stride()
        result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
            go, case[1], case[0], align_corners, case[2], case[3], grad_input=out
        )
        assert result is out and result.stride() == strides
        if layout == "strided":
            torch.testing.assert_close(
                storage[..., 1::2], torch.full_like(out, 987), rtol=0, atol=0
            )
    else:
        result = flag_gems.ops.upsample_bicubic2d_backward(
            go, case[1], case[0], align_corners, case[2], case[3]
        )
    _check(go, result, case, align_corners, out_variant=out_variant)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
@pytest.mark.parametrize("point", [(0, 0), (0, 8), (6, 0), (6, 8), (3, 4)])
def test_upsample_bicubic2d_backward_impulse(align_corners, dtype, out_variant, point):
    case = ((1, 1, 3, 4), (7, 9), None, None)
    go = torch.zeros((1, 1, 7, 9), dtype=dtype, device=flag_gems.device)
    go[0, 0, point[0], point[1]] = 1
    if out_variant:
        out = torch.empty(case[0], dtype=dtype, device=flag_gems.device)
        result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
            go, case[1], case[0], align_corners, grad_input=out
        )
        assert result is out
    else:
        result = flag_gems.ops.upsample_bicubic2d_backward(
            go, case[1], case[0], align_corners
        )
    _check(go, result, case, align_corners, out_variant=out_variant)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("out_shape", [(0,), (2, 3), (1, 1, 2, 3)])
@pytest.mark.upsample_bicubic2d_backward_grad_input
def test_upsample_bicubic2d_backward_resize(out_shape):
    go = torch.randn((1, 1, 5, 7), device=flag_gems.device)
    out = torch.empty(out_shape, device=flag_gems.device)
    result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
        go, (5, 7), (1, 1, 3, 4), False, grad_input=out
    )
    assert result is out
    _check(go, result, ((1, 1, 3, 4), (5, 7), None, None), False)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
@pytest.mark.parametrize(
    "error",
    [
        "rank",
        "output_len",
        "input_len",
        "output_zero",
        "input_zero",
        "shape",
        "batch",
        "channel",
        "dtype",
        "scale_zero",
        "scale_negative",
        "scale_nan",
    ],
)
def test_upsample_bicubic2d_backward_invalid(error, out_variant):
    go = torch.empty((1, 2, 5, 7), device=flag_gems.device)
    output_size, input_size, sh, sw = (5, 7), (1, 2, 3, 4), None, None
    if error == "rank":
        go = go[0]
    elif error == "output_len":
        output_size = (5,)
    elif error == "input_len":
        input_size = (1, 2, 3)
    elif error == "output_zero":
        output_size = (0, 7)
    elif error == "input_zero":
        input_size = (1, 2, 0, 4)
    elif error == "shape":
        output_size = (5, 8)
    elif error == "batch":
        input_size = (2, 2, 3, 4)
    elif error == "channel":
        input_size = (1, 3, 3, 4)
    elif error == "dtype":
        go = go.to(torch.int32)
    elif error == "scale_zero":
        sh = 0.0
    elif error == "scale_negative":
        sw = -1.0
    else:
        sh = float("nan")
    with pytest.raises(RuntimeError):
        if out_variant:
            out = torch.empty(0, device=flag_gems.device)
            flag_gems.ops.upsample_bicubic2d_backward_grad_input(
                go, output_size, input_size, False, sh, sw, grad_input=out
            )
        else:
            flag_gems.ops.upsample_bicubic2d_backward(
                go, output_size, input_size, False, sh, sw
            )


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("mismatch", ["dtype", "device"])
@pytest.mark.upsample_bicubic2d_backward_grad_input
def test_upsample_bicubic2d_backward_out_error(mismatch):
    go = torch.empty((1, 1, 5, 7), device=flag_gems.device)
    out = torch.empty(
        (1, 1, 3, 4),
        dtype=torch.float16 if mismatch == "dtype" else go.dtype,
        device="cpu" if mismatch == "device" else go.device,
    )
    with pytest.raises(RuntimeError):
        flag_gems.ops.upsample_bicubic2d_backward_grad_input(
            go, (5, 7), (1, 1, 3, 4), False, grad_input=out
        )


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
def test_upsample_bicubic2d_backward_deterministic(dtype, out_variant):
    go = torch.randn((2, 3, 13, 17), device=flag_gems.device, dtype=dtype)
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        if out_variant:
            out = torch.empty((2, 3, 5, 7), device=go.device, dtype=dtype)
            first = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
                go, (13, 17), (2, 3, 5, 7), False, grad_input=out
            ).clone()
            second = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
                go, (13, 17), (2, 3, 5, 7), False, grad_input=out
            )
            assert second is out
        else:
            first = flag_gems.ops.upsample_bicubic2d_backward(
                go, (13, 17), (2, 3, 5, 7), False
            )
            second = flag_gems.ops.upsample_bicubic2d_backward(
                go, (13, 17), (2, 3, 5, 7), False
            )
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize(
    "shape,output_size,point",
    [
        ((1, 1, 3, 3), (5, 5), (2, 2)),
        ((1, 1, 1, 1), (3, 3), (0, 0)),
        ((1, 1, 1, 3), (1, 5), (0, 2)),
        ((1, 1, 5, 5), (9, 9), (4, 4)),
        ((1, 1, 5, 5), (10, 10), (5, 5)),
        ((1, 1, 129, 129), (9, 9), (4, 4)),
    ],
)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
def test_upsample_bicubic2d_backward_nonfinite_cpu_reference(
    shape, output_size, point, align_corners, value, out_variant
):
    cpu = torch.zeros((*shape[:2], *output_size), dtype=torch.float32)
    cpu[0, 0, point[0], point[1]] = value
    go = cpu.to(flag_gems.device)
    reference = torch.ops.aten.upsample_bicubic2d_backward.default(
        cpu, output_size, shape, align_corners
    )
    if out_variant:
        out = torch.empty(shape, dtype=go.dtype, device=go.device)
        result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
            go, output_size, shape, align_corners, grad_input=out
        )
        assert result is out
    else:
        result = flag_gems.ops.upsample_bicubic2d_backward(
            go, output_size, shape, align_corners
        )
    torch.testing.assert_close(result.cpu(), reference, equal_nan=True, rtol=0, atol=0)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "iluvatar"),
    reason="Ascend rejects FP64; Iluvatar native and compiled FP64 arithmetic fail nonzero CPU reference inputs",
)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
def test_upsample_bicubic2d_backward_fp64_opmath(align_corners, out_variant):
    shape, output_size = (2, 3, 7, 9), (13, 17)
    cpu = torch.linspace(-2, 2, 2 * 3 * 13 * 17, dtype=torch.float64).reshape(
        2, 3, 13, 17
    )
    go = cpu.to(flag_gems.device)
    reference = torch.ops.aten.upsample_bicubic2d_backward.default(
        cpu, output_size, shape, align_corners, 1.7, 1.9
    )
    if out_variant:
        out = torch.empty(shape, dtype=go.dtype, device=go.device)
        result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
            go, output_size, shape, align_corners, 1.7, 1.9, grad_input=out
        )
        assert result is out
    else:
        result = flag_gems.ops.upsample_bicubic2d_backward(
            go, output_size, shape, align_corners, 1.7, 1.9
        )
    torch.testing.assert_close(result.cpu(), reference, rtol=1e-11, atol=1e-11)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar",
    reason="Explicit FP64 arithmetic rejection applies to Iluvatar",
)
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
def test_upsample_bicubic2d_backward_iluvatar_fp64_rejected(out_variant):
    go = torch.ones((1, 1, 3, 4), dtype=torch.float64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="FP64.*Iluvatar"):
        if out_variant:
            out = torch.empty((1, 1, 2, 3), dtype=go.dtype, device=go.device)
            flag_gems.ops.upsample_bicubic2d_backward_grad_input(
                go, (3, 4), (1, 1, 2, 3), False, grad_input=out
            )
        else:
            flag_gems.ops.upsample_bicubic2d_backward(go, (3, 4), (1, 1, 2, 3), False)


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64", "f16", "b16"])
@pytest.mark.parametrize("scale", [1e100, 1e308, float("inf")])
@pytest.mark.parametrize("axes", ["both", "h", "w"])
@pytest.mark.parametrize("value", [0.0, float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
def test_upsample_bicubic2d_backward_extreme_positive_scale(
    dtype, scale, axes, value, out_variant
):
    shape, output_size = (1, 2, 3, 4), (5, 7)
    scales = (scale if axes != "w" else 2.0, scale if axes != "h" else 2.0)
    cpu = ((torch.arange(70) % 19 - 9) / 8).to(dtype).reshape(1, 2, 5, 7)
    if value != 0.0:
        cpu[0, 0, 2, 3] = value
    go = cpu.to(flag_gems.device)
    reference = torch.ops.aten.upsample_bicubic2d_backward.default(
        cpu.double(), output_size, shape, False, *scales
    )
    if out_variant:
        out = torch.empty(shape, dtype=dtype, device=go.device)
        result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
            go, output_size, shape, False, *scales, grad_input=out
        )
        assert result is out
    else:
        result = flag_gems.ops.upsample_bicubic2d_backward(
            go, output_size, shape, False, *scales
        )
    # These dyadic inputs and limiting half-pixel coordinates have exact weights.
    torch.testing.assert_close(
        result.cpu(), reference.to(dtype), rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.upsample_bicubic2d_backward
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend scalar BF16 rounding regression",
)
@pytest.mark.parametrize(
    "out_variant",
    [
        False,
        pytest.param(True, marks=pytest.mark.upsample_bicubic2d_backward_grad_input),
    ],
)
def test_upsample_bicubic2d_backward_bfloat16_scalar_rounding(out_variant):
    # Small addresses with a large coordinate bound select the second scalar
    # route. The result contains a BF16 halfway value with an even lower bit.
    shape, output_size = (1, 1, 3, 4), (5, 7)
    cpu = torch.zeros((1, 1, 5, 7), dtype=torch.bfloat16)
    cpu.flatten()[:2] = 0.125
    go = cpu.to(flag_gems.device)
    reference = torch.ops.aten.upsample_bicubic2d_backward.default(
        cpu.double(), output_size, shape, False, 1e-6, 2.0
    ).bfloat16()
    if out_variant:
        backing = torch.full((1, 1, 3, 8), 256, dtype=go.dtype, device=go.device)
        out = backing[..., ::2]
        strides = out.stride()
        result = flag_gems.ops.upsample_bicubic2d_backward_grad_input(
            go, output_size, shape, False, 1e-6, 2.0, grad_input=out
        )
        assert result is out and result.stride() == strides
        torch.testing.assert_close(
            backing[..., 1::2], torch.full_like(out, 256), rtol=0, atol=0
        )
    else:
        result = flag_gems.ops.upsample_bicubic2d_backward(
            go, output_size, shape, False, 1e-6, 2.0
        )
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0)
