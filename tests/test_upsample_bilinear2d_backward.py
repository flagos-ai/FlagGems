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

from .accuracy_utils import gems_assert_close, to_reference
from .conftest import TO_CPU

_MUSA_NATIVE_BUG = False
if flag_gems.vendor_name == "mthreads":
    import torch_musa

    _MUSA_NATIVE_BUG = torch_musa.__version__ == "2.9.1+a18d871"

_CANN_SIZE_RATIO = False
if flag_gems.vendor_name == "ascend":
    from torch_npu.npu.utils import get_cann_version

    _CANN_SIZE_RATIO = get_cann_version() in ("8.5.0", "9.0.0")

DTYPES = [
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name in ("ascend", "iluvatar")
            or (
                flag_gems.vendor_name == "mthreads"
                and not TO_CPU
                and not _MUSA_NATIVE_BUG
            ),
            reason=(
                "FP64 arithmetic is unavailable on Iluvatar; "
                "native backward rejects FP64 on Ascend/MUSA; "
                "MUSA CPU-reference validation remains enabled"
            ),
        ),
    ),
    torch.float16,
    torch.bfloat16,
]
CASES = [
    ("identity", (2, 3, 7, 9), (7, 9), None, None, "contiguous"),
    ("identity_scales", (1, 2, 7, 9), (7, 9), 2.5, 3.0, "strided"),
    ("2x", (2, 3, 7, 9), (14, 18), None, None, "contiguous"),
    ("fractional", (3, 2, 7, 11), (13, 17), None, None, "contiguous"),
    ("downsample", (2, 5, 13, 17), (6, 8), None, None, "contiguous"),
    ("strong_downsample", (1, 3, 37, 53), (2, 3), None, None, "contiguous"),
    ("explicit", (2, 3, 7, 11), (9, 15), 2.3, 2.1, "contiguous"),
    ("explicit_h", (1, 4, 9, 7), (13, 11), 2.0, None, "contiguous"),
    ("explicit_w", (1, 2, 7, 13), (11, 19), None, 2.5, "contiguous"),
    ("explicit_downsample", (2, 3, 17, 23), (3, 5), 0.4, 0.5, "contiguous"),
    ("out_h_one", (2, 3, 13, 7), (1, 11), None, None, "contiguous"),
    ("out_w_one", (2, 3, 7, 13), (11, 1), None, None, "contiguous"),
    ("out_one", (2, 3, 7, 13), (1, 1), None, None, "contiguous"),
    ("in_h_one", (2, 3, 1, 7), (11, 13), None, None, "contiguous"),
    ("in_w_one", (2, 3, 7, 1), (13, 11), None, None, "contiguous"),
    ("in_one", (2, 3, 1, 1), (9, 13), None, None, "contiguous"),
    ("single", (1, 1, 1, 1), (1, 1), None, None, "contiguous"),
    ("empty_n", (0, 3, 7, 9), (13, 17), None, None, "contiguous"),
    ("empty_c", (2, 0, 7, 9), (13, 17), None, None, "contiguous"),
    ("strided", (2, 3, 7, 9), (13, 17), None, None, "strided"),
    ("channels_last", (2, 3, 7, 9), (13, 17), None, None, "channels_last"),
    ("large", (2, 7, 63, 95), (127, 189), None, None, "contiguous"),
    ("disjoint", (1, 1, 257, 257), (97, 113), None, None, "strided"),
]


def _make_grad(case, dtype):
    _, input_size, output_size, _, _, layout = case
    n, c = input_size[:2]
    h, w = output_size
    if layout == "strided":
        return torch.randn((n, c, h, w * 2), device=flag_gems.device, dtype=dtype)[
            ..., ::2
        ]
    grad = torch.randn((n, c, h, w), device=flag_gems.device, dtype=dtype)
    if layout == "channels_last":
        grad = grad.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    return grad


def _reference_grad(grad, input_size=None, output_size=None):
    # This exact TorchMUSA release dispatches native bilinear backward to nearest.
    if _MUSA_NATIVE_BUG or (
        _CANN_SIZE_RATIO
        and input_size is not None
        and tuple(input_size[-2:]) == tuple(output_size)
    ):
        # The task requires identity to copy even when ATen applies scales.
        return grad.cpu().to(torch.float64)
    reference = to_reference(grad, upcast=TO_CPU)
    if not TO_CPU and grad.dtype in (torch.float16, torch.bfloat16):
        # The kernel accumulates low-precision gradients in FP32. Native scatter
        # rounds after every atomic update when called directly in FP16/BF16.
        reference = reference.float()
    return reference


def _reference_scales(input_size, output_size, scales_h, scales_w):
    # CANN 8.5.0/9.0.0's directly tested C API ignores nonidentity scales. Match that
    # backend contract in the CPU oracle; identity remains the requested copy.
    if _CANN_SIZE_RATIO and tuple(input_size[-2:]) != tuple(output_size):
        return None, None
    return scales_h, scales_w


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
def test_upsample_bilinear2d_backward(case, align_corners, dtype):
    _, input_size, output_size, sh, sw, _ = case
    grad = _make_grad(case, dtype)
    ref_sh, ref_sw = (
        _reference_scales(input_size, output_size, sh, sw) if TO_CPU else (sh, sw)
    )
    reference = torch.ops.aten.upsample_bilinear2d_backward.default(
        _reference_grad(grad, input_size, output_size),
        output_size,
        input_size,
        align_corners,
        ref_sh,
        ref_sw,
    )
    result = flag_gems.ops.upsample_bilinear2d_backward(
        grad,
        output_size,
        input_size,
        align_corners,
        sh,
        sw,
    )
    assert result.shape == input_size
    assert result.dtype == dtype and result.device == grad.device
    gems_assert_close(
        result.cpu() if reference.device.type == "cpu" else result, reference, dtype
    )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("case", CASES, ids=lambda case: case[0])
@pytest.mark.parametrize("out_layout", ["contiguous", "strided"])
@pytest.mark.upsample_bilinear2d_backward_grad_input
def test_upsample_bilinear2d_backward_grad_input(
    case, align_corners, dtype, out_layout
):
    _, input_size, output_size, sh, sw, _ = case
    grad = _make_grad(case, dtype)
    ref_grad = _reference_grad(grad, input_size, output_size)
    ref_sh, ref_sw = (
        _reference_scales(input_size, output_size, sh, sw) if TO_CPU else (sh, sw)
    )
    ref_out = torch.empty(input_size, dtype=ref_grad.dtype, device=ref_grad.device)
    reference = torch.ops.aten.upsample_bilinear2d_backward.grad_input(
        ref_grad,
        output_size,
        input_size,
        align_corners,
        ref_sh,
        ref_sw,
        grad_input=ref_out,
    )
    n, c, h, w = input_size
    if out_layout == "strided":
        output = torch.empty((n, c, h, w * 2), device=grad.device, dtype=dtype)[
            ..., ::2
        ]
    else:
        output = torch.empty(input_size, device=grad.device, dtype=dtype)
    strides = output.stride()
    result = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
        grad,
        output_size,
        input_size,
        align_corners,
        sh,
        sw,
        grad_input=output,
    )
    assert result is output and reference is ref_out
    assert result.stride() == strides
    gems_assert_close(
        result.cpu() if reference.device.type == "cpu" else result, reference, dtype
    )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("pixel", [(0, 0), (0, 6), (4, 0), (4, 6), (2, 3)])
def test_upsample_bilinear2d_backward_edges(align_corners, pixel):
    grad = torch.zeros((1, 1, 5, 7), device=flag_gems.device)
    grad[0, 0, pixel[0], pixel[1]] = 1
    reference = torch.ops.aten.upsample_bilinear2d_backward.default(
        _reference_grad(grad),
        (5, 7),
        (1, 1, 3, 4),
        align_corners,
    )
    result = flag_gems.ops.upsample_bilinear2d_backward(
        grad,
        (5, 7),
        (1, 1, 3, 4),
        align_corners,
    )
    gems_assert_close(
        result.cpu() if reference.device.type == "cpu" else result,
        reference,
        torch.float32,
    )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("shape", [(0,), (7,)])
@pytest.mark.upsample_bilinear2d_backward_grad_input
def test_upsample_bilinear2d_backward_grad_input_resize(shape):
    grad = torch.randn((1, 2, 5, 7), device=flag_gems.device)
    output = torch.empty(shape, device=grad.device)
    ref_grad = _reference_grad(grad)
    ref_out = torch.empty(shape, dtype=ref_grad.dtype, device=ref_grad.device)
    reference = torch.ops.aten.upsample_bilinear2d_backward.grad_input(
        ref_grad,
        (5, 7),
        (1, 2, 3, 4),
        False,
        grad_input=ref_out,
    )
    result = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
        grad,
        (5, 7),
        (1, 2, 3, 4),
        False,
        grad_input=output,
    )
    assert result is output and reference is ref_out
    assert result.shape == ref_out.shape == (1, 2, 3, 4)
    gems_assert_close(
        result.cpu() if reference.device.type == "cpu" else result,
        reference,
        torch.float32,
    )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("mismatch", ["dtype", "device"])
@pytest.mark.upsample_bilinear2d_backward_grad_input
def test_upsample_bilinear2d_backward_grad_input_errors(mismatch):
    grad = torch.randn((1, 2, 5, 7), device=flag_gems.device)
    output = torch.empty(
        (1, 2, 3, 4),
        device="cpu" if mismatch == "device" else grad.device,
        dtype=torch.float16 if mismatch == "dtype" else grad.dtype,
    )
    with pytest.raises(RuntimeError):
        torch.ops.aten.upsample_bilinear2d_backward.grad_input(
            grad,
            (5, 7),
            (1, 2, 3, 4),
            False,
            grad_input=output,
        )
    with pytest.raises(RuntimeError):
        flag_gems.ops.upsample_bilinear2d_backward_grad_input(
            grad,
            (5, 7),
            (1, 2, 3, 4),
            False,
            grad_input=output,
        )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize(
    "shape,output_size,input_size",
    [
        ((2, 5, 7), (5, 7), (1, 2, 3, 4)),
        ((1, 2, 5, 7), (5,), (1, 2, 3, 4)),
        ((1, 2, 5, 7), (5, 7), (2, 3, 4)),
        ((1, 2, 5, 7), (5, 8), (1, 2, 3, 4)),
        ((1, 2, 5, 7), (5, 7), (2, 2, 3, 4)),
        ((1, 2, 5, 7), (5, 7), (1, 3, 3, 4)),
        ((1, 2, 5, 7), (5, 7), (1, 2, 0, 4)),
        ((1, 2, 5, 7), (0, 7), (1, 2, 3, 4)),
    ],
)
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.parametrize("out_variant", [False, True])
def test_upsample_bilinear2d_backward_invalid(
    shape, output_size, input_size, out_variant
):
    grad = torch.randn(shape, device=flag_gems.device)
    if out_variant:
        output = torch.empty(0, device=grad.device)
        with pytest.raises(RuntimeError):
            torch.ops.aten.upsample_bilinear2d_backward.grad_input(
                grad, output_size, input_size, False, grad_input=output
            )
        with pytest.raises(RuntimeError):
            flag_gems.ops.upsample_bilinear2d_backward_grad_input(
                grad, output_size, input_size, False, grad_input=output
            )
    else:
        with pytest.raises(RuntimeError):
            torch.ops.aten.upsample_bilinear2d_backward.default(
                grad, output_size, input_size, False
            )
        with pytest.raises(RuntimeError):
            flag_gems.ops.upsample_bilinear2d_backward(
                grad, output_size, input_size, False
            )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan")])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.parametrize("out_variant", [False, True])
def test_upsample_bilinear2d_backward_scale_contract(scale, axis, out_variant):
    # The explicit positive-scale contract is stricter than ATen's inferred-scale fallback.
    scales = [None, None]
    scales[axis] = scale
    grad = torch.randn((1, 2, 5, 7), device=flag_gems.device)
    with pytest.raises(RuntimeError, match="greater than 0"):
        if out_variant:
            output = torch.empty((1, 2, 3, 4), device=grad.device)
            flag_gems.ops.upsample_bilinear2d_backward_grad_input(
                grad, (5, 7), (1, 2, 3, 4), False, *scales, grad_input=output
            )
        else:
            flag_gems.ops.upsample_bilinear2d_backward(
                grad, (5, 7), (1, 2, 3, 4), False, *scales
            )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.parametrize("out_variant", [False, True])
@pytest.mark.parametrize("warn_only", [False, True])
def test_upsample_bilinear2d_backward_deterministic(
    align_corners, out_variant, warn_only
):
    grad = torch.randn((1, 2, 127, 133), device=flag_gems.device)
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        if out_variant:
            output = torch.empty((1, 2, 65, 67), device=grad.device)
            first = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
                grad, (127, 133), (1, 2, 65, 67), align_corners, grad_input=output
            )
            assert first is output
            saved = first.clone()
            second = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
                grad, (127, 133), (1, 2, 65, 67), align_corners, grad_input=output
            )
            assert second is output
            assert torch.equal(saved, second)
        else:
            first = flag_gems.ops.upsample_bilinear2d_backward(
                grad, (127, 133), (1, 2, 65, 67), align_corners
            )
            second = flag_gems.ops.upsample_bilinear2d_backward(
                grad, (127, 133), (1, 2, 65, 67), align_corners
            )
            assert torch.equal(first, second)

    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.parametrize("out_variant", [False, True])
def test_upsample_bilinear2d_backward_clamped_scale_cpu_reference(dtype, out_variant):
    # CUDA ATen can write out of bounds for these explicitly inconsistent scales.
    # The CPU implementation defines the clamped-tail oracle independently.
    grad = torch.randn((1, 2, 5, 7), device=flag_gems.device, dtype=dtype)
    reference = torch.ops.aten.upsample_bilinear2d_backward.default(
        grad.cpu().to(torch.float64),
        (5, 7),
        (1, 2, 3, 4),
        False,
        *_reference_scales((1, 2, 3, 4), (5, 7), 0.3, 0.4),
    )
    if out_variant:
        output = torch.empty((1, 2, 3, 8), device=grad.device, dtype=dtype)[..., ::2]
        result = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
            grad, (5, 7), (1, 2, 3, 4), False, 0.3, 0.4, grad_input=output
        )
        assert result is output
    else:
        result = flag_gems.ops.upsample_bilinear2d_backward(
            grad, (5, 7), (1, 2, 3, 4), False, 0.3, 0.4
        )
    gems_assert_close(result.cpu(), reference, dtype)


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
def test_upsample_bilinear2d_backward_grad_input_channels_last(dtype, align_corners):
    grad = torch.randn((2, 13, 17, 8), device=flag_gems.device, dtype=dtype).permute(
        0, 3, 1, 2
    )
    output = torch.empty((2, 7, 9, 8), device=grad.device, dtype=dtype).permute(
        0, 3, 1, 2
    )
    reference = torch.ops.aten.upsample_bilinear2d_backward.default(
        _reference_grad(grad), (13, 17), (2, 8, 7, 9), align_corners
    )
    result = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
        grad, (13, 17), (2, 8, 7, 9), align_corners, grad_input=output
    )
    assert result is output
    assert result.is_contiguous(memory_format=torch.channels_last)
    gems_assert_close(
        result.cpu() if reference.device.type == "cpu" else result, reference, dtype
    )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.skipif(
    flag_gems.vendor_name in ("ascend", "iluvatar"),
    reason="Ascend native rejects DT_DOUBLE; Iluvatar FP64 arithmetic is unavailable",
)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("out_variant", [False, True])
@pytest.mark.parametrize(
    "input_hw,output_hw",
    [((65, 67), (129, 133)), ((64, 64), (128, 128)), ((128, 128), (64, 64))],
)
def test_upsample_bilinear2d_backward_fp64_opmath(
    align_corners, out_variant, input_hw, output_hw
):
    cpu = torch.full((1, 16, *output_hw), 1.0 + 2.0**-30, dtype=torch.float64)
    grad = cpu.to(flag_gems.device)
    reference = torch.ops.aten.upsample_bilinear2d_backward.default(
        cpu, output_hw, (1, 16, *input_hw), align_corners
    )
    if out_variant:
        output = torch.empty(
            (1, 16, input_hw[0], input_hw[1] * 2),
            device=grad.device,
            dtype=torch.float64,
        )[..., ::2]
        result = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
            grad, output_hw, (1, 16, *input_hw), align_corners, grad_input=output
        )
        assert result is output
    else:
        result = flag_gems.ops.upsample_bilinear2d_backward(
            grad, output_hw, (1, 16, *input_hw), align_corners
        )
    torch.testing.assert_close(result.cpu(), reference, rtol=2e-12, atol=2e-12)


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("out_variant", [False, True])
@pytest.mark.parametrize(
    "input_size,output_size,align_corners,pixel,dtype",
    [
        ((1, 1, 3, 4), (5, 7), False, (2, 3), torch.float32),
        ((1, 1, 3, 4), (6, 8), False, (0, 0), torch.float32),
        ((1, 1, 3, 4), (6, 8), False, (5, 7), torch.float32),
        ((1, 1, 1, 3), (2, 6), False, (0, 0), torch.float32),
        ((1, 1, 1, 3), (2, 6), False, (1, 5), torch.float32),
        ((1, 1, 6, 8), (3, 4), False, (1, 2), torch.float32),
        ((1, 1, 3, 4), (5, 7), True, (0, 0), torch.float32),
        ((1, 2, 65, 67), (127, 133), False, (0, 0), torch.float32),
        ((1, 2, 65, 67), (127, 133), False, (126, 132), torch.float32),
        ((1, 2, 65, 67), (127, 133), True, (0, 0), torch.float32),
        ((1, 2, 65, 67), (127, 133), True, (126, 132), torch.float32),
        # Both repeated taps have positive coefficients here: Inf must remain Inf.
        ((1, 1, 1, 3), (5, 7), False, (4, 6), torch.float32),
        ((1, 1, 1, 3), (5, 7), True, (4, 6), torch.float32),
        ((1, 1, 257, 257), (97, 113), False, (96, 112), torch.float32),
        ((1, 1, 257, 257), (97, 113), True, (96, 112), torch.float32),
        pytest.param(
            (1, 16, 65, 67),
            (129, 133),
            True,
            (128, 132),
            torch.float64,
            marks=pytest.mark.skipif(
                flag_gems.vendor_name in ("ascend", "iluvatar"),
                reason="Ascend native rejects DT_DOUBLE; Iluvatar FP64 arithmetic is unavailable",
            ),
        ),
    ],
)
def test_upsample_bilinear2d_backward_nonfinite_cpu_reference(
    input_size, output_size, align_corners, pixel, dtype, value, out_variant
):
    cpu = torch.zeros((*input_size[:2], *output_size), dtype=dtype)
    cpu[0, 0, pixel[0], pixel[1]] = value
    reference = torch.ops.aten.upsample_bilinear2d_backward.default(
        cpu, output_size, input_size, align_corners
    )
    grad = cpu.to(flag_gems.device)
    if out_variant:
        n, c, h, w = input_size
        output = torch.empty((n, c, h, 2 * w), device=grad.device, dtype=dtype)[
            ..., ::2
        ]
        result = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
            grad, output_size, input_size, align_corners, grad_input=output
        )
        assert result is output
    else:
        result = flag_gems.ops.upsample_bilinear2d_backward(
            grad, output_size, input_size, align_corners
        )
    torch.testing.assert_close(result.cpu(), reference, rtol=0, atol=0, equal_nan=True)


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar",
    reason="The explicit unsupported-FP64 guard applies to Iluvatar",
)
@pytest.mark.parametrize("out_variant", [False, True])
def test_upsample_bilinear2d_backward_unsupported_fp64(out_variant):
    grad = torch.empty((1, 1, 5, 7), device=flag_gems.device, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="does not support FP64 arithmetic"):
        if out_variant:
            output = torch.empty((1, 1, 3, 4), device=grad.device, dtype=grad.dtype)
            flag_gems.ops.upsample_bilinear2d_backward_grad_input(
                grad, (5, 7), (1, 1, 3, 4), False, grad_input=output
            )
        else:
            flag_gems.ops.upsample_bilinear2d_backward(
                grad, (5, 7), (1, 1, 3, 4), False
            )


@pytest.mark.upsample_bilinear2d_backward
@pytest.mark.upsample_bilinear2d_backward_grad_input
@pytest.mark.parametrize("scale", [1e100, 1e308, float("inf")])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("align_corners", [False, True])
@pytest.mark.parametrize("out_variant", [False, True])
def test_upsample_bilinear2d_backward_extreme_positive_scales(
    scale, dtype, align_corners, out_variant
):
    input_size, output_size = (1, 2, 3, 4), (5, 7)
    grad = torch.randn((1, 2, 5, 7), dtype=dtype, device=flag_gems.device)
    sh, sw = _reference_scales(input_size, output_size, scale, scale)
    reference = torch.ops.aten.upsample_bilinear2d_backward.default(
        grad.cpu().double(), output_size, input_size, align_corners, sh, sw
    )
    if out_variant:
        output = torch.empty(input_size, dtype=dtype, device=grad.device)
        result = flag_gems.ops.upsample_bilinear2d_backward_grad_input(
            grad,
            output_size,
            input_size,
            align_corners,
            scale,
            scale,
            grad_input=output,
        )
        assert result is output
    else:
        result = flag_gems.ops.upsample_bilinear2d_backward(
            grad, output_size, input_size, align_corners, scale, scale
        )
    gems_assert_close(result.cpu(), reference, dtype)
