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

CASES = [
    ((1, 1, 2, 3, 4), (4, 6, 8), (None, None, None)),
    ((2, 3, 4, 5, 6), (6, 8, 3), (None, None, None)),
    ((1, 2, 5, 4, 3), (3, 7, 8), (0.6, 1.75, 8.0 / 3.0)),
    ((1, 1, 2, 3, 4), (4, 6, 8), (1.5, 2.5, 0.75)),
    ((2, 4, 6, 7, 8), (6, 7, 8), (None, 1.0, -1.0)),
]


def _nearest_exact3d_backward_cpu(grad_output, output_size, input_size, *scales):
    # Scatter the output gradient to the input sample selected by each
    # nearest-exact forward coordinate. Explicit scales may leave part of the
    # output beyond the input grid; the CUDA backward drops those contributions.
    scales = scales or (None, None, None)
    input_depth, input_height, input_width = input_size[-3:]
    indices = []
    for source, output, scale in zip(input_size[-3:], output_size, scales):
        factor = scale if scale is not None and scale > 0 else output / source
        index = ((torch.arange(output, dtype=torch.float64) + 0.5) / factor).floor()
        indices.append(index.long())
    depth, height, width = torch.meshgrid(*indices, indexing="ij")
    valid = (depth < input_depth) & (height < input_height) & (width < input_width)
    flat_index = (depth[valid] * input_height + height[valid]) * input_width + width[
        valid
    ]

    acc_dtype = {
        torch.float64: torch.float64,
        torch.uint8: torch.int64,
    }.get(grad_output.dtype, torch.float32)
    batch, channels = input_size[:2]
    totals = torch.zeros(
        (batch, channels, input_depth * input_height * input_width), dtype=acc_dtype
    )
    values = grad_output.to(device="cpu", dtype=acc_dtype)[..., valid]
    totals.scatter_add_(2, flat_index.expand(batch, channels, -1), values)
    return totals.reshape(input_size).to(grad_output.dtype)


def _reference(grad_output, output_size, input_size, *scales):
    if flag_gems.vendor_name == "mthreads":
        # MUSA eager uses legacy-nearest mapping for uneven resize ratios.
        return _nearest_exact3d_backward_cpu(
            grad_output, output_size, input_size, *scales
        )
    if scales == (1.5, 2.5, 0.75):
        # Native CPU clamps gradients for inconsistent explicit scales, unlike CUDA.
        result = torch.ops.aten._upsample_nearest_exact3d_backward.default(
            grad_output, output_size, input_size, *scales
        )
        return result.cpu() if utils.TO_CPU else result
    return torch.ops.aten._upsample_nearest_exact3d_backward.default(
        utils.to_reference(grad_output), output_size, input_size, *scales
    )


def _assert_result(result, ref, grad_output):
    assert result.device == grad_output.device
    if flag_gems.vendor_name == "mthreads":
        result = result.cpu()
    utils.gems_assert_close(result, ref, grad_output.dtype)


@pytest.mark.upsample_nearest_exact3d_backward
@pytest.mark.underscore_upsample_nearest_exact3d_backward
@pytest.mark.parametrize("input_size,output_size,scales", CASES)
@pytest.mark.parametrize("dtype", [*utils.FLOAT_DTYPES, torch.float64])
def test_upsample_nearest_exact3d_backward(input_size, output_size, scales, dtype):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame cannot preserve float64 input storage")
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        pytest.skip("Kunlunxin cannot preserve float64 input storage")
    if flag_gems.vendor_name == "ascend":
        uses_device_reference = not utils.TO_CPU or scales == (1.5, 2.5, 0.75)
        if dtype == torch.float64 and uses_device_reference:
            pytest.skip(
                "Ascend native nearest-exact3d backward does not support float64"
            )
        if dtype == torch.float16 and not uses_device_reference:
            pytest.skip(
                "Ascend environment CPU nearest-exact3d backward raises SIGILL for float16"
            )
    grad_output = torch.randn(
        (*input_size[:2], *output_size), dtype=dtype, device=flag_gems.device
    )
    assert grad_output.dtype == dtype
    ref = _reference(grad_output, output_size, input_size, *scales)
    result = flag_gems._upsample_nearest_exact3d_backward(
        grad_output, output_size, input_size, *scales
    )

    _assert_result(result, ref, grad_output)


@pytest.mark.upsample_nearest_exact3d_backward
@pytest.mark.underscore_upsample_nearest_exact3d_backward
def test_upsample_nearest_exact3d_backward_uint8():
    if flag_gems.vendor_name in ("ascend", "kunlunxin", "enflame"):
        pytest.skip(
            "Native nearest-exact3d backward does not support uint8 on this backend"
        )
    input_size = (1, 2, 3, 4, 5)
    output_size = (7, 6, 9)
    grad_output = torch.randint(
        0,
        256,
        (*input_size[:2], *output_size),
        dtype=torch.uint8,
        device=flag_gems.device,
    )
    if flag_gems.vendor_name == "mthreads":
        ref = _nearest_exact3d_backward_cpu(grad_output, output_size, input_size)
    else:
        ref = torch.ops.aten._upsample_nearest_exact3d_backward.default(
            grad_output, output_size, input_size
        )
    result = flag_gems._upsample_nearest_exact3d_backward(
        grad_output, output_size, input_size
    )
    assert result.device == grad_output.device
    assert result.dtype == grad_output.dtype
    if flag_gems.vendor_name == "mthreads":
        result = result.cpu()
    assert torch.equal(result, ref)


@pytest.mark.upsample_nearest_exact3d_backward
@pytest.mark.underscore_upsample_nearest_exact3d_backward
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_upsample_nearest_exact3d_backward_noncontiguous(dtype):
    if flag_gems.vendor_name == "ascend" and utils.TO_CPU and dtype == torch.float16:
        pytest.skip(
            "Ascend environment CPU nearest-exact3d backward raises SIGILL for float16"
        )
    input_size = (2, 3, 3, 4, 5)
    output_size = (4, 5, 6)
    base = torch.randn((2, 3, 8, 10, 12), dtype=dtype, device=flag_gems.device)
    grad_output = base[:, :, ::2, ::2, ::2]
    assert not grad_output.is_contiguous()

    ref = _reference(grad_output, output_size, input_size)
    result = flag_gems._upsample_nearest_exact3d_backward(
        grad_output, output_size, input_size
    )
    _assert_result(result, ref, grad_output)


@pytest.mark.upsample_nearest_exact3d_backward_grad_input
@pytest.mark.underscore_upsample_nearest_exact3d_backward_grad_input
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_upsample_nearest_exact3d_backward_grad_input(noncontiguous):
    input_size = (2, 3, 3, 4, 5)
    output_size = (5, 7, 9)
    grad_output = torch.randn(
        (*input_size[:2], *output_size),
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_grad_output = (
        grad_output.cpu() if noncontiguous else utils.to_reference(grad_output)
    )

    if noncontiguous:
        grad_input = torch.empty(
            (2, 3, 3, 5, 4), dtype=torch.float32, device=flag_gems.device
        ).transpose(-1, -2)
        assert not grad_input.is_contiguous()
        ref_grad_input = torch.empty(
            (2, 3, 3, 5, 4), dtype=torch.float32, device=ref_grad_output.device
        ).transpose(-1, -2)
    else:
        # The native out overload resizes an empty destination in place.
        grad_input = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
        ref_grad_input = torch.empty(
            0, dtype=torch.float32, device=ref_grad_output.device
        )

    ref = torch.ops.aten._upsample_nearest_exact3d_backward.grad_input(
        ref_grad_output,
        output_size,
        input_size,
        grad_input=ref_grad_input,
    )

    result = flag_gems._upsample_nearest_exact3d_backward_grad_input(
        grad_output,
        output_size,
        input_size,
        None,
        None,
        None,
        grad_input=grad_input,
    )

    assert result is grad_input
    assert ref is ref_grad_input
    assert tuple(result.shape) == input_size
    if noncontiguous:
        torch.testing.assert_close(result.cpu(), ref)
    else:
        utils.gems_assert_close(result, ref, torch.float32)
