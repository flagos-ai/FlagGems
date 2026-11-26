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

import subprocess
import sys

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

_SHAPES = [(3, 7)] if cfg.QUICK_MODE else [(), (7,), (3, 7), (5, 65)]


def _make_case(shape, dtype, use_weight, *, noncontiguous=False):
    if noncontiguous:
        N, C = shape
        input = torch.randn((N, C * 2), dtype=dtype, device=flag_gems.device)[:, ::2]
        target = torch.randint(
            0, C, (N * 2,), dtype=torch.int64, device=flag_gems.device
        )[::2]
        weight = torch.randn((C * 2,), dtype=dtype, device=flag_gems.device)[::2]
        return input, target, weight

    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    C = 1 if input.dim() == 0 else input.shape[-1]
    target_shape = (input.shape[0],) if input.dim() == 2 else ()
    target = torch.randint(
        0, C, target_shape, dtype=torch.int64, device=flag_gems.device
    )
    weight = (
        torch.randn((C,), dtype=dtype, device=flag_gems.device) if use_weight else None
    )
    return input, target, weight


def _reference(input, target, p, margin, weight, reduction):
    ref_input = utils.to_reference(input, True)
    ref_target = utils.to_reference(target)
    ref_weight = utils.to_reference(weight, True)
    return torch.ops.aten.multi_margin_loss(
        ref_input,
        ref_target,
        p,
        margin,
        ref_weight,
        reduction,
    )


@pytest.mark.multi_margin_loss
@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_multi_margin_loss(shape, dtype, p, reduction, use_weight):
    input, target, weight = _make_case(shape, dtype, use_weight)
    margin = 0.7
    ref = _reference(input, target, p, margin, weight, reduction)

    result = flag_gems.ops.multi_margin_loss(
        input, target, p, margin, weight, reduction
    )

    if input.dim() <= 1:
        assert result.shape == torch.Size([])
        if ref.numel() == 1:
            ref = ref.reshape(())
    else:
        expected_shape = (input.shape[0],) if reduction == 0 else ()
        assert tuple(result.shape) == expected_shape

    N = input.shape[0] if input.dim() == 2 else 1
    C = input.shape[-1] if input.dim() > 0 else 1
    reduce_dim = C if reduction == 0 else N * C
    utils.gems_assert_close(
        result,
        ref,
        dtype,
        reduce_dim=reduce_dim,
        equal_nan=True,
    )


@pytest.mark.multi_margin_loss
@pytest.mark.multi_margin_loss_backward
@pytest.mark.parametrize("shape", [(7,), (3, 7), (4, 65)])
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_multi_margin_loss_backward(shape, dtype, p, reduction, use_weight):
    input, target, weight = _make_case(shape, dtype, use_weight)
    margin = 0.7
    output_shape = (shape[0],) if len(shape) == 2 and reduction == 0 else ()
    grad_output = torch.randn(output_shape, dtype=dtype, device=flag_gems.device)

    ref = torch.ops.aten.multi_margin_loss_backward(
        utils.to_reference(grad_output, True),
        utils.to_reference(input, True),
        utils.to_reference(target),
        p,
        margin,
        utils.to_reference(weight, True),
        reduction,
    )
    result = flag_gems.ops.multi_margin_loss_backward(
        grad_output,
        input,
        target,
        p,
        margin,
        weight,
        reduction,
    )

    N = input.shape[0] if input.dim() == 2 else 1
    C = input.shape[-1]
    reduce_dim = N * C if reduction == 1 else C
    utils.gems_assert_close(
        result,
        ref,
        dtype,
        reduce_dim=reduce_dim,
        equal_nan=True,
    )


@pytest.mark.multi_margin_loss
@pytest.mark.multi_margin_loss_backward
def test_multi_margin_loss_forward_backward_consistency():
    input, target, weight = _make_case((3, 7), torch.float32, True)
    ref_input = utils.to_reference(input.detach(), True).requires_grad_(True)
    ref_target = utils.to_reference(target)
    ref_weight = utils.to_reference(weight, True)

    ref_output = torch.nn.functional.multi_margin_loss(
        ref_input,
        ref_target,
        p=2,
        margin=0.7,
        weight=ref_weight,
        reduction="mean",
    )
    ref_grad = torch.autograd.grad(ref_output, ref_input)[0]

    output = flag_gems.ops.multi_margin_loss(
        input,
        target,
        2,
        0.7,
        weight,
        1,
    )
    result_grad = flag_gems.ops.multi_margin_loss_backward(
        torch.ones_like(output),
        input,
        target,
        2,
        0.7,
        weight,
        1,
    )

    utils.gems_assert_close(result_grad, ref_grad, torch.float32, reduce_dim=21)


@pytest.mark.multi_margin_loss
@pytest.mark.multi_margin_loss_out
@pytest.mark.multi_margin_loss_backward_out
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_multi_margin_loss_out_variants(dtype, p, reduction, use_weight):
    input, target, weight = _make_case((3, 7), dtype, use_weight)
    out = torch.empty((2,), dtype=input.dtype, device=input.device)
    output_shape = (3,) if reduction == 0 else ()
    grad_output = torch.randn(output_shape, dtype=input.dtype, device=input.device)
    grad_input = torch.empty((1,), dtype=input.dtype, device=input.device)

    result = flag_gems.ops.multi_margin_loss_out(
        input, target, p, 0.7, weight, reduction, out=out
    )
    grad_result = flag_gems.ops.multi_margin_loss_backward_out(
        grad_output,
        input,
        target,
        p,
        0.7,
        weight,
        reduction,
        grad_input=grad_input,
    )

    ref = _reference(input, target, p, 0.7, weight, reduction)
    ref_grad = torch.ops.aten.multi_margin_loss_backward(
        utils.to_reference(grad_output, True),
        utils.to_reference(input, True),
        utils.to_reference(target),
        p,
        0.7,
        utils.to_reference(weight, True),
        reduction,
    )
    assert result is out
    assert grad_result is grad_input
    assert out.shape == torch.Size(output_shape)
    assert grad_input.shape == input.shape
    output_reduce_dim = 7 if reduction == 0 else 21
    grad_reduce_dim = 21 if reduction == 1 else 7
    utils.gems_assert_close(out, ref, dtype, reduce_dim=output_reduce_dim)
    utils.gems_assert_close(
        grad_input,
        ref_grad,
        dtype,
        reduce_dim=grad_reduce_dim,
    )


@pytest.mark.multi_margin_loss
@pytest.mark.parametrize("dtype", utils.ALL_FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multi_margin_loss_empty_batch(dtype, reduction):
    input = torch.empty((0, 7), dtype=dtype, device=flag_gems.device)
    target = torch.empty((0,), dtype=torch.int64, device=flag_gems.device)
    result = flag_gems.ops.multi_margin_loss(input, target, 1, 1.0, None, reduction)

    if reduction == 0:
        assert result.shape == torch.Size([0])
        assert result.numel() == 0
    elif reduction == 1:
        assert result.shape == torch.Size([])
        assert torch.isnan(result).item()
    else:
        assert result.shape == torch.Size([])
        assert result.item() == 0


@pytest.mark.multi_margin_loss
@pytest.mark.parametrize("target_shape", [(), (1,)])
@pytest.mark.parametrize("input_shape", [(), (7,)])
def test_multi_margin_loss_unbatched_none_is_scalar(input_shape, target_shape):
    input = torch.randn(input_shape, dtype=torch.float32, device=flag_gems.device)
    C = 1 if input.dim() == 0 else input.shape[0]
    target = torch.zeros(target_shape, dtype=torch.int64, device=flag_gems.device)
    target.fill_(C - 1)
    result = flag_gems.ops.multi_margin_loss(input, target, 1, 1.0, None, 0)
    assert result.shape == torch.Size([])


@pytest.mark.multi_margin_loss
def test_multi_margin_loss_noncontiguous_inputs():
    input, target, weight = _make_case((5, 65), torch.float32, True, noncontiguous=True)
    assert not input.is_contiguous()
    assert not target.is_contiguous()
    assert not weight.is_contiguous()
    ref = _reference(input, target, 2, 0.7, weight, 0)
    result = flag_gems.ops.multi_margin_loss(input, target, 2, 0.7, weight, 0)
    utils.gems_assert_close(result, ref, torch.float32, reduce_dim=65)


@pytest.mark.multi_margin_loss
@pytest.mark.skipif(
    flag_gems.runtime.device.vendor_name != "hygon",
    reason="covers Hygon-specific fused/partial/generic routing boundaries",
)
@pytest.mark.parametrize("shape", [(129, 256), (129, 1000), (3, 1024), (3, 1025)])
@pytest.mark.parametrize("reduction", [1, 2])
@pytest.mark.parametrize("use_weight", [False, True])
def test_multi_margin_loss_hygon_reduced_routes(shape, reduction, use_weight):
    input, target, weight = _make_case(shape, torch.float32, use_weight)
    ref = _reference(input, target, 2, 0.7, weight, reduction)
    result = flag_gems.ops.multi_margin_loss(
        input,
        target,
        2,
        0.7,
        weight,
        reduction,
    )
    utils.gems_assert_close(
        result,
        ref,
        torch.float32,
        reduce_dim=shape[0] * shape[1],
    )


@pytest.mark.multi_margin_loss
@pytest.mark.parametrize("reduction", [-1, 3, 99])
def test_multi_margin_loss_rejects_invalid_reduction(reduction):
    input, target, _ = _make_case((3, 7), torch.float32, False)
    with pytest.raises(RuntimeError, match="reduction"):
        flag_gems.ops.multi_margin_loss(input, target, 1, 1.0, None, reduction)


@pytest.mark.multi_margin_loss
@pytest.mark.parametrize("p", [0, 1.5, 3])
def test_multi_margin_loss_rejects_invalid_p(p):
    input, target, _ = _make_case((3, 7), torch.float32, False)
    with pytest.raises(RuntimeError, match="p must be 1 or 2"):
        flag_gems.ops.multi_margin_loss(input, target, p, 1.0, None, 1)


@pytest.mark.multi_margin_loss
def test_multi_margin_loss_validates_shapes_and_dtypes():
    input, target, _ = _make_case((3, 7), torch.float32, False)
    cases = [
        (input, target.to(torch.int32), None, "target must have dtype int64"),
        (input, target[:2], None, r"shape \[N\]"),
        (input.reshape(1, 3, 7), target, None, "scalar, 1D, or 2D"),
        (input, target, torch.ones(6, device=input.device), r"shape \[C\]"),
        (
            input,
            target,
            torch.ones(7, dtype=torch.float16, device=input.device),
            "same dtype",
        ),
    ]
    for case_input, case_target, case_weight, match in cases:
        with pytest.raises(RuntimeError, match=match):
            flag_gems.ops.multi_margin_loss(
                case_input, case_target, 1, 1.0, case_weight, 1
            )

    empty_1d = torch.empty((0,), dtype=torch.float32, device=flag_gems.device)
    scalar_target = torch.zeros((), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="non-empty"):
        flag_gems.ops.multi_margin_loss(empty_1d, scalar_target, 1, 1.0, None, 1)

    empty_classes = torch.empty((3, 0), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match="non-empty"):
        flag_gems.ops.multi_margin_loss(empty_classes, target, 1, 1.0, None, 1)


@pytest.mark.multi_margin_loss
def test_multi_margin_loss_requires_vector_target_for_2d_input():
    input = torch.randn((1, 7), dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros((), dtype=torch.int64, device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"shape \[N\]"):
        flag_gems.ops.multi_margin_loss(input, target, 1, 1.0, None, 0)


@pytest.mark.multi_margin_loss
def test_multi_margin_loss_requires_vector_weight():
    input = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros((), dtype=torch.int64, device=flag_gems.device)
    weight = torch.ones((), dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError, match=r"shape \[C\]"):
        flag_gems.ops.multi_margin_loss(input, target, 1, 1.0, weight, 0)


@pytest.mark.multi_margin_loss
def test_multi_margin_loss_out_of_bounds_target_in_subprocess():
    # A CUDA device assert poisons its process context, so the invalid launch
    # must never share the pytest process used by the remaining accuracy cases.
    code = """
import torch
import flag_gems

input = torch.randn((2, 7), dtype=torch.float32, device=flag_gems.device)
target = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
flag_gems.ops.multi_margin_loss(input, target, 1, 1.0, None, 0)
# Backends with a synchronous target check cache valid immutable targets. An
# in-place update must advance Tensor._version and force the second validation.
target[1] = 7
flag_gems.ops.multi_margin_loss(input, target, 1, 1.0, None, 0)
flag_gems.runtime.torch_device_fn.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )
    assert result.returncode != 0, (
        "out-of-bounds target completed successfully\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


@pytest.mark.multi_margin_loss
def test_multi_margin_loss_backward_out_of_bounds_target_in_subprocess():
    code = """
import torch
import flag_gems

input = torch.randn((2, 7), dtype=torch.float32, device=flag_gems.device)
target = torch.tensor([0, 1], dtype=torch.int64, device=flag_gems.device)
grad_output = torch.randn((2,), dtype=torch.float32, device=flag_gems.device)
flag_gems.ops.multi_margin_loss_backward(
    grad_output, input, target, 1, 1.0, None, 0
)
target[1] = 7
flag_gems.ops.multi_margin_loss_backward(
    grad_output, input, target, 1, 1.0, None, 0
)
flag_gems.runtime.torch_device_fn.synchronize()
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
        check=False,
    )
    assert result.returncode != 0, (
        "out-of-bounds backward target completed successfully\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
