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
from .conftest import QUICK_MODE

DTYPES = utils.ALL_FLOAT_DTYPES
SHAPES = [(1, 4), (2, 17), (8, 64), (33, 129)]
if QUICK_MODE:
    SHAPES = [(2, 17)]

BIAS_MODES = [(False, False), (True, False), (False, True), (True, True)]


def _make_inputs(batch_size, hidden_size, dtype, *, noncontiguous=False):
    device = flag_gems.device
    if noncontiguous:
        grad_hy = torch.randn(hidden_size, batch_size, dtype=dtype, device=device).T
        input_gates = torch.randn(
            3 * hidden_size, batch_size, dtype=dtype, device=device
        ).T
        hidden_gates = torch.randn(
            3 * hidden_size, batch_size, dtype=dtype, device=device
        ).T
        hx = torch.randn(hidden_size, batch_size, dtype=dtype, device=device).T
    else:
        grad_hy = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
        input_gates = torch.randn(
            batch_size, 3 * hidden_size, dtype=dtype, device=device
        )
        hidden_gates = torch.randn(
            batch_size, 3 * hidden_size, dtype=dtype, device=device
        )
        hx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    return grad_hy, input_gates, hidden_gates, hx


def _make_bias(hidden_size, dtype, enabled, *, noncontiguous=False):
    if not enabled:
        return None
    size = 3 * hidden_size
    if noncontiguous:
        return torch.randn(size * 2, dtype=dtype, device=flag_gems.device)[::2]
    return torch.randn(size, dtype=dtype, device=flag_gems.device)


def _reference_args(args):
    return tuple(None if value is None else utils.to_reference(value) for value in args)


def _assert_outputs_close(result, reference, dtype, batch_size):
    # Native and fused kernels use different low-precision transcendental
    # implementations, so use the repository's reduction-scaled tolerance.
    base_atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    for output_index, (actual, expected) in enumerate(zip(result, reference)):
        if expected is None:
            assert actual is None
        else:
            assert actual is not None
            # BF16 bias reductions need a slightly wider base tolerance because
            # the fused reduction order differs from TensorIterator reduction.
            atol = 2e-3 if dtype == torch.bfloat16 and output_index >= 3 else base_atol
            utils.gems_assert_close(
                actual,
                expected,
                dtype,
                reduce_dim=max(batch_size, 1),
                atol=atol,
            )


@pytest.mark.thnn_differentiable_gru_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("has_input_bias,has_hidden_bias", BIAS_MODES)
def test_thnn_differentiable_gru_cell_backward(
    dtype, shape, has_input_bias, has_hidden_bias
):
    batch_size, hidden_size = shape
    args = _make_inputs(batch_size, hidden_size, dtype)
    args += (
        _make_bias(hidden_size, dtype, has_input_bias),
        _make_bias(hidden_size, dtype, has_hidden_bias),
    )
    reference = torch.ops.aten._thnn_differentiable_gru_cell_backward(
        *_reference_args(args)
    )

    with flag_gems.use_gems():
        result = torch.ops.aten._thnn_differentiable_gru_cell_backward(*args)

    _assert_outputs_close(result, reference, dtype, batch_size)


@pytest.mark.thnn_differentiable_gru_cell_backward
@pytest.mark.parametrize("dtype", DTYPES)
def test_thnn_differentiable_gru_cell_backward_noncontiguous(dtype):
    batch_size, hidden_size = 5, 19
    args = _make_inputs(batch_size, hidden_size, dtype, noncontiguous=True)
    args += (
        _make_bias(hidden_size, dtype, True, noncontiguous=True),
        _make_bias(hidden_size, dtype, True, noncontiguous=True),
    )
    assert all(not tensor.is_contiguous() for tensor in args)
    reference = torch.ops.aten._thnn_differentiable_gru_cell_backward(
        *_reference_args(args)
    )

    with flag_gems.use_gems():
        result = flag_gems._thnn_differentiable_gru_cell_backward(*args)

    _assert_outputs_close(result, reference, dtype, batch_size)


@pytest.mark.thnn_differentiable_gru_cell_backward
@pytest.mark.parametrize("shape", [(0, 8), (2, 0)])
def test_thnn_differentiable_gru_cell_backward_empty(shape):
    batch_size, hidden_size = shape
    dtype = torch.float32
    args = _make_inputs(batch_size, hidden_size, dtype)
    args += (
        _make_bias(hidden_size, dtype, True),
        _make_bias(hidden_size, dtype, False),
    )
    reference = torch.ops.aten._thnn_differentiable_gru_cell_backward(
        *_reference_args(args)
    )

    with flag_gems.use_gems():
        result = torch.ops.aten._thnn_differentiable_gru_cell_backward(*args)

    _assert_outputs_close(result, reference, dtype, batch_size)


@pytest.mark.thnn_differentiable_gru_cell_backward
def test_thnn_differentiable_gru_cell_backward_invalid_shape():
    args = _make_inputs(2, 8, torch.float32)
    bad_input_gates = torch.randn(2, 23, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._thnn_differentiable_gru_cell_backward(
            args[0], bad_input_gates, args[2], args[3], None, None
        )
