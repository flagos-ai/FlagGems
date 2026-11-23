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

# aten::_thnn_fused_lstm_cell_backward_impl only ships a CUDA kernel: GCU
# (PrivateUse1) has no kernel for it and there is no CPU kernel either, so the
# reference leg cannot be produced for any reference device. Only that
# reference-dependent accuracy case is skipped for enflame; the shape/dtype
# contract case below stays runnable on GCU.
ATEN_REFERENCE_UNSUPPORTED = flag_gems.vendor_name == "enflame"
ATEN_REFERENCE_SKIP_REASON = (
    "torch aten not support due to: aten::_thnn_fused_lstm_cell_backward_impl is "
    "CUDA-only (no CPU / PrivateUse1 kernel), so the reference leg cannot run"
)

# LSTM cell backward test shapes
LSTM_SHAPES = [
    (1, 4),  # batch=1, hidden=4
    (2, 8),  # batch=2, hidden=8
    (4, 16),  # batch=4, hidden=16
    (8, 32),  # batch=8, hidden=32
    (16, 64),  # batch=16, hidden=64
]


def _make_impl_args(batch_size, hidden_size, dtype):
    """Build (grad_hy, grad_cy, cx, cy, workspace) without any aten reference op.

    The forward leg uses the FlagGems fused LSTM cell kernel (which has a GCU
    kernel), so the inputs come from real gate activations without touching the
    CUDA-only aten forward/backward ops.
    """
    device = flag_gems.device
    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    bias = torch.randn(4 * hidden_size, dtype=dtype, device=device)
    with torch.no_grad():
        hy, cy, workspace = flag_gems._thnn_fused_lstm_cell(
            input_gates, hidden_gates, cx, bias, bias
        )
    grad_hy = torch.randn_like(hy)
    grad_cy = torch.randn_like(cy)
    return grad_hy, grad_cy, cx, cy, workspace


@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell_backward_impl_output_contract(shape, dtype):
    """Check the output contract of _thnn_fused_lstm_cell_backward_impl.

    Runs without the CUDA-only aten reference: verifies shapes, dtypes,
    contiguity and the has_bias=False empty-bias convention.
    """
    batch_size, hidden_size = shape
    grad_hy, grad_cy, cx, cy, workspace = _make_impl_args(
        batch_size, hidden_size, dtype
    )
    with torch.no_grad():
        grad_gates, grad_cx, grad_biases = (
            flag_gems._thnn_fused_lstm_cell_backward_impl(
                grad_hy, grad_cy, cx, cy, workspace, True
            )
        )
        empty_grad_gates, empty_grad_cx, empty_grad_biases = (
            flag_gems._thnn_fused_lstm_cell_backward_impl(
                grad_hy, grad_cy, cx, cy, workspace, False
            )
        )

    expected_gates_shape = (batch_size, 4 * hidden_size)
    assert tuple(grad_gates.shape) == expected_gates_shape
    assert tuple(grad_cx.shape) == (batch_size, hidden_size)
    assert tuple(grad_biases.shape) == (4 * hidden_size,)
    assert grad_gates.dtype == grad_cx.dtype == grad_biases.dtype == dtype
    assert grad_gates.is_contiguous() and grad_cx.is_contiguous()
    assert grad_biases.is_contiguous()
    assert not torch.isnan(grad_gates).any()
    assert not torch.isnan(grad_cx).any()
    # ATen contract: has_bias=False returns empty (not None) bias gradients.
    assert tuple(empty_grad_biases.shape) == (0,)
    assert empty_grad_gates.shape == grad_gates.shape
    assert empty_grad_cx.shape == grad_cx.shape


@pytest.mark.skipif(
    ATEN_REFERENCE_UNSUPPORTED,
    reason=ATEN_REFERENCE_SKIP_REASON,
)
@pytest.mark.thnn_fused_lstm_cell_backward_impl
@pytest.mark.parametrize("shape", LSTM_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_thnn_fused_lstm_cell_backward_impl(shape, dtype):
    """Test accuracy for _thnn_fused_lstm_cell_backward_impl."""
    if flag_gems.vendor_name == "cambricon" and dtype == torch.bfloat16:
        pytest.skip("Issue #5254: Not supported")
    batch_size, hidden_size = shape
    dev = flag_gems.device

    # Create input tensors on CUDA (forward op is CUDA-only)
    input_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    hidden_gates = torch.randn(batch_size, 4 * hidden_size, dtype=dtype, device=dev)
    cx = torch.randn(batch_size, hidden_size, dtype=dtype, device=dev)
    input_bias = torch.zeros(4 * hidden_size, dtype=dtype, device=dev)
    hidden_bias = torch.randn(4 * hidden_size, dtype=dtype, device=dev)

    # Forward pass (CUDA-only; _thnn_fused_lstm_cell has no CPU kernel)
    hx, cy, workspace = torch.ops.aten._thnn_fused_lstm_cell(
        input_gates, hidden_gates, cx, input_bias, hidden_bias
    )

    # Gradient tensors for backward
    grad_hy = torch.randn_like(hx)
    grad_cy = torch.randn_like(cy)

    # Backward pass — ATen reference (outside use_gems, on CUDA)
    ref_out = torch.ops.aten._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, True
    )
    res_out = flag_gems._thnn_fused_lstm_cell_backward_impl(
        grad_hy, grad_cy, cx, cy, workspace, True
    )

    # Compare outputs — ref_out order: (grad_input_gates, grad_cx, grad_biases)
    for i, (ref, res) in enumerate(zip(ref_out, res_out)):
        assert (
            res.shape == ref.shape
        ), f"Shape mismatch at output[{i}]: {res.shape} vs {ref.shape}"
        assert (
            res.dtype == ref.dtype
        ), f"Dtype mismatch at output[{i}]: {res.dtype} vs {ref.dtype}"
        utils.gems_assert_close(res.cpu(), ref.cpu(), dtype, atol=5e-2)
