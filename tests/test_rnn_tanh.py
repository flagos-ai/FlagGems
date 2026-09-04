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

from contextlib import nullcontext

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

pytestmark = pytest.mark.rnn_tanh

# T-Head's ACDNN currently rejects tanh RNN mode.  Disable that unavailable
# fast path so Torch's generic implementation remains usable as the reference.
if flag_gems.vendor_name == "thead":
    torch.backends.cudnn.enabled = False

_RNN_ACCELERATOR_AVAILABLE = not cfg.TO_CPU and (
    (flag_gems.device == "cuda" and torch.cuda.is_available())
    or (
        flag_gems.device == "npu" and hasattr(torch, "npu") and torch.npu.is_available()
    )
)


@pytest.fixture(autouse=True)
def _ieee_reference():
    # Accuracy comparisons must not mix FP32 Gems math with a TF32 reference.
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def _make_case(
    seq_len,
    batch_size,
    input_size,
    hidden_size,
    dtype,
    num_layers=1,
    has_biases=True,
    bidirectional=False,
    batch_first=False,
):
    shape = (
        (batch_size, seq_len, input_size)
        if batch_first
        else (seq_len, batch_size, input_size)
    )
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    directions = 2 if bidirectional else 1
    hx = torch.randn(
        num_layers * directions,
        batch_size,
        hidden_size,
        dtype=dtype,
        device=flag_gems.device,
    )
    rnn = torch.nn.RNN(
        input_size,
        hidden_size,
        num_layers,
        nonlinearity="tanh",
        bias=has_biases,
        bidirectional=bidirectional,
        batch_first=batch_first,
    ).to(dtype=dtype, device=flag_gems.device)
    return inp, hx, tuple(rnn._flat_weights)


def _reference_rnn(op, *args):
    def to_reference(arg):
        if isinstance(arg, torch.Tensor):
            return utils.to_reference(arg)
        if isinstance(arg, (tuple, list)):
            return type(arg)(to_reference(value) for value in arg)
        return arg

    return op(*(to_reference(arg) for arg in args))


def _assert_rnn_close(actual, expected, dtype):
    # Smallest passing native Torch CPU/GPU atols across the forward shapes
    # with seeds 0, 1, 2, 2026 and the unchanged dtype-specific Gems rtol.
    # FP32 uses the framework default; these bounds are not fitted to Gems.
    atol = {
        torch.float32: 1e-4,
        torch.float16: 0.0011334419832564893,
        torch.bfloat16: 0.007965088356286289,
    }[dtype]
    utils.gems_assert_close(actual[0], expected[0], dtype, atol=atol)
    utils.gems_assert_close(actual[1], expected[1], dtype, atol=atol)


@pytest.mark.rnn_tanh
@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "num_layers,has_biases,bidirectional,batch_first",
    [
        (1, True, False, False),
        (1, False, False, True),
        (1, True, True, False),
        (2, True, False, True),
        (2, False, True, False),
    ],
)
def test_rnn_tanh_forward_modes(
    dtype, num_layers, has_biases, bidirectional, batch_first
):
    inp, hx, params = _make_case(
        5,
        3,
        16,
        24,
        dtype,
        num_layers,
        has_biases,
        bidirectional,
        batch_first,
    )
    reference = _reference_rnn(
        torch.ops.aten.rnn_tanh.input,
        inp,
        hx,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )
    actual = flag_gems.rnn_tanh(
        inp,
        hx,
        params,
        has_biases,
        num_layers,
        0.0,
        False,
        bidirectional,
        batch_first,
    )
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_bfloat16_medium_hidden():
    """Check a medium hidden size with bfloat16 inputs."""
    dtype = torch.bfloat16
    inp, hx, params = _make_case(3, 2, 64, 128, dtype)
    reference = _reference_rnn(
        torch.ops.aten.rnn_tanh.input,
        inp,
        hx,
        params,
        True,
        1,
        0.0,
        False,
        False,
        False,
    )
    actual = flag_gems.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("seq_len", [17, 18, 20])
def test_rnn_tanh_bfloat16_long_bidirectional(seq_len):
    """Check long bidirectional sequences across two layers."""
    dtype = torch.bfloat16
    inp, hx, params = _make_case(
        seq_len, 3, 64, 128, dtype, num_layers=2, bidirectional=True
    )
    reference = _reference_rnn(
        torch.ops.aten.rnn_tanh.input, inp, hx, params, True, 2, 0.0, False, True, False
    )
    actual = flag_gems.rnn_tanh(inp, hx, params, True, 2, 0.0, False, True, False)
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize(
    "shape",
    [(16, 4, 32), (32, 8, 64), (64, 16, 128)],
)
def test_rnn_tanh_comprehensive_forward_shapes(dtype, shape):
    """Check representative sequence, batch, and hidden sizes."""
    seq_len, batch_size, hidden_size = shape
    inp, hx, params = _make_case(
        seq_len,
        batch_size,
        hidden_size,
        hidden_size,
        dtype,
    )
    reference = _reference_rnn(
        torch.ops.aten.rnn_tanh.input,
        inp,
        hx,
        params,
        True,
        1,
        0.0,
        False,
        False,
        False,
    )
    actual = flag_gems.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_large_hidden():
    inp, hx, params = _make_case(3, 2, 32, 320, torch.float32)
    reference = _reference_rnn(
        torch.ops.aten.rnn_tanh.input,
        inp,
        hx,
        params,
        True,
        1,
        0.0,
        False,
        False,
        False,
    )
    actual = flag_gems.rnn_tanh(inp, hx, params, True, 1, 0.0, False, False, False)
    _assert_rnn_close(actual, reference, torch.float32)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_dropout_one():
    inp, hx, params = _make_case(5, 3, 16, 16, torch.float32, num_layers=2)
    # Hygon's Torch 2.4.1 MIOpen RNN path ignores dropout. Use the generic
    # reference only for this dropout test; restore the backend setting before
    # running Gems and leave all performance baselines unchanged.
    reference_context = (
        torch.backends.cudnn.flags(enabled=False)
        if flag_gems.vendor_name == "hygon"
        else nullcontext()
    )
    with reference_context:
        reference = _reference_rnn(
            torch.ops.aten.rnn_tanh.input,
            inp,
            hx,
            params,
            True,
            2,
            1.0,
            True,
            False,
            False,
        )
    actual = flag_gems.rnn_tanh(inp, hx, params, True, 2, 1.0, True, False, False)
    _assert_rnn_close(actual, reference, torch.float32)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
@pytest.mark.parametrize("bidirectional", [False, True])
@pytest.mark.parametrize("num_layers", [1, 2])
def test_rnn_tanh_packed_forward(num_layers, bidirectional):
    dtype = torch.float32
    input_size, hidden_size = 16, 16
    padded = torch.randn(5, 4, input_size, dtype=dtype)
    lengths = torch.tensor([5, 4, 2, 1], dtype=torch.int64)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        padded, lengths, enforce_sorted=True
    )
    packed_data = packed.data.to(flag_gems.device)
    directions = 2 if bidirectional else 1
    hx = torch.randn(
        num_layers * directions,
        4,
        hidden_size,
        device=flag_gems.device,
        dtype=dtype,
    )
    module = torch.nn.RNN(
        input_size,
        hidden_size,
        num_layers,
        bidirectional=bidirectional,
    ).to(device=flag_gems.device, dtype=dtype)
    params = tuple(module._flat_weights)

    reference = _reference_rnn(
        torch.ops.aten.rnn_tanh.data,
        packed_data,
        packed.batch_sizes,
        hx,
        params,
        True,
        num_layers,
        0.0,
        True,
        bidirectional,
    )
    actual = flag_gems.rnn_tanh_data(
        packed_data,
        packed.batch_sizes,
        hx,
        params,
        True,
        num_layers,
        0.0,
        True,
        bidirectional,
    )
    _assert_rnn_close(actual, reference, dtype)


@pytest.mark.skipif(
    not _RNN_ACCELERATOR_AVAILABLE,
    reason="Triton RNN kernel requires a CUDA or NPU accelerator",
)
def test_rnn_tanh_packed_dispatch():
    padded = torch.randn(4, 3, 16)
    lengths = torch.tensor([4, 3, 1], dtype=torch.int64)
    packed = torch.nn.utils.rnn.pack_padded_sequence(padded, lengths)
    packed_data = packed.data.to(flag_gems.device)
    hx = torch.randn(1, 3, 16, device=flag_gems.device)
    module = torch.nn.RNN(16, 16).to(flag_gems.device)
    params = tuple(module._flat_weights)
    reference = _reference_rnn(
        torch.ops.aten.rnn_tanh.data,
        packed_data,
        packed.batch_sizes,
        hx,
        params,
        True,
        1,
        0.0,
        False,
        False,
    )
    actual = flag_gems.rnn_tanh_data(
        packed_data,
        packed.batch_sizes,
        hx,
        params,
        True,
        1,
        0.0,
        False,
        False,
    )
    _assert_rnn_close(actual, reference, torch.float32)
