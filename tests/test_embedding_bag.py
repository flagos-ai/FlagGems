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

import json
import os
import signal
import subprocess
import sys

import pytest
import torch
import triton
from packaging import version

import flag_gems
from flag_gems import (
    _embedding_bag,
    _embedding_bag_backward,
    _embedding_bag_forward_only,
)

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

DTYPES = [
    torch.float16,
    torch.float32,
    pytest.param(
        torch.bfloat16,
        marks=pytest.mark.skipif(
            not utils.bf16_is_supported, reason="Device does not support bfloat16"
        ),
    ),
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            not utils.fp64_is_supported, reason="Device does not support float64"
        ),
    ),
]
INDEX_DTYPES = [torch.int32, torch.int64]
ASCEND_ABI = flag_gems.vendor_name == "ascend"
_LEGACY_ASCEND = ASCEND_ABI and version.parse(triton.__version__) < version.parse("3.5")
_ERROR_CHILD = "FLAGGEMS_EMBEDDING_BAG_ERROR_CHILD"
CASES = [
    pytest.param([1, 2, 0, 1, 4, 7, 2], [0, 0, 2, 2, 5], False, 0, id="ragged"),
    pytest.param([1, 2, 0, 1, 4, 7, 2], [0, 0, 2, 2, 5, 7], True, 0, id="terminal"),
    pytest.param([0, 0, 0], [0, 2], False, 0, id="padding-only"),
    pytest.param([], [0, 0, 0], False, -1, id="empty-bags"),
    pytest.param([], [], False, -1, id="zero-bags"),
    pytest.param([], [0], True, -1, id="zero-bags-terminal"),
]


def _run_isolated_error(test_name, args):
    # CUDA device assertions invalidate their context. Test each invalid input
    # in a fresh process, with debug disabled in the environment to ensure that
    # operator validation does not depend on a user-supplied debug setting.
    code = """
import json
import sys
from tests import test_embedding_bag as suite
name, args = json.loads(sys.argv[1])
if name in ('test_embedding_bag_invalid_values', 'test_embedding_bag_invalid_after_warmup'):
    args[0] = getattr(suite, args[0])
getattr(suite, name)(*args)
"""
    env = dict(os.environ, TRITON_DEBUG="0")
    env[_ERROR_CHILD] = "1"
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps((test_name, args))],
        env=env,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
    )
    output = result.stdout + result.stderr
    target = (
        "_embedding_bag_backward"
        if test_name.startswith("test_embedding_bag_backward")
        else args[0]
    )
    marker = f"ISOLATED_EMBEDDING_BAG_ENTERING_INVALID_CALL hygon {target}\n"
    _, entered, diagnostics = result.stderr.partition(marker)
    expected_assertions = (
        (
            "device assertion failed: 'embedding_bag_backward received invalid "
            "indices, offsets, or auxiliary data'",
        )
        if target == "_embedding_bag_backward"
        else (
            "device assertion failed: 'embedding_bag: invalid index or offset'",
            "device assertion failed: 'embedding_bag: invalid terminal offset'",
        )
    )
    # DTK aborts on a device assertion. Require its exact target diagnostic after
    # preparation synchronized successfully; the marker is not error evidence.
    asserted_hygon = (
        flag_gems.vendor_name == "hygon"
        and result.returncode == -signal.SIGABRT
        and entered
        and any(message in diagnostics for message in expected_assertions)
    )
    assert result.returncode == 0 or asserted_hygon, output


def _mark_invalid_call(target):
    flag_gems.runtime.torch_device_fn.synchronize()
    if os.environ.get(_ERROR_CHILD) == "1":
        print(
            "ISOLATED_EMBEDDING_BAG_ENTERING_INVALID_CALL",
            flag_gems.vendor_name,
            target,
            file=sys.stderr,
            flush=True,
        )


def _inputs(index_values, offset_values, dtype, index_dtype, dim=17, strided=False):
    generator = torch.Generator().manual_seed(2026)
    weight = torch.randn((8, dim), generator=generator, dtype=torch.float64).to(dtype)
    indices = torch.tensor(index_values, dtype=index_dtype)
    offsets = torch.tensor(offset_values, dtype=index_dtype)
    weight = weight.to(flag_gems.device)
    indices = indices.to(flag_gems.device)
    offsets = offsets.to(flag_gems.device)
    if strided:
        weight = weight.t().contiguous().t()
        indices = torch.stack((indices, indices), dim=1)[:, 0]
        offsets = torch.stack((offsets, offsets), dim=1)[:, 0]
    return weight, indices, offsets


def _oracle(weight, indices, offsets, mode, include_last, padding, psw=None):
    weight = weight.cpu().double()
    rows = indices.cpu().tolist()
    starts = offsets.cpu().tolist()
    num_bags = len(starts) - int(include_last)
    out = torch.zeros((num_bags, weight.shape[1]), dtype=torch.float64)
    sizes = torch.zeros(num_bags, dtype=indices.dtype)
    mapping = torch.zeros(len(rows), dtype=indices.dtype)
    maximum = torch.full(out.shape, 0 if ASCEND_ABI else -1, dtype=indices.dtype)
    sample_weights = psw.cpu().double() if psw is not None else None
    for bag in range(num_bags):
        end = starts[bag + 1] if bag + 1 < len(starts) else len(rows)
        positions = [p for p in range(starts[bag], end) if rows[p] != padding]
        mapping[starts[bag] : end] = bag
        sizes[bag] = len(positions)
        if not positions:
            continue
        row_ids = torch.tensor([rows[p] for p in positions], dtype=torch.int64)
        values = weight[row_ids]
        if mode == 2:
            out[bag], winners = values.max(dim=0)
            maximum[bag] = row_ids[winners].to(indices.dtype)
        else:
            if sample_weights is not None:
                values = values * sample_weights[positions, None]
            out[bag] = values.sum(dim=0)
            if mode == 1:
                out[bag] /= len(positions)
    return out, mapping, sizes, maximum


def _check_forward(
    op, values, starts, include_last, padding, mode, dtype, index_dtype, dim=17
):
    weight, indices, offsets = _inputs(values, starts, dtype, index_dtype, dim)
    expected = _oracle(weight, indices, offsets, mode, include_last, padding)
    actual = op(
        weight, indices, offsets, False, mode, False, None, include_last, padding
    )
    assert len(actual) == 4
    assert actual[0].device == weight.device
    flag_gems.testing.assert_close(
        actual[0].cpu(), expected[0], dtype, reduce_dim=max(1, len(values))
    )
    for tensor in actual[1:]:
        assert tensor.dtype == index_dtype
        assert tensor.device == indices.device
    expected_mapping = (
        expected[1][:0] if ASCEND_ABI and mode == 0 and padding == -1 else expected[1]
    )
    torch.testing.assert_close(actual[1].cpu(), expected_mapping, rtol=0, atol=0)
    expected_sizes = (
        torch.zeros_like(expected[2]) if ASCEND_ABI and mode == 0 else expected[2]
    )
    sizes_per_bag = ASCEND_ABI or (
        flag_gems.vendor_name == "mthreads" and op is _embedding_bag
    )
    assert actual[2].shape == (
        (len(starts) - int(include_last),) if sizes_per_bag else (len(starts),)
    )
    torch.testing.assert_close(
        actual[2][: expected[0].shape[0]].cpu(), expected_sizes, rtol=0, atol=0
    )
    if mode == 2:
        torch.testing.assert_close(actual[3].cpu(), expected[3], rtol=0, atol=0)
    elif ASCEND_ABI:
        torch.testing.assert_close(actual[3].cpu(), expected_sizes, rtol=0, atol=0)
    else:
        assert actual[3].numel() == 0


@pytest.mark.embedding_bag
@pytest.mark.parametrize("values,starts,include_last,padding", CASES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_embedding_bag(values, starts, include_last, padding, index_dtype, dtype, mode):
    _check_forward(
        _embedding_bag, values, starts, include_last, padding, mode, dtype, index_dtype
    )


@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("values,starts,include_last,padding", CASES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_embedding_bag_forward_only(
    values, starts, include_last, padding, index_dtype, dtype, mode
):
    _check_forward(
        _embedding_bag_forward_only,
        values,
        starts,
        include_last,
        padding,
        mode,
        dtype,
        index_dtype,
    )


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dim", [1, 33])
@pytest.mark.parametrize("num_indices", [2, 33])
def test_embedding_bag_single_bag(op, mode, dim, num_indices):
    _check_forward(
        op,
        [1 + i % 7 for i in range(num_indices)],
        [0],
        False,
        0,
        mode,
        torch.float32,
        torch.int64,
        dim,
    )


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("padding", [-1, 1])
@pytest.mark.parametrize("include_last", [False, True])
@pytest.mark.parametrize("dim", [0, 1, 33])
def test_embedding_bag_single_index(op, mode, padding, include_last, dim):
    starts = [0, 0, 1, 1] if include_last else [0, 0, 1]
    weight, indices, offsets = _inputs([1], starts, torch.float32, torch.int64, dim)
    expected = _oracle(weight, indices, offsets, mode, include_last, padding)
    actual = op(
        weight, indices, offsets, False, mode, False, None, include_last, padding
    )
    torch.testing.assert_close(actual[0].cpu().double(), expected[0])
    expected_mapping = (
        expected[1][:0] if ASCEND_ABI and mode == 0 and padding == -1 else expected[1]
    )
    torch.testing.assert_close(actual[1].cpu(), expected_mapping, rtol=0, atol=0)
    sizes = torch.zeros_like(expected[2]) if ASCEND_ABI and mode == 0 else expected[2]
    torch.testing.assert_close(actual[2][:3].cpu(), sizes, rtol=0, atol=0)
    if mode == 2:
        torch.testing.assert_close(actual[3].cpu(), expected[3], rtol=0, atol=0)


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("dim", [1, 33] if QUICK_MODE else [0, 1, 33, 128, 513])
def test_embedding_bag_strided_weighted(op, dtype, weighted, dim):
    weight, indices, offsets = _inputs(
        [1, 2, 1, 0, 5, 6], [0, 3, 3, 6], dtype, torch.int64, dim, True
    )
    psw = (
        torch.tensor([0.5, -1, 2, 0.25, 1.5, -2], dtype=dtype, device=weight.device)
        if weighted
        else None
    )
    expected = _oracle(weight, indices, offsets, 0, True, 0, psw)
    result = op(weight, indices, offsets, False, 0, False, psw, True, 0)
    flag_gems.testing.assert_close(result[0].cpu(), expected[0], dtype, reduce_dim=3)
    torch.testing.assert_close(result[1].cpu(), expected[1], rtol=0, atol=0)


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize(
    "index_dtype,offset_dtype", [(torch.int32, torch.int64), (torch.int64, torch.int32)]
)
def test_embedding_bag_mixed_indices_negative_padding(op, index_dtype, offset_dtype):
    weight, indices, offsets = _inputs([1, 6, 2, 6], [0, 2], torch.float32, index_dtype)
    offsets = offsets.to(offset_dtype)
    out = op(weight, indices, offsets, padding_idx=-2)
    expected = _oracle(
        weight, indices.to(torch.int64), offsets.to(torch.int64), 0, False, 6
    )
    flag_gems.testing.assert_close(out[0].cpu(), expected[0], torch.float32)
    assert all(t.dtype == torch.int64 for t in out[1:])


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("include_last", [False, True])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("padding", [-1, 0])
def test_embedding_bag_native_abi(op, include_last, mode, padding):
    # MUSA 5.2's native forward asserts on int32 helpers without a terminal
    # offset. The independent mathematical cases above still cover int32.
    index_dtype = torch.int64 if flag_gems.vendor_name == "mthreads" else torch.int32
    weight, indices, offsets = _inputs(
        [1, 0, 2, 1],
        [0, 0, 2, 4] if include_last else [0, 0, 2],
        torch.float32,
        index_dtype,
    )
    native = getattr(torch.ops.aten, op.__name__).default
    args = (weight, indices, offsets, False, mode, False, None, include_last, padding)
    expected = native(*args)
    actual = op(*args)
    num_bags = actual[0].shape[0]
    for i, (result, reference) in enumerate(zip(actual, expected)):
        assert result.dtype == reference.dtype
        assert result.shape == reference.shape
        if i == 2:
            result, reference = result[:num_bags], reference[:num_bags]
        torch.testing.assert_close(result.cpu(), reference.cpu())


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("weighted", [False, True])
def test_embedding_bag_nonfinite_and_ties(op, weighted):
    weight = torch.tensor(
        [
            [0, 0, 0],
            [float("nan"), 2, float("-inf")],
            [3, 2, float("-inf")],
            [4, float("nan"), 5],
        ],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    indices = torch.tensor([1, 2, 3, 0], dtype=torch.int64, device=weight.device)
    offsets = torch.tensor([0, 0, 3], dtype=torch.int64, device=weight.device)
    psw = (
        torch.tensor([1, 1, 1, float("inf")], device=weight.device)
        if weighted
        else None
    )
    mode = 0 if weighted else 2
    args = (weight, indices, offsets, False, mode, False, psw, False, 0)
    expected = getattr(torch.ops.aten, op.__name__).default(*args)
    actual = op(*args)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result.cpu(), reference.cpu(), equal_nan=True)


def _backward_oracle(
    grad,
    indices,
    mapping,
    sizes,
    maximum,
    num_weights,
    mode,
    frequency,
    sparse,
    padding,
    psw=None,
):
    dtype = grad.dtype
    grad = grad.cpu().double()
    rows = indices.cpu().tolist()
    mapping = mapping.cpu().tolist()
    sizes = sizes.cpu().tolist()
    maximum = maximum.cpu()
    out = torch.zeros((num_weights, grad.shape[1]), dtype=torch.float64)
    if mode == 2 and not sparse:
        for bag in range(grad.shape[0]):
            if sizes[bag] == 0:
                continue
            for feature in range(grad.shape[1]):
                row = int(maximum[bag, feature])
                if row >= 0 and row != padding:
                    out[row, feature] += grad[bag, feature]
        return out
    sample_weights = psw.cpu().double() if psw is not None else None
    sparse_rows, sparse_values = [], []
    for position, row in enumerate(rows):
        if row == padding:
            continue
        bag = mapping[position]
        scale = 1.0
        if mode == 1:
            if sparse:
                # ATen materializes the reciprocal in grad dtype before the
                # per-occurrence multiplication, including FP16/BF16 rounding.
                size = torch.tensor(sizes[bag], dtype=dtype)
                scale = float(size.reciprocal())
            else:
                scale /= sizes[bag]
        if frequency:
            scale /= rows.count(row)
        if sample_weights is not None:
            scale *= sample_weights[position]
        contribution = grad[bag] * scale
        if sparse:
            sparse_rows.append(row)
            sparse_values.append(contribution.to(dtype))
        else:
            out[row] += contribution
    if sparse:
        values = (
            torch.stack(sparse_values)
            if sparse_values
            else torch.empty((0, grad.shape[1]), dtype=dtype)
        )
        return torch.sparse_coo_tensor(
            torch.tensor([sparse_rows], dtype=torch.int64), values, out.shape
        )
    return out


def _check_backward_result(result, expected, dtype, reduce_dim):
    if result.is_sparse:
        # COO values are stored in the requested dtype before any duplicate
        # reduction. A cast-once dense oracle would test different arithmetic.
        torch.testing.assert_close(
            result._indices().cpu(), expected._indices(), rtol=0, atol=0
        )
        flag_gems.testing.assert_close(
            result._values().cpu(), expected._values(), dtype
        )
        reference = expected.to_dense()
        actual = result.cpu().to_dense()
    else:
        reference, actual = expected, result.cpu()
    flag_gems.testing.assert_close(actual, reference, dtype, reduce_dim=reduce_dim)


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize(
    "sparse,frequency", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("include_last", [False, True])
def test_embedding_bag_backward(
    dtype, index_dtype, mode, sparse, frequency, include_last
):
    weight, indices, offsets = _inputs(
        [1, 2, 1, 0, 2, 2, 5],
        [0, 0, 3, 3, 7] if include_last else [0, 0, 3, 3],
        dtype,
        index_dtype,
        33,
    )
    output, mapping, sizes, maximum = _embedding_bag(
        weight, indices, offsets, False, mode, sparse, None, include_last, 0
    )
    grad = torch.randn(
        (output.shape[1], output.shape[0]), dtype=dtype, device=weight.device
    ).t()
    result = _embedding_bag_backward(
        grad,
        indices,
        offsets,
        mapping,
        sizes,
        maximum,
        8,
        frequency,
        mode,
        sparse,
        None,
        0,
    )
    expected = _backward_oracle(
        grad, indices, mapping, sizes, maximum, 8, mode, frequency, sparse, 0
    )
    assert result.shape == (8, 33)
    assert result.dtype == dtype
    assert result.device == weight.device
    if sparse:
        assert result.layout == torch.sparse_coo
    else:
        assert result.layout == torch.strided
    _check_backward_result(result, expected, dtype, 7)


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sparse", [False, True])
def test_embedding_bag_backward_weighted_empty_mapping(dtype, sparse):
    weight, indices, offsets = _inputs([1, 2, 1, 0, 5], [0, 0, 3], dtype, torch.int64)
    psw = torch.tensor([0.5, -1, 2, 0, 1.5], dtype=dtype, device=weight.device)
    output, mapping, sizes, maximum = _embedding_bag(
        weight, indices, offsets, False, 0, sparse, psw, False, 0
    )
    grad = torch.ones_like(output)
    result = _embedding_bag_backward(
        grad, indices, offsets, mapping[:0], sizes, maximum, 8, False, 0, sparse, psw, 0
    )
    expected = _backward_oracle(
        grad, indices, mapping, sizes, maximum, 8, 0, False, sparse, 0, psw
    )
    _check_backward_result(result, expected, dtype, 3)


@pytest.mark.skipif(
    ASCEND_ABI,
    reason="Native NPU backward falls back to CPU; the mathematical oracle covers this backend",
)
@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "mode,sparse,frequency",
    [
        (0, False, True),
        (1, False, True),
        (2, False, True),
        (0, True, False),
        (1, True, False),
        (2, True, False),
    ],
)
def test_embedding_bag_backward_native(dtype, mode, sparse, frequency):
    weight, indices, offsets = _inputs(
        [1, 2, 1, 0, 2, 3], [0, 0, 3], dtype, torch.int64, 17
    )
    # CUDA's native MAX backward has no BF16 dispatch. Float32 is an independent
    # native reference for this case; BF16 storage is checked by the math tests.
    native_weight = (
        weight.float()
        if dtype == torch.bfloat16 and mode == 2 and not sparse
        else weight
    )
    native_forward = torch.ops.aten._embedding_bag.default(
        native_weight, indices, offsets, frequency, mode, sparse, None, False, 0
    )
    grad = torch.randn_like(native_forward[0])
    args = (
        grad,
        indices,
        offsets,
        *native_forward[1:],
        8,
        frequency,
        mode,
        sparse,
        None,
        0,
    )
    expected = torch.ops.aten._embedding_bag_backward.default(*args)
    actual = _embedding_bag_backward(*args)
    _check_backward_result(actual, expected.cpu(), grad.dtype, 6)


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("sparse", [False, True])
def test_embedding_bag_backward_zero_dimension(mode, sparse):
    weight, indices, offsets = _inputs(
        [1, 2, 1], [0, 0, 2], torch.float32, torch.int64, 0
    )
    out, mapping, sizes, maximum = _embedding_bag(weight, indices, offsets, mode=mode)
    result = _embedding_bag_backward(
        out, indices, offsets, mapping, sizes, maximum, 8, False, mode, sparse
    )
    assert result.shape == (8, 0)
    assert result.numel() == 0
    if sparse:
        torch.testing.assert_close(
            result._indices().cpu(), indices.cpu().reshape(1, -1)
        )


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("num_indices", [255, 256, 257, 1025])
@pytest.mark.parametrize("mode", [0, 1])
def test_embedding_bag_backward_sparse_chunk_boundaries(num_indices, mode):
    rows = torch.arange(num_indices, dtype=torch.int64) % 8
    split = num_indices // 2
    weight, indices, offsets = _inputs(
        rows.tolist(), [0, split, num_indices], torch.float32, torch.int64, 9
    )
    output, mapping, sizes, maximum = _embedding_bag(
        weight, indices, offsets, mode=mode, include_last_offset=True, padding_idx=0
    )
    grad_cpu = torch.arange(18, dtype=torch.float32).reshape(2, 9) / 10
    actual = _embedding_bag_backward(
        grad_cpu.to(output.device),
        indices,
        offsets,
        mapping,
        sizes,
        maximum,
        8,
        False,
        mode,
        True,
        None,
        0,
    )
    keep = rows != 0
    bags = (torch.arange(num_indices) >= split).to(torch.int64)
    expected_values = grad_cpu[bags[keep]]
    if mode == 1:
        counts = torch.tensor([keep[:split].sum(), keep[split:].sum()])
        reciprocal = counts.to(expected_values.dtype).reciprocal()
        expected_values *= reciprocal[bags[keep], None]
    assert actual.is_sparse and actual.shape == (8, 9)
    torch.testing.assert_close(actual._indices().cpu(), rows[keep].reshape(1, -1))
    torch.testing.assert_close(actual._values().cpu(), expected_values)


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize(
    "dtype,positive,negative,contribution",
    [
        (torch.float16, 4.00390625, -2.66796875, 1.333984375),
        pytest.param(
            torch.bfloat16,
            1.015625,
            -0.6796875,
            0.33984375,
            marks=pytest.mark.skipif(
                not utils.bf16_is_supported, reason="Device does not support bfloat16"
            ),
        ),
    ],
)
def test_embedding_bag_backward_sparse_mean_rounding(
    dtype, positive, negative, contribution
):
    weight, indices, offsets = _inputs(
        [1, 2, 1, 0, 2, 3], [0, 0, 3], dtype, torch.int64, 1
    )
    _, mapping, sizes, maximum = _embedding_bag(
        weight, indices, offsets, mode=1, padding_idx=0
    )
    grad = torch.tensor(
        [[0], [positive], [negative]], dtype=dtype, device=weight.device
    )
    result = _embedding_bag_backward(
        grad, indices, offsets, mapping, sizes, maximum, 8, False, 1, True, None, 0
    ).cpu()
    expected_values = torch.tensor(
        [[contribution]] * 3 + [[-contribution]] * 2, dtype=dtype
    )
    torch.testing.assert_close(
        result._indices(), torch.tensor([[1, 2, 1, 2, 3]]), rtol=0, atol=0
    )
    torch.testing.assert_close(result._values(), expected_values, rtol=0, atol=0)
    # The two contributions to row 2 cancel only with ATen's intermediate
    # reciprocal rounding; a single final cast leaves a nonzero residual.
    torch.testing.assert_close(result.to_dense()[2], torch.zeros(1, dtype=dtype))


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("values,starts,include_last,padding", CASES[2:])
def test_embedding_bag_backward_empty(
    mode, sparse, values, starts, include_last, padding
):
    weight, indices, offsets = _inputs(values, starts, torch.float32, torch.int64)
    out, mapping, sizes, maximum = _embedding_bag(
        weight, indices, offsets, False, mode, sparse, None, include_last, padding
    )
    result = _embedding_bag_backward(
        torch.ones_like(out),
        indices,
        offsets,
        mapping,
        sizes,
        maximum,
        8,
        False,
        mode,
        sparse,
        None,
        padding,
    )
    result = result.cpu().to_dense() if sparse else result.cpu()
    torch.testing.assert_close(result, torch.zeros((8, 17)), rtol=0, atol=0)


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dim", [0, 7])
@pytest.mark.parametrize("num_bags", [0, 3])
def test_embedding_bag_empty_table(op, mode, dim, num_bags):
    weight = torch.empty((0, dim), device=flag_gems.device)
    indices = torch.empty(0, dtype=torch.int64, device=weight.device)
    offsets = torch.zeros(num_bags, dtype=torch.int64, device=weight.device)
    output, mapping, sizes, maximum = op(weight, indices, offsets, mode=mode)
    torch.testing.assert_close(output.cpu(), torch.zeros((num_bags, dim)))
    assert mapping.numel() == 0
    torch.testing.assert_close(sizes.cpu(), torch.zeros(num_bags, dtype=sizes.dtype))
    if mode == 2:
        torch.testing.assert_close(
            maximum.cpu(),
            torch.full((num_bags, dim), 0 if ASCEND_ABI else -1, dtype=maximum.dtype),
        )
    for sparse in (False, True):
        result = _embedding_bag_backward(
            torch.ones_like(output),
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            0,
            False,
            mode,
            sparse,
        )
        assert result.shape == (0, dim)
        if sparse:
            assert result.is_sparse
            assert result._nnz() == 0
        else:
            assert result.numel() == 0


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize(
    "values,starts,last",
    [
        ([1], [1], False),
        ([1, 2], [0, 2, 1], False),
        ([1], [0, 2], False),
        ([1], [0, -1], False),
        ([8], [0], False),
        ([-2], [0], False),
        ([1], [0, 0], True),
        ([1], [], False),
    ],
)
def test_embedding_bag_invalid_values(op, values, starts, last):
    if (
        flag_gems.vendor_name in ("nvidia", "hygon", "mthreads", "ascend", "iluvatar")
        and os.environ.get(_ERROR_CHILD) != "1"
    ):
        _run_isolated_error(
            "test_embedding_bag_invalid_values", [op.__name__, values, starts, last]
        )
        return
    weight, indices, offsets = _inputs(values, starts, torch.float32, torch.int64)
    _mark_invalid_call(op.__name__)
    with pytest.raises((ValueError, RuntimeError)):
        op(weight, indices, offsets, include_last_offset=last)
        flag_gems.runtime.torch_device_fn.synchronize()


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("mode", [1, 2])
def test_embedding_bag_invalid_after_warmup(op, mode):
    if os.environ.get(_ERROR_CHILD) != "1":
        _run_isolated_error(
            "test_embedding_bag_invalid_after_warmup", [op.__name__, mode]
        )
        return
    weight, indices, offsets = _inputs([1, 2, 1], [0, 2], torch.float32, torch.int64)
    op(weight, indices, offsets, mode=0)
    flag_gems.runtime.torch_device_fn.synchronize()
    # Loading a different valid specialization must not disable later errors.
    indices[0] = weight.shape[0]
    _mark_invalid_call(op.__name__)
    with pytest.raises(RuntimeError):
        op(weight, indices, offsets, mode=mode)
        flag_gems.runtime.torch_device_fn.synchronize()


@pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar",
    reason="CoreX assertion regression after loading mixed dtype kernels",
)
@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize(
    "target", ["forward", "forward-only", "backward", "backward-first"]
)
def test_embedding_bag_iluvatar_error_after_mixed_dtypes(target):
    if os.environ.get(_ERROR_CHILD) != "1":
        _run_isolated_error(
            "test_embedding_bag_iluvatar_error_after_mixed_dtypes", [target]
        )
        return
    device = flag_gems.device
    indices = torch.tensor([1, 1, 2], device=device)
    offsets = torch.tensor([0, 2], device=device)
    mapping = torch.tensor([0, 0, 1], device=device)
    sizes = torch.tensor([2, 1], device=device)
    maximum = torch.empty(0, dtype=torch.int64, device=device)
    weight = torch.ones((4, 33), device=device)
    if target != "backward-first":
        _embedding_bag(weight, indices, offsets)
    # Loading new dtype, width and frequency specializations must not
    # disconnect the SDK's device assertion state from subsequent errors.
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for dim in (1, 33):
            grad = torch.ones((2, dim), dtype=dtype, device=device)
            result = _embedding_bag_backward(
                grad, indices, offsets, mapping, sizes, maximum, 4, True, 0, False
            )
            expected = torch.zeros((4, dim), dtype=dtype)
            expected[1:3] = 1
            torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    indices[0] = weight.shape[0]
    flag_gems.runtime.torch_device_fn.synchronize()
    with pytest.raises(RuntimeError, match="(?i)assert"):
        if target == "forward":
            _embedding_bag(weight, indices, offsets, mode=1)
        elif target == "forward-only":
            _embedding_bag_forward_only(weight, indices, offsets, mode=2)
        else:
            _embedding_bag_backward(
                grad, indices, offsets, mapping, sizes, maximum, 4, True, 0, False
            )
        flag_gems.runtime.torch_device_fn.synchronize()


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize(
    "error",
    [
        "weight-rank",
        "indices-rank",
        "offsets-rank",
        "indices-dtype",
        "mode",
        "padding",
        "weights-mode",
        "weights-size",
        "weights-dtype",
    ],
)
def test_embedding_bag_invalid_metadata(op, error):
    weight, indices, offsets = _inputs([1, 2], [0], torch.float32, torch.int64)
    kwargs = {}
    if error == "weight-rank":
        weight = weight.flatten()
    elif error == "indices-rank":
        indices = indices.unsqueeze(0)
    elif error == "offsets-rank":
        offsets = offsets.unsqueeze(0)
    elif error == "indices-dtype":
        indices = indices.float()
    elif error == "mode":
        kwargs["mode"] = 3
    elif error == "padding":
        kwargs["padding_idx"] = 8
    elif error == "weights-mode":
        kwargs.update(mode=1, per_sample_weights=torch.ones(2, device=weight.device))
    elif error == "weights-size":
        kwargs["per_sample_weights"] = torch.ones(1, device=weight.device)
    elif error == "weights-dtype":
        kwargs["per_sample_weights"] = torch.ones(
            2, dtype=torch.float16, device=weight.device
        )
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        op(weight, indices, offsets, **kwargs)


@pytest.mark.embedding_bag_backward
def test_embedding_bag_backward_sparse_frequency_error():
    weight, indices, offsets = _inputs([1, 2], [0], torch.float32, torch.int64)
    output, mapping, sizes, maximum = _embedding_bag(weight, indices, offsets)
    with pytest.raises((ValueError, RuntimeError), match="freq"):
        _embedding_bag_backward(
            torch.ones_like(output),
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            8,
            True,
            0,
            True,
        )


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize(
    "error",
    [
        "index",
        "offset",
        "mapping",
        "size",
        "maximum",
        "maximum-shape",
        "weights-mode",
        "weights-dtype",
    ],
)
def test_embedding_bag_backward_invalid(error):
    if (
        flag_gems.vendor_name in ("nvidia", "hygon", "iluvatar", "mthreads")
        and os.environ.get(_ERROR_CHILD) != "1"
    ):
        _run_isolated_error("test_embedding_bag_backward_invalid", [error])
        return
    mode = 2 if error.startswith("maximum") else 1
    weight, indices, offsets = _inputs([1, 2, 1], [0, 2], torch.float32, torch.int64)
    out, mapping, sizes, maximum = _embedding_bag(weight, indices, offsets, mode=mode)
    flag_gems.runtime.torch_device_fn.synchronize()
    psw = None
    if error == "index":
        indices[0] = 8
    elif error == "offset":
        offsets[0] = 1
    elif error == "mapping":
        mapping[0] = 2
    elif error == "size":
        sizes[0] = 0
    elif error == "maximum":
        maximum[0, 0] = 8
    elif error == "maximum-shape":
        maximum = maximum[:, :1]
    elif error == "weights-mode":
        psw = torch.ones_like(indices, dtype=weight.dtype)
    elif error == "weights-dtype":
        mode = 0
        psw = torch.ones_like(indices, dtype=torch.float16)
    grad = torch.ones_like(out)
    _mark_invalid_call("_embedding_bag_backward")
    with pytest.raises(RuntimeError):
        _embedding_bag_backward(
            grad,
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            8,
            False,
            mode,
            False,
            psw,
        )
        flag_gems.runtime.torch_device_fn.synchronize()


@pytest.mark.embedding_bag_backward
def test_embedding_bag_backward_deterministic():
    weight, indices, offsets = _inputs([1, 2, 1, 2], [0, 2], torch.float32, torch.int64)
    output, mapping, sizes, maximum = _embedding_bag(weight, indices, offsets)
    args = (
        torch.ones_like(output),
        indices,
        offsets,
        mapping,
        sizes,
        maximum,
        8,
        False,
        0,
        False,
    )
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        try:
            first = _embedding_bag_backward(*args)
        except RuntimeError as error:
            assert "determin" in str(error).lower()
        else:
            second = _embedding_bag_backward(*args)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


@pytest.mark.embedding_bag_backward
@pytest.mark.skipif(
    not _LEGACY_ASCEND,
    reason="Regression for legacy Ascend masked access to empty buffers",
)
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize(
    "frequency,sparse", [(False, False), (True, False), (False, True)]
)
def test_embedding_bag_backward_legacy_ascend_empty_table_invalid(
    index_dtype, frequency, sparse
):
    if os.environ.get(_ERROR_CHILD) != "1":
        _run_isolated_error(
            "test_embedding_bag_backward_legacy_ascend_empty_table_invalid",
            [index_dtype, frequency, sparse],
        )
        return
    dtype = getattr(torch, index_dtype)
    grad = torch.ones((1, 3), dtype=torch.float32, device=flag_gems.device)
    indices = torch.tensor([0], dtype=dtype, device=grad.device)
    offsets = torch.tensor([0], dtype=dtype, device=grad.device)
    mapping = torch.tensor([0], dtype=dtype, device=grad.device)
    sizes = torch.tensor([1], dtype=dtype, device=grad.device)
    maximum = torch.empty(0, dtype=dtype, device=grad.device)
    _mark_invalid_call("_embedding_bag_backward")
    with pytest.raises(
        RuntimeError, match=r"embedding_bag index is outside \[0, num_weights\)"
    ):
        _embedding_bag_backward(
            grad,
            indices,
            offsets,
            mapping,
            sizes,
            maximum,
            0,
            frequency,
            0,
            sparse,
        )
        flag_gems.runtime.torch_device_fn.synchronize()


@pytest.mark.embedding_bag_backward
@pytest.mark.skipif(
    not _LEGACY_ASCEND,
    reason="Regression for legacy Ascend zero-width sparse index compaction",
)
@pytest.mark.parametrize("num_indices", [255, 256, 257, 1025])
@pytest.mark.parametrize("index_dtype", INDEX_DTYPES)
def test_embedding_bag_backward_legacy_ascend_sparse_zero_dim_prefix(
    num_indices, index_dtype
):
    rows = torch.arange(num_indices, dtype=index_dtype) % 8
    split = num_indices // 3
    weight, indices, offsets = _inputs(
        rows.tolist(), [0, split, split, num_indices], torch.float32, index_dtype, 0
    )
    output, mapping, sizes, maximum = _embedding_bag(
        weight, indices, offsets, include_last_offset=True, padding_idx=0
    )
    actual = _embedding_bag_backward(
        output, indices, offsets, mapping, sizes, maximum, 8, False, 0, True, None, 0
    )
    expected = rows[rows != 0].to(torch.int64).reshape(1, -1)
    assert actual.is_sparse and actual.shape == (8, 0)
    assert actual._nnz() == expected.numel()
    assert actual._values().shape == (expected.numel(), 0)
    torch.testing.assert_close(actual._indices().cpu(), expected, rtol=0, atol=0)
