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

import os

import pytest
import torch

import flag_gems
from flag_gems import (
    _embedding_bag,
    _embedding_bag_backward,
    _embedding_bag_forward_only,
)

from . import accuracy_utils as utils
from .test_embedding_bag import (
    _ERROR_CHILD,
    _LEGACY_ASCEND,
    ASCEND_ABI,
    CASES,
    DTYPES,
    INDEX_DTYPES,
    _check_iluvatar_error_after_mixed_dtypes,
    _inputs,
    _mark_invalid_call,
    _run_isolated_error,
)


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


@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("op", [_embedding_bag, _embedding_bag_forward_only])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dim", [0, 7])
@pytest.mark.parametrize("num_bags", [0, 3])
def test_embedding_bag_backward_empty_table(op, mode, dim, num_bags):
    weight = torch.empty((0, dim), device=flag_gems.device)
    indices = torch.empty(0, dtype=torch.int64, device=weight.device)
    offsets = torch.zeros(num_bags, dtype=torch.int64, device=weight.device)
    output, mapping, sizes, maximum = op(weight, indices, offsets, mode=mode)
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


@pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar",
    reason="CoreX assertion regression after loading mixed dtype kernels",
)
@pytest.mark.embedding_bag_backward
@pytest.mark.parametrize("target", ["backward", "backward-first"])
def test_embedding_bag_backward_iluvatar_error_after_mixed_dtypes(target):
    if os.environ.get(_ERROR_CHILD) != "1":
        _run_isolated_error(
            "test_embedding_bag_backward_iluvatar_error_after_mixed_dtypes", [target]
        )
        return
    _check_iluvatar_error_after_mixed_dtypes(target)
