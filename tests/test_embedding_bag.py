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
import importlib
name, args = json.loads(sys.argv[1])
module = "test_embedding_bag_backward" if name.startswith("test_embedding_bag_backward") else "test_embedding_bag"
suite = importlib.import_module(f"tests.{module}")
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


@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
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


def _check_iluvatar_error_after_mixed_dtypes(target):
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


@pytest.mark.skipif(
    flag_gems.vendor_name != "iluvatar",
    reason="CoreX assertion regression after loading mixed dtype kernels",
)
@pytest.mark.embedding_bag
@pytest.mark.embedding_bag_forward_only
@pytest.mark.parametrize("target", ["forward", "forward-only"])
def test_embedding_bag_iluvatar_error_after_mixed_dtypes(target):
    if os.environ.get(_ERROR_CHILD) != "1":
        _run_isolated_error(
            "test_embedding_bag_iluvatar_error_after_mixed_dtypes", [target]
        )
        return
    _check_iluvatar_error_after_mixed_dtypes(target)


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
