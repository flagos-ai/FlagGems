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
from . import conftest as cfg

if cfg.QUICK_MODE:
    CASES = [((), "active"), ((3, 17), "mixed")]
    DTYPES = [torch.float32]
else:
    CASES = [
        ((), "active"),
        ((), "sentinel"),
        ((1,), "active"),
        ((7,), "middle"),
        ((3, 17), "mixed"),
        ((2, 257), "mixed"),
    ]
    DTYPES = utils.ALL_FLOAT_DTYPES


def _make_target(shape, pattern, device):
    if len(shape) == 0:
        value = -1 if pattern == "sentinel" else 0
        return torch.tensor(value, dtype=torch.int64, device=device)

    n_rows = shape[0] if len(shape) == 2 else 1
    n_classes = shape[-1]
    rows = torch.empty((n_rows, n_classes), dtype=torch.int64)
    for row in range(n_rows):
        rows[row] = (torch.arange(n_classes, dtype=torch.int64) * 3 + row) % n_classes

        row_pattern = pattern
        if pattern == "mixed":
            row_pattern = ("first", "middle", "none")[row % 3]
        if row_pattern == "first":
            rows[row, 0] = -1
        elif row_pattern == "middle":
            stop = max(1, n_classes // 2)
            if stop > 1:
                rows[row, 1] = rows[row, 0]
            rows[row, stop] = -1
        elif row_pattern == "sentinel":
            rows[row, 0] = -1
        elif row_pattern not in ("active", "none"):
            raise ValueError(f"unknown target pattern: {row_pattern}")

    target = rows.reshape(shape)
    return target.to(device)


def _reference(input, target, reduction):
    input_cpu = input.detach().cpu()
    target_cpu = target.detach().cpu()
    original_shape = input_cpu.shape
    n_rows = original_shape[0] if input_cpu.ndim == 2 else 1
    n_classes = original_shape[-1] if input_cpu.ndim != 0 else 1
    acc_dtype = torch.float64 if input_cpu.dtype == torch.float64 else torch.float32
    input_rows = input_cpu.reshape(n_rows, n_classes).to(acc_dtype)
    target_rows = target_cpu.reshape(n_rows, n_classes)
    mask = torch.zeros((n_rows, n_classes), dtype=input_cpu.dtype)
    row_losses = torch.zeros((n_rows,), dtype=acc_dtype)

    for row in range(n_rows):
        active_ids = []
        for target_id in target_rows[row].tolist():
            if target_id == -1:
                break
            active_ids.append(target_id)
            mask[row, target_id] = 1

        non_target = mask[row] == 0
        for target_id in active_ids:
            margins = 1.0 - input_rows[row, target_id] + input_rows[row]
            contributions = torch.where(margins > 0, margins, 0.0)
            row_losses[row] += contributions[non_target].sum()
        row_losses[row] /= n_classes

    if input_cpu.ndim <= 1:
        loss = row_losses[0].to(input_cpu.dtype)
    elif reduction == 0:
        loss = row_losses.to(input_cpu.dtype)
    elif reduction == 1:
        loss = row_losses.mean().to(input_cpu.dtype)
    else:
        loss = row_losses.sum().to(input_cpu.dtype)

    mask = mask.reshape(original_shape)
    if not utils.TO_CPU:
        loss = loss.to(input.device)
        mask = mask.to(input.device)
    return loss, mask


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("shape,pattern", CASES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multilabel_margin_loss_forward(shape, pattern, dtype, reduction):
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = _make_target(shape, pattern, flag_gems.device)
    ref_loss, ref_mask = _reference(input, target, reduction)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, reduction)

    n_classes = shape[-1] if shape else 1
    n_rows = shape[0] if len(shape) == 2 else 1
    utils.gems_assert_close(
        loss,
        ref_loss,
        dtype,
        equal_nan=True,
        reduce_dim=max(1, n_rows * n_classes),
    )
    utils.gems_assert_equal(is_target, ref_mask)
    assert is_target.shape == input.shape
    assert is_target.dtype == input.dtype


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_multilabel_margin_loss_forward_empty_batch(reduction):
    input = torch.empty((0, 11), dtype=torch.float32, device=flag_gems.device)
    target = torch.empty((0, 11), dtype=torch.int64, device=flag_gems.device)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, reduction)

    assert is_target.shape == input.shape
    assert is_target.dtype == input.dtype
    if reduction == 0:
        assert loss.shape == (0,)
    elif reduction == 2:
        assert loss.shape == ()
        assert loss.item() == 0.0
    else:
        assert loss.shape == ()
        assert torch.isnan(loss).item()


@pytest.mark.multilabel_margin_loss_forward
def test_multilabel_margin_loss_forward_noncontiguous():
    shape = (3, 17)
    input = torch.randn(
        (shape[1], shape[0]), dtype=torch.float32, device=flag_gems.device
    ).transpose(0, 1)
    target_values = _make_target(shape, "mixed", flag_gems.device)
    target = torch.empty(
        (shape[1], shape[0]), dtype=torch.int64, device=flag_gems.device
    ).transpose(0, 1)
    target.copy_(target_values)
    assert not input.is_contiguous()
    assert not target.is_contiguous()
    ref_loss, ref_mask = _reference(input, target, 0)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, 0)

    utils.gems_assert_close(loss, ref_loss, torch.float32, reduce_dim=shape[1])
    utils.gems_assert_equal(is_target, ref_mask)


@pytest.mark.multilabel_margin_loss_forward
def test_multilabel_margin_loss_forward_nan_and_inf():
    input = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.5, -0.5],
        dtype=torch.float32,
        device=flag_gems.device,
    )
    target = torch.tensor([0, 1, -1, 2, 3], dtype=torch.int64, device=flag_gems.device)
    ref_loss, ref_mask = _reference(input, target, 0)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, 0)

    utils.gems_assert_close(
        loss, ref_loss, torch.float32, equal_nan=True, reduce_dim=input.numel()
    )
    utils.gems_assert_equal(is_target, ref_mask)


@pytest.mark.multilabel_margin_loss_forward
def test_multilabel_margin_loss_forward_large_class_watchdog_path():
    shape = (1, 2049)
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    target = torch.arange(shape[1], dtype=torch.int64, device=flag_gems.device).reshape(
        shape
    )
    target[:, 1:128] = 0
    target[:, 128] = -1
    ref_loss, ref_mask = _reference(input, target, 2)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, 2)

    utils.gems_assert_close(loss, ref_loss, torch.float32, reduce_dim=shape[1])
    utils.gems_assert_equal(is_target, ref_mask)


@pytest.mark.multilabel_margin_loss_forward
def test_multilabel_margin_loss_forward_large_class_full_target():
    shape = (2, 257)
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    target = torch.arange(shape[1], dtype=torch.int64, device=flag_gems.device).expand(
        shape
    )

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, 2)

    assert loss.shape == ()
    assert loss.item() == 0.0
    utils.gems_assert_equal(is_target, utils.to_reference(torch.ones_like(input)))


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("shape", [(2, 2048), (1, 4096)])
@pytest.mark.parametrize("reduction", [0, 1])
def test_multilabel_margin_loss_forward_large_class_duplicate(shape, reduction):
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    n_classes = shape[1]
    target = (
        torch.arange(n_classes, dtype=torch.int64, device=flag_gems.device)
        .expand(shape)
        .clone()
    )
    stop = n_classes // 4
    target[:, :stop] %= max(1, stop // 4)
    target[:, stop] = -1
    ref_loss, ref_mask = _reference(input, target, reduction)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, reduction)

    utils.gems_assert_close(
        loss,
        ref_loss,
        torch.float32,
        reduce_dim=shape[0] * shape[1],
    )
    utils.gems_assert_equal(is_target, ref_mask)


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("reduction", [0, 1])
def test_multilabel_margin_loss_forward_large_class_duplicate_tail(dtype, reduction):
    shape = (1, 2049)
    input = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = (
        torch.arange(shape[1], dtype=torch.int64, device=flag_gems.device)
        .expand(shape)
        .clone()
    )
    stop = 257
    target[:, :stop] %= 64
    target[:, stop] = -1
    ref_loss, ref_mask = _reference(input, target, reduction)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, reduction)

    utils.gems_assert_close(
        loss,
        ref_loss,
        dtype,
        reduce_dim=shape[0] * shape[1],
    )
    utils.gems_assert_equal(is_target, ref_mask)


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("reduction", [0, 1])
def test_multilabel_margin_loss_forward_large_class_empty_prefix(reduction):
    shape = (1, 257)
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    target = torch.arange(shape[1], dtype=torch.int64, device=flag_gems.device).reshape(
        shape
    )
    target[:, 0] = -1

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, reduction)

    utils.gems_assert_equal(loss, utils.to_reference(torch.zeros_like(loss)))
    utils.gems_assert_equal(is_target, utils.to_reference(torch.zeros_like(input)))


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("reduction", [0, 1])
def test_multilabel_margin_loss_forward_large_class_bucket_boundaries(reduction):
    shape = (3, 257)
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    target = (
        torch.arange(shape[1], dtype=torch.int64, device=flag_gems.device)
        .expand(shape)
        .clone()
    )
    target[0, 0] = -1
    target[1, 9] = -1
    target[2, 17] = -1
    ref_loss, ref_mask = _reference(input, target, reduction)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, reduction)

    utils.gems_assert_close(
        loss,
        ref_loss,
        torch.float32,
        reduce_dim=shape[0] * shape[1],
    )
    utils.gems_assert_equal(is_target, ref_mask)


@pytest.mark.multilabel_margin_loss_forward
def test_multilabel_margin_loss_forward_many_rows_target_length_reduce():
    shape = (65, 257)
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    target = (
        torch.arange(shape[1], dtype=torch.int64, device=flag_gems.device)
        .expand(shape)
        .clone()
    )
    for row in range(shape[0]):
        target[row, row % 18] = -1
    ref_loss, ref_mask = _reference(input, target, 0)

    loss, is_target = flag_gems.multilabel_margin_loss_forward(input, target, 0)

    utils.gems_assert_close(
        loss,
        ref_loss,
        torch.float32,
        reduce_dim=shape[0] * shape[1],
    )
    utils.gems_assert_equal(is_target, ref_mask)


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize(
    "target_values",
    [
        [-2, 0, 1, 2],
        [4, 0, 1, 2],
        [-1, 4, 0, 1],
        [-(2**63), 0, 1, 2],
        [2**63 - 1, 0, 1, 2],
    ],
)
def test_multilabel_margin_loss_forward_invalid_target_values_raise(target_values):
    input = torch.randn((4,), dtype=torch.float32, device=flag_gems.device)
    target = torch.tensor(target_values, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError, match="target values"):
        flag_gems.multilabel_margin_loss_forward(input, target, 1)


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("invalid_position", [256, 2048])
def test_multilabel_margin_loss_forward_invalid_after_distant_sentinel_raises(
    invalid_position,
):
    shape = (1, 2049)
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    target = torch.arange(shape[1], dtype=torch.int64, device=flag_gems.device).reshape(
        shape
    )
    target[:, 0] = -1
    target[:, invalid_position] = -2

    with pytest.raises(RuntimeError, match="target values"):
        flag_gems.multilabel_margin_loss_forward(input, target, 1)


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize(
    "input_shape,target_shape,target_dtype",
    [
        ((3,), (1,), torch.int64),
        ((), (1,), torch.int64),
        ((2, 3), (2, 3), torch.int32),
    ],
)
def test_multilabel_margin_loss_forward_invalid_target_metadata_raises(
    input_shape, target_shape, target_dtype
):
    input = torch.randn(input_shape, dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros(target_shape, dtype=target_dtype, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.multilabel_margin_loss_forward(input, target, 1)


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 0)])
def test_multilabel_margin_loss_forward_invalid_input_shape_raises(shape):
    input = torch.randn(shape, dtype=torch.float32, device=flag_gems.device)
    target = torch.zeros(shape, dtype=torch.int64, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.multilabel_margin_loss_forward(input, target, 1)
