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

from . import base
from .generated_operator_utils import OperatorBenchmark

# Same geometries as the core shape file for this operator: (N, C) pairs.
_DEFAULT_SHAPES = [
    (8, 128),
    (32, 256),
    (16, 512),
    (8, 1024),
    (2, 2048),
    (1, 4096),
]
_REDUCTIONS = [0, 1, 2]

_DTYPES = [torch.float32, torch.float16]
if flag_gems.runtime.device.support_bf16:
    _DTYPES.append(torch.bfloat16)
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)


def _descriptor_dims(shape):
    # (N, C) extents of one descriptor; a 0-dim descriptor means one sample.
    if isinstance(shape, (list, tuple)):
        dims = shape
    elif isinstance(shape, dict):
        dims = shape.get("input")
    else:
        raise ValueError(
            f"descriptor must be a list of extents or a dict with 'input', got {shape!r}"
        )
    if not isinstance(dims, (list, tuple)):
        raise ValueError(f"descriptor must be a list of extents, got {dims!r}")
    # Extents are used as given: coercing them would silently change a requested
    # workload, so anything that is not an actual int extent is rejected.
    if any(isinstance(extent, bool) or not isinstance(extent, int) for extent in dims):
        raise ValueError(f"descriptor extents must be ints, got {tuple(dims)}")
    if len(dims) > 2:
        raise ValueError(
            f"multilabel_margin_loss expects a rank<=2 descriptor, got {dims}"
        )
    if any(extent < 0 for extent in dims):
        raise ValueError(f"negative extent in descriptor {dims}")
    if dims and dims[-1] == 0:
        raise ValueError(f"descriptor {dims} has an empty class axis")
    return tuple(dims)


def _n_classes(dims):
    return dims[-1] if dims else 1


def _active_classes(n_classes, reduction):
    # Active classes per row: 1, about half, or all of them. No upper cap.
    if reduction == 0:
        return 1
    if reduction == 1:
        return max(1, n_classes // 2)
    return n_classes


def _case_fn(shape, dtype):
    del dtype
    dims = _descriptor_dims(shape)
    n_classes = _n_classes(dims)
    for reduction in _REDUCTIONS:
        active = _active_classes(n_classes, reduction)
        yield base.BenchmarkCasePlan(
            shape={"input": list(dims), "target": list(dims)},
            params={"reduction": reduction, "target_active": active},
            builder_args=(dims, reduction, active),
        )


def _make_target(dims, active, device):
    # Target rows whose active ids are the last ``active`` classes, descending.
    # Descending ids taken from the top of the range (C-1, C-2, ...) form a
    # non-prefix active set that always reaches the final class, so a kernel that
    # assumes ``active ids are 0..k-1`` produces a different result.
    target = torch.full(dims, -1, dtype=torch.int64, device=device)
    if target.numel() == 0:
        return target
    n_classes = dims[-1] if dims else 1
    rows = (
        target.reshape(1, n_classes)
        if target.dim() < 2
        else target.reshape(-1, n_classes)
    )
    n_active = max(0, min(active, n_classes))
    if n_active:
        ids = torch.arange(
            n_classes - n_active, n_classes, dtype=torch.int64, device=device
        ).flip(0)
        rows[..., :n_active] = ids
    return target


def _build_inputs_fn(plan, dtype, device):
    dims, reduction, active = plan.builder_args
    inp = torch.randn(dims, dtype=dtype, device=device)
    target = _make_target(dims, active, device)
    return inp, target, reduction


class MultiLabelMarginLossBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_DEFAULT_SHAPES)


@pytest.mark.multilabel_margin_loss
def test_multilabel_margin_loss():
    bench = MultiLabelMarginLossBenchmark(
        op_name="multilabel_margin_loss",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.multilabel_margin_loss,
        gems_op=getattr(flag_gems, "multilabel_margin_loss", None),
        dtypes=_DTYPES,
    )
    bench.run()
