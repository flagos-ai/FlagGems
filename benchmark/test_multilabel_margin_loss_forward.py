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

import warnings

import pytest
import torch

import flag_gems

from . import base, consts

BENCH_DTYPES = list(consts.FLOAT_DTYPES)
if flag_gems.runtime.device.support_fp64:
    BENCH_DTYPES.append(torch.float64)


class MultilabelMarginLossBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        # The generic comprehensive list includes (10000, 65536), which is not
        # a meaningful benchmark for this deliberately O(N*C^2) operator.
        return []


def multilabel_margin_loss_forward_input_fn(shape, dtype, device):
    n_rows, n_classes = shape
    input = torch.randn(shape, dtype=dtype, device=device)
    target = (
        torch.arange(n_classes, dtype=torch.int64, device=device)
        .expand(n_rows, n_classes)
        .clone()
    )
    stop = max(1, n_classes // 2)
    if stop < n_classes:
        target[:, stop] = -1
    yield input, target, {"reduction": 1}

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        duplicate_target = target.clone()
        duplicate_target[:, :stop] %= max(1, stop // 4)
        yield input, duplicate_target, {"reduction": 0}

        full_target = torch.arange(n_classes, dtype=torch.int64, device=device).expand(
            n_rows, n_classes
        )
        yield input, full_target, {"reduction": 2}


def _native_fallback_reason(dtype):
    input = torch.randn((2, 16), dtype=dtype, device=flag_gems.device)
    target = (
        torch.arange(16, dtype=torch.int64, device=flag_gems.device)
        .expand(2, 16)
        .clone()
    )
    target[:, 8] = -1

    flag_gems.runtime.torch_device_fn.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            outputs = torch.ops.aten.multilabel_margin_loss_forward(input, target, 1)
            flag_gems.runtime.torch_device_fn.synchronize()
        except RuntimeError as error:
            return f"native reference is unavailable: {error}"

    messages = [str(item.message) for item in caught]
    for message in messages:
        lowered = message.lower()
        if "cpu" in lowered and (
            "fallback" in lowered or "fall back" in lowered or "falling back" in lowered
        ):
            return f"native reference uses a CPU fallback: {message}"
    if any(output.device != input.device for output in outputs):
        return "native reference returned CPU output"
    return None


@pytest.mark.multilabel_margin_loss_forward
@pytest.mark.parametrize("dtype", BENCH_DTYPES)
def test_multilabel_margin_loss_forward(dtype):
    fallback_reason = _native_fallback_reason(dtype)
    if fallback_reason is not None:
        pytest.skip(fallback_reason)

    bench = MultilabelMarginLossBenchmark(
        op_name="multilabel_margin_loss_forward",
        input_fn=multilabel_margin_loss_forward_input_fn,
        torch_op=torch.ops.aten.multilabel_margin_loss_forward,
        dtypes=[dtype],
    )
    bench.run()
