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

import time

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# aten::_print(str s) -> () takes no tensor, so the payload is the workload: `shape`
# records its character count and the phase-two builder returns the message itself.
# The measurement covers the host write; no device launch is added for it.

# Payload sizes run from a short log line to a large dump. The default sweep stays
# bounded because every measured call emits the whole payload; a caller-provided
# shape file may request any non-negative size and is passed through unchanged.
DEFAULT_MESSAGE_CHARS = [16, 256, 4096, 65536, 1048576]

# The payload is text, so no dtype takes part in the measurement. One placeholder
# dtype keeps the case list from repeating identical rows.
BENCH_DTYPES = [torch.float32]


def _message_size(row):
    """Validate one payload descriptor: a non-negative int or [int]."""
    if isinstance(row, bool) or not isinstance(row, (int, list, tuple)):
        raise ValueError(
            f"message descriptor {row!r} is invalid: use a non-negative int or [int]"
        )
    if isinstance(row, (list, tuple)):
        if len(row) != 1:
            raise ValueError(f"message descriptor {row!r} must hold exactly one int")
        return _message_size(row[0])
    if row < 0:
        raise ValueError(f"message descriptor {row!r} must be non-negative")
    return row


def _make_message(chars):
    """Deterministic printable payload of exactly ``chars`` characters."""
    if chars == 0:
        return ""
    return ("flag_gems _print " * (chars // 17 + 1))[:chars]


def _case_fn(shape, dtype):
    del dtype
    chars = _message_size(shape)
    yield base.BenchmarkCasePlan(
        shape={"message_chars": chars},
        params={},
        builder_args=(chars,),
    )


def _build_inputs_fn(plan, dtype, device):
    del dtype, device
    # Phase two returns the flat positional argument the operator takes plus the
    # keyword mapping the shared unpacker expects.
    return (_make_message(_message_size(plan.builder_args)), {})


class PrintBenchmark(OperatorBenchmark):
    """Two-phase benchmark of the host-side string printer."""

    # One call emits the whole payload, so the sample count is fixed instead of
    # being sized by elapsed time, which for a large payload would emit thousands
    # of copies. The reported value stays a per-call latency in ms.
    WARMUP_CALLS = 5
    MEASURED_CALLS = 20

    # This override times exactly the callable it is given, so the reference
    # measurement stays the plain torch.ops.aten._print call. A host write has no
    # device event to time, so it is measured with a wall clock; the marker keeps a
    # native-only run able to report this as the reference runner.
    @base.reference_uses_torch_op
    def get_latency(self, op, *args, **kwargs):
        for _ in range(self.WARMUP_CALLS):
            op(*args, **kwargs)
        base.torch_device_fn.synchronize()
        start = time.perf_counter()
        for _ in range(self.MEASURED_CALLS):
            op(*args, **kwargs)
        base.torch_device_fn.synchronize()
        return (time.perf_counter() - start) * 1000 / self.MEASURED_CALLS

    def set_shapes(self, shape_file_path=None, default_shapes=None):
        defaults = [
            _message_size(row) for row in (default_shapes or DEFAULT_MESSAGE_CHARS)
        ]
        if shape_file_path is None:
            # No shape file configured, so the payload sizes above are the sweep.
            self.shapes = defaults
        else:
            # The shared loader resolves a requested file, raising for an absent or
            # malformed one, and falls back to the payload sizes above when the file
            # has no entry for this operator or class.
            super().set_shapes(shape_file_path, default_shapes=defaults)
        # Normalize and de-duplicate while preserving the requested order, so a size
        # given in both forms counts once and no configured size is dropped.
        self.shapes = list(dict.fromkeys(_message_size(row) for row in self.shapes))


@pytest.mark.print
def test__print():
    bench = PrintBenchmark(
        op_name="_print",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._print,
        gems_op=getattr(flag_gems, "_print", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
