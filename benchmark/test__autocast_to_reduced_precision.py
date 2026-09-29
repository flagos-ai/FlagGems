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

"""Benchmark for ``aten::_autocast_to_reduced_precision``.

The op takes five arguments -- ``(self, cuda_enabled, cpu_enabled, cuda_dtype,
cpu_dtype)`` -- so the pointwise families, which build a single-tensor call,
cannot express it and a two-phase ``OperatorBenchmark`` is used instead. Each
case times one dispatch convention: ``cuda_enabled=True`` measures a conversion
path and ``cuda_enabled=False`` the pass-through path. ``torch_op`` is the ATen
reference (the timing baseline) and ``gems_op`` the FlagGems candidate resolved
through ``--override``; both are invoked with the same semantics.

Input dtypes and the conversion-target rows are filtered once, at import, from
the ``flag_gems.runtime.device`` capability flags, so listing and execution see
the identical capability-filtered case set and no support is probed at run time.
"""

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# The five original scales. ``OperatorBenchmark.set_shapes`` keeps resolving a
# ``--shape_file`` entry by operator name and then by class name, and only falls
# back to these defaults when the file carries no entry for this operator.
ACTRP_SHAPES = [
    (1024 * 1024,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (16, 128, 64, 60),
]

# Static capability flags, read once from the device descriptor; the benchmark
# never probes operator support while running and has no permissive default.
_SUPPORT_BF16 = flag_gems.runtime.device.support_bf16
_SUPPORT_FP8 = flag_gems.runtime.device.support_fp8

# ``cpu_dtype`` is a ScalarType schema argument, not an allocation, and is unused
# for a device input on the device branch; it stays different from every
# ``cuda_dtype`` so a candidate that reads the wrong branch times another cast.
CPU_DTYPE = torch.float64

# ``cuda_enabled=False`` keeps the pass-through dispatch in the timing; each
# conversion target owns a distinct cast, and both FP8 formats are listed because
# their overflow rules differ.
_FLAG_TARGET_CASES = [
    (True, False, torch.float16),
    (False, False, torch.float16),
]
if _SUPPORT_BF16:
    _FLAG_TARGET_CASES.append((True, False, torch.bfloat16))
if _SUPPORT_FP8:
    _FLAG_TARGET_CASES.append((True, False, torch.float8_e4m3fn))
    _FLAG_TARGET_CASES.append((True, False, torch.float8_e5m2))

# The input dtype list is filtered with the same static flags, so listing and
# execution expose one capability-filtered case set.
INPUT_DTYPES = [
    dtype for dtype in consts.FLOAT_DTYPES if dtype != torch.bfloat16 or _SUPPORT_BF16
]


def _validated_shape(shape):
    """Reject metadata that cannot describe an allocation before allocating.

    Extents must be non-negative integers; ``bool`` is not an extent. Empty and
    scalar shapes stay valid.
    """
    extents = []
    for extent in shape:
        if isinstance(extent, bool) or not isinstance(extent, int) or extent < 0:
            raise ValueError(
                f"Invalid extent {extent!r} in shape {shape!r}: extents must be "
                "non-negative integers (bool is not an extent)."
            )
        extents.append(extent)
    return tuple(extents)


def _case_fn(shape, dtype):
    del dtype
    valid_shape = _validated_shape(shape)
    for cuda_enabled, cpu_enabled, cuda_dtype in _FLAG_TARGET_CASES:
        yield base.BenchmarkCasePlan(
            shape={"input": list(valid_shape)},
            params={
                "cuda_enabled": cuda_enabled,
                "cpu_enabled": cpu_enabled,
                "cuda_dtype": str(cuda_dtype),
                "cpu_dtype": str(CPU_DTYPE),
            },
            builder_args=(valid_shape, cuda_enabled, cpu_enabled, cuda_dtype),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, cuda_enabled, cpu_enabled, cuda_dtype = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    # The tensor is the positional ``self``; ``unpack_to_args_kwargs`` only folds
    # top-level dicts into the call kwargs, so it must stay a top-level element
    # and the schema parameters travel as keywords.
    return inp, {
        "cuda_enabled": cuda_enabled,
        "cpu_enabled": cpu_enabled,
        "cuda_dtype": cuda_dtype,
        "cpu_dtype": CPU_DTYPE,
    }


class AutocastToReducedPrecisionBenchmark(OperatorBenchmark):
    """This operator's five scales; a ``--shape_file`` entry still takes precedence."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=ACTRP_SHAPES)


@pytest.mark.autocast_to_reduced_precision
def test__autocast_to_reduced_precision():
    bench = AutocastToReducedPrecisionBenchmark(
        op_name="_autocast_to_reduced_precision",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._autocast_to_reduced_precision,
        gems_op=getattr(flag_gems, "_autocast_to_reduced_precision", None),
        dtypes=INPUT_DTYPES,
    )
    bench.run()
