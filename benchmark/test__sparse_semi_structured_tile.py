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

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

# The native kernel exists for float16 and bfloat16 only: float32 is rejected
# ('RuntimeError: Float'), and bfloat16 is gated on the static device capability
# flag. The same list drives listing and execution.
SSST_DTYPES = [torch.float16] + (
    [torch.bfloat16] if flag_gems.runtime.device.support_bf16 else []
)

# Default workloads. (192, 2048) is 64-aligned but not 128-aligned, so it
# exercises the geometry where only the default use_cutlass path is valid; every
# other shape is 128-aligned and accepts all call forms.
SSST_SHAPES = [
    (128, 128),
    (512, 512),
    (1024, 1024),
    (2048, 1024),
    (4096, 1024),
    (192, 2048),
]

# (algorithm, use_cutlass) covering every valid combination; None is the omitted
# algorithm argument, i.e. the schema default, a distinct call form from the
# explicit empty string.
SSST_CALL_FORMS = [
    (None, True),
    (None, False),
    ("", True),
    ("", False),
    ("largest_values_greedy", True),
    ("largest_values_greedy", False),
    ("largest_abs_values_greedy", True),
    ("largest_abs_values_greedy", False),
]


def _validate_shape(shape):
    """Reject custom descriptors that are not valid workloads of this operator.

    The kernel sparsifies 2-D tiles, needs positive extents and fails for a zero
    extent ('CUDA error: invalid configuration argument'), and its result is only
    reproducible at 64-aligned rows and columns, so a descriptor outside that
    geometry cannot be timed meaningfully. An invalid descriptor is reported
    rather than silently replaced by a default workload.
    """
    if not isinstance(shape, (tuple, list)):
        raise ValueError(
            "descriptor %r is not a shape sequence (tuple/list), got %s"
            % (shape, type(shape).__name__)
        )
    if len(shape) != 2:
        raise ValueError(
            "descriptor %r has rank %d; this operator sparsifies 2-D tensors only"
            % (tuple(shape), len(shape))
        )
    rows, cols = shape
    for name, extent in (("rows", rows), ("cols", cols)):
        if isinstance(extent, bool) or not isinstance(extent, int):
            raise ValueError(
                "%s extent must be an int, got %r in %r" % (name, extent, tuple(shape))
            )
        if extent <= 0:
            raise ValueError(
                "%s extent must be positive, got %r in %r"
                % (name, extent, tuple(shape))
            )
        if extent % 64:
            raise ValueError(
                "%s must be a multiple of 64 for a reproducible workload, got %d in %r"
                % (name, extent, tuple(shape))
            )
    return rows, cols


def _case_fn(shape, dtype):
    del dtype
    rows, cols = _validate_shape(shape)
    for algorithm, use_cutlass in SSST_CALL_FORMS:
        if not use_cutlass and (rows % 128 or cols % 128):
            # The non-cutlass kernel only supports rows/cols multiples of 128
            # ('Only supports rows/cols multiples of 128'), so this form is not a
            # valid workload for that geometry.
            continue
        kwargs = {"use_cutlass": use_cutlass}
        if algorithm is not None:
            kwargs["algorithm"] = algorithm
        yield base.BenchmarkCasePlan(
            shape={"input": tuple(shape)},
            params={
                "algorithm": "omitted" if algorithm is None else algorithm,
                "use_cutlass": use_cutlass,
            },
            builder_args=(tuple(shape), dict(kwargs)),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, kwargs = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    # The trailing dict becomes call kwargs, so the reference and the candidate
    # receive exactly the same arguments.
    return inp, dict(kwargs)


class SparseSemiStructuredTileBenchmark(OperatorBenchmark):
    """Case-based benchmark for the 2-D sparse semi-structured tiling op."""

    def set_shapes(self, shape_file_path=None):
        # Honor a custom shape file; SSST_SHAPES are this operator's own defaults.
        super().set_shapes(shape_file_path, default_shapes=SSST_SHAPES)


@pytest.mark.sparse_semi_structured_tile
def test__sparse_semi_structured_tile():
    bench = SparseSemiStructuredTileBenchmark(
        op_name="_sparse_semi_structured_tile",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_semi_structured_tile,
        gems_op=getattr(flag_gems, "_sparse_semi_structured_tile", None),
        dtypes=SSST_DTYPES,
    )
    bench.run()
