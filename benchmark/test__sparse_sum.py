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
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, consts
from .generated_operator_utils import OperatorBenchmark

# _sparse_sum starts with an underscore, so register its marker explicitly.
setattr(
    pytest.mark,
    "_sparse_sum",
    MarkDecorator(Mark("_sparse_sum", (), {}, _ispytest=True), _ispytest=True),
)

# bfloat16 needs the backend kernel; the gate is a static capability flag, so
# listing stays allocation-free and never probes tensors. The benchmark only runs
# the full-reduction overload, for which the native operator implements every dtype
# in the consts.FLOAT_DTYPES set.
_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype is not torch.bfloat16 or flag_gems.runtime.device.support_bf16
]

# Sparse descriptors are (sparse_shape, nnz) pairs: the operator is COO-only, so
# the shape file contract has to carry the stored-entry count as well. A sparse
# reduction's work scales with nnz, so the stored values are sized for a meaningful
# measurement while the shapes stay cheap to rebuild; nnz > numel for the small
# shapes guarantees duplicate coordinates, so native coalescing has real merging
# work to do. The 0-dim scalar shape is included: a sparse COO tensor stores an
# (0, nnz) index buffer and native reduces it like any other shape.
_SPARSE_SUM_SHAPES = [
    ((1024, 1024), 65536),
    ((1024, 1024), 262144),
    ((1024, 1024), 1048576),
    ((4096, 4096), 1048576),
    ((20, 320, 15), 262144),
    ((16, 128, 64, 60), 262144),
    ((16, 7, 57, 32, 29), 262144),
    ((), 4096),
]

# Comprehensive level adds the larger variants; they stay (shape, nnz)
# descriptors so the same case_fn keeps consuming them.
_SPARSE_SUM_MORE_SHAPES = [
    ((4096, 4096), 2097152),
    ((2048, 2048), 2097152),
    ((256, 256, 256), 1048576),
]


def _check_descriptor(sparse_shape, nnz):
    # Inconsistent descriptors are rejected here, during metadata planning, instead
    # of being silently rewritten into a different workload: a zero-sized extent can
    # only store zero entries, and neither an extent nor nnz may be negative.
    if nnz < 0 or any(dim < 0 for dim in sparse_shape):
        raise ValueError(
            f"_sparse_sum descriptor {tuple(sparse_shape)}/{nnz} has a negative "
            "shape extent or nnz"
        )
    if nnz and any(dim == 0 for dim in sparse_shape):
        raise ValueError(
            f"_sparse_sum descriptor {tuple(sparse_shape)}/{nnz} requests {nnz} "
            "stored entries for a shape with a zero-sized extent"
        )


def _case_fn(shape, dtype):
    # Two-phase GenericBenchmark: each entry is a (shape, nnz) descriptor and one
    # BenchmarkCasePlan is emitted per descriptor, so the case id and params stay
    # metadata-only and tensor construction is deferred to the build phase. The
    # descriptor is validated before it is planned, so a bad shape file fails during
    # listing without allocating anything.
    del dtype
    sparse_shape, nnz = shape
    _check_descriptor(sparse_shape, nnz)
    yield base.BenchmarkCasePlan(
        shape={"input": sparse_shape},
        params={"nnz": nnz},
        builder_args=(sparse_shape, nnz),
    )


def _build_inputs_fn(plan, dtype, device):
    # Coordinates are drawn with replacement, so the operand is uncoalesced and
    # contains duplicates, which mirrors the native workload that has to merge
    # them during the reduction.
    sparse_shape, nnz = plan.builder_args
    _check_descriptor(sparse_shape, nnz)
    if nnz == 0:
        # The zero-sized shapes land here too, since _check_descriptor requires them
        # to carry nnz == 0. Such an operand stores an empty (rank, 0) index buffer;
        # randint(0, dim, ...) would reject the empty range of a zero-sized axis
        # before the operator ever runs.
        indices = torch.empty((len(sparse_shape), 0), dtype=torch.long, device=device)
        values = torch.empty(0, dtype=dtype, device=device)
    elif len(sparse_shape) == 0:
        # A 0-dim sparse COO tensor stores an (0, nnz) index buffer.
        indices = torch.empty((0, nnz), dtype=torch.long, device=device)
        values = torch.randn(nnz, dtype=dtype, device=device)
    else:
        indices = torch.stack(
            [
                torch.randint(0, dim, (nnz,), dtype=torch.long, device=device)
                for dim in sparse_shape
            ]
        )
        values = torch.randn(nnz, dtype=dtype, device=device)
    inp = torch.sparse_coo_tensor(indices, values, sparse_shape, device=device)
    return inp, {}


class SparseSumBenchmark(OperatorBenchmark):
    """Benchmark restricted to (sparse shape, nnz) workload descriptors.

    ``set_shapes``/``set_more_shapes`` are overridden because the default
    implementations read the dense core_shapes.yaml set, which cannot express a
    COO operand's stored-entry count. An operator-specific entry in a
    caller-supplied shape file still takes precedence.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_SPARSE_SUM_SHAPES)
        # The generic merge in base.set_shapes is reached only through the dense
        # path, so the descriptor-form extras are merged here instead.
        if (
            base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE
            and not base.Config.query
        ):
            self.shapes = list(dict.fromkeys(self.shapes + self.set_more_shapes()))

    def set_more_shapes(self):
        # Extras must keep the descriptor form, otherwise dense shapes would be
        # fed to the descriptor case_fn.
        return _SPARSE_SUM_MORE_SHAPES


@pytest.mark._sparse_sum
def test__sparse_sum():
    bench = SparseSumBenchmark(
        op_name="_sparse_sum",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        # torch_op is the perf reference and gems_op the injected candidate;
        # both use the same single-operand call form.
        torch_op=torch.ops.aten._sparse_sum,
        gems_op=getattr(flag_gems, "_sparse_sum", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
