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

from collections.abc import Mapping

import pytest
import torch

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# The five scales of this operator's benchmark. They are the class defaults, so
# the shared reader uses them whenever the shape file names neither this operator
# nor one of its benchmark classes. The primary 16x16 descriptor of every scale
# stays the first case of that scale.
_DEFAULT_SHAPES = [
    (4096, 4096),
    (1024, 1024),
    (64, 512, 512),
    (16, 256, 256),
    (512, 2048),
]

# Candidate square block sizes, largest first: every scale below is a multiple of
# the primary 16x16 block.
_BLOCK_SIZES = [(16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]

# Extra (blocksize, dense_dim, block_stride) descriptors per scale: rectangular
# blocks (a different block grid), an explicitly passed dense tail, and payloads
# that populate only every n-th block row. Each is resolved after the dense
# descriptor of the same scale.
_SUPPLEMENTAL_DESCRIPTORS = {
    (4096, 4096): [((16, 8), None, 4), ((8, 16), None, 1)],
    (1024, 1024): [((8, 8), None, 2)],
    (64, 512, 512): [((8, 4), None, 1), ((16, 16), 1, 1)],
    (16, 256, 256): [((4, 8), None, 8)],
    (512, 2048): [((4, 4), None, 1)],
}

# The only fields an explicit mapping descriptor may carry.
_DESCRIPTOR_FIELDS = ("shape", "input", "blocksize", "dense_dim", "block_stride")

_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _dense_dim(dense_dim):
    return 0 if dense_dim is None else dense_dim


def _matrix_dims(shape, dense_dim):
    dense_dim = _dense_dim(dense_dim)
    return tuple(shape[len(shape) - 2 - dense_dim : len(shape) - dense_dim])


def _batch_size(shape, dense_dim):
    size = 1
    for extent in shape[: len(shape) - 2 - _dense_dim(dense_dim)]:
        size *= extent
    return size


def _require_int(value, what):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}: {value!r}")
    return value


def _validated_shape(shape):
    if not isinstance(shape, (tuple, list)):
        raise TypeError(f"a shape must be a sequence of extents, got {shape!r}")
    extents = tuple(shape)
    if len(extents) < 2:
        raise ValueError(
            "to_sparse_bsr converts the last two dimensions into a block "
            f"matrix, so a shape of rank {len(extents)} cannot be converted: "
            f"{extents}"
        )
    for extent in extents:
        _require_int(extent, "shape entry")
        if extent < 0:
            raise ValueError(f"shape entries must be non-negative, got {extent}")
    return extents


def _validated_dense_dim(extents, dense_dim):
    if dense_dim is None:
        return None
    _require_int(dense_dim, "dense_dim")
    if dense_dim < 0 or dense_dim > len(extents) - 2:
        raise ValueError(
            f"dense_dim must be in [0, {len(extents) - 2}] for shape {extents}, "
            f"got {dense_dim}"
        )
    return dense_dim


def _validated_blocksize(blocksize):
    if not isinstance(blocksize, (tuple, list)):
        raise TypeError(f"a blocksize must be a sequence, got {blocksize!r}")
    extents = tuple(blocksize)
    if len(extents) != 2:
        raise ValueError(f"blocksize must hold exactly 2 extents, got {len(extents)}")
    for extent in extents:
        _require_int(extent, "blocksize extent")
        if extent <= 0:
            raise ValueError(f"blocksize extents must be positive, got {extents}")
    return extents


def _validated_stride(block_stride):
    if block_stride is None:
        return 1
    _require_int(block_stride, "block_stride")
    if block_stride <= 0:
        raise ValueError(f"block_stride must be positive, got {block_stride}")
    return block_stride


def _descriptor(shape, blocksize, dense_dim, block_stride):
    # One validated workload descriptor; plain shapes and explicit descriptors
    # both travel through here, so listing and execution cannot diverge. An
    # omitted dense_dim stays omitted, so the schema default is exercised, and a
    # zero extent is kept as long as the batch product stays non-zero, which is
    # exactly what the native operator requires.
    extents = _validated_shape(shape)
    _validated_dense_dim(extents, dense_dim)
    if _batch_size(extents, dense_dim) == 0:
        raise ValueError(
            "to_sparse_bsr requires a non-zero batch product, got "
            f"{extents} with dense_dim {_dense_dim(dense_dim)}"
        )
    blocks = _validated_blocksize(blocksize)
    rows, cols = _matrix_dims(extents, dense_dim)
    if rows % blocks[0] or cols % blocks[1]:
        raise ValueError(f"blocksize {blocks} does not divide the matrix {rows}x{cols}")
    return extents, blocks, dense_dim, _validated_stride(block_stride)


def _default_blocksize(shape, dense_dim):
    # Documented default descriptor of a plain shape: the largest candidate
    # square block that divides the matrix, or 1x1 for odd extents.
    rows, cols = _matrix_dims(shape, dense_dim)
    for blocksize in _BLOCK_SIZES:
        if rows % blocksize[0] == 0 and cols % blocksize[1] == 0:
            return blocksize
    return (1, 1)


def _descriptor_from_fields(shape, blocksize, dense_dim, block_stride):
    # The structure is validated before the default block size is derived, so an
    # explicit blocksize is never replaced by a computed default and an invalid
    # dense_dim cannot silently shift the extracted matrix.
    extents = _validated_shape(shape)
    _validated_dense_dim(extents, dense_dim)
    if blocksize is None:
        blocksize = _default_blocksize(extents, dense_dim)
    return _descriptor(extents, blocksize, dense_dim, block_stride)


def _descriptor_from_mapping(spec):
    unknown = sorted(str(key) for key in set(spec) - set(_DESCRIPTOR_FIELDS))
    if unknown:
        raise ValueError(f"unknown descriptor fields {unknown} in {dict(spec)}")
    for key in ("shape", "input"):
        if spec.get(key) is not None and not isinstance(spec[key], (tuple, list)):
            raise TypeError(f"descriptor {key!r} must be a sequence: {spec[key]!r}")
    given = {key: spec[key] for key in ("shape", "input") if spec.get(key) is not None}
    if len(given) == 2 and tuple(given["shape"]) != tuple(given["input"]):
        raise ValueError(
            f"descriptor declares conflicting shape and input entries: {dict(spec)}"
        )
    if not given:
        raise ValueError(f"descriptor is missing a shape entry: {dict(spec)}")
    shape = given.get("shape", given.get("input"))
    return _descriptor_from_fields(
        shape, spec.get("blocksize"), spec.get("dense_dim"), spec.get("block_stride")
    )


def _descriptor_from_sequence(spec):
    # `(shape, blocksize, dense_dim, block_stride)` with optional trailing
    # entries: a shape file can carry this form, because OperatorBenchmark's
    # reader rewrites every entry through its recursive `as_tuple` and a nested
    # sequence survives that unchanged.
    if len(spec) > 4:
        raise ValueError(
            "a sequence descriptor holds at most (shape, blocksize, dense_dim, "
            f"block_stride), got {len(spec)} entries: {spec}"
        )
    return _descriptor_from_fields(*(list(spec) + [None] * (4 - len(spec))))


def _descriptors_for_shape(shape):
    # A plain shape resolves to its documented default descriptor first, then to
    # any supplemental descriptor registered for that shape. Duplicates are
    # dropped so a shape cannot be listed twice.
    extents = _validated_shape(shape)
    candidates = [(_default_blocksize(extents, None), None, 1)]
    candidates.extend(_SUPPLEMENTAL_DESCRIPTORS.get(extents, ()))
    descriptors = []
    for blocksize, dense_dim, block_stride in candidates:
        entry = _descriptor(extents, blocksize, dense_dim, block_stride)
        if entry not in descriptors:
            descriptors.append(entry)
    return descriptors


def _descriptors_for(spec):
    # A spec is a plain shape, a sequence descriptor (its first entry is itself a
    # shape) or a mapping descriptor with the same fields. Invalid ranks and
    # invalid descriptors are rejected here rather than silently rewritten.
    if isinstance(spec, Mapping):
        return [_descriptor_from_mapping(spec)]
    if not isinstance(spec, (tuple, list)):
        raise TypeError(f"unsupported shape spec: {spec!r}")
    if spec and isinstance(spec[0], (tuple, list)):
        return [_descriptor_from_sequence(spec)]
    return _descriptors_for_shape(spec)


def _kept_block_rows(shape, blocksize, dense_dim, block_stride, device):
    # Boolean selector over the dense shape: True inside every kept block row
    # (`row_block % block_stride == 0`). It is built by expansion, so it needs no
    # index tensor and is identical in every batch element.
    dense = _dense_dim(dense_dim)
    rows, _ = _matrix_dims(shape, dense_dim)
    n_row_blocks = rows // blocksize[0]
    lead_dims = len(shape) - 2 - dense
    selector = torch.zeros(
        (1,) * lead_dims + (n_row_blocks, 1, 1) + (1,) * dense,
        dtype=torch.bool,
        device=device,
    )
    selector[(slice(None),) * lead_dims + (slice(None, None, block_stride),)] = True
    full = (
        tuple(shape[:lead_dims])
        + (n_row_blocks, blocksize[0], _matrix_dims(shape, dense_dim)[1])
        + tuple(shape[len(shape) - dense :])
    )
    return selector.expand(full).reshape(shape)


def _case_fn(shape, dtype):
    del dtype
    for plan_shape, blocksize, dense_dim, block_stride in _descriptors_for(shape):
        yield base.BenchmarkCasePlan(
            shape={"input": list(plan_shape)},
            params={
                "blocksize": list(blocksize),
                "dense_dim": dense_dim,
                # Stored payload density: 1 populates every block row, n > 1 only
                # every n-th one, which is exactly what the fixture stores.
                "block_stride": block_stride,
            },
            builder_args=(plan_shape, blocksize, dense_dim, block_stride),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, blocksize, dense_dim, block_stride = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    dense = _dense_dim(dense_dim)
    if inp.numel():
        if block_stride > 1:
            # Clear every unselected block row, so the payload really is the
            # sparse one the case metadata advertises instead of the full random
            # tensor, and every batch element keeps the same stored-block count.
            inp.masked_fill_(
                ~_kept_block_rows(
                    shape, blocksize, dense_dim, block_stride, inp.device
                ),
                0,
            )
        rows, cols = _matrix_dims(shape, dense_dim)
        lead = (slice(None),) * (len(shape) - 2 - dense)
        # One anchor at the first position of every block of every kept block
        # row. Slicing the tensor writes it in place, and the same slice applies
        # to all batch elements.
        anchor = (
            lead
            + (
                slice(0, rows, block_stride * blocksize[0]),
                slice(0, cols, blocksize[1]),
            )
            + (0,) * dense
        )
        inp[anchor] = 1.0
    args = {"blocksize": list(blocksize)}
    if dense_dim is not None:
        args["dense_dim"] = dense_dim
    return inp, args


class ToSparseBsrBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = _DEFAULT_SHAPES

    def set_more_shapes(self):
        # The inherited list holds a flat element count and generic 2-D/3-D
        # shapes that describe no block grid, and merging it would also append a
        # rank-1 shape this operator cannot convert; the scales above are the only
        # ones this operator adds.
        return []

    def set_shapes(self, shape_file_path=None):
        # The shared reader in generated_operator_utils keeps both mapping and
        # nested-sequence descriptors intact and already falls back to the class
        # defaults when the shape file names neither this operator nor one of its
        # classes, so no local YAML/MRO handling is needed here.
        return super().set_shapes(
            shape_file_path, default_shapes=type(self).DEFAULT_SHAPES
        )


@pytest.mark.to_sparse_bsr
def test_to_sparse_bsr():
    bench = ToSparseBsrBenchmark(
        op_name="to_sparse_bsr",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.to_sparse_bsr,
        gems_op=getattr(flag_gems, "to_sparse_bsr", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
