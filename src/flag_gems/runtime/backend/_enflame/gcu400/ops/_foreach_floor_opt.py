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

"""GCU400-specific ``_foreach_floor`` / ``_foreach_floor_`` large-shape path.

The common foreach unary executor already has int64 metadata caching, kernel launch replay
and p4 vectorized packing; a self-built 1D fused kernel rebuilds metadata each time and lacks alignment hints,
measuring only 40% of the common path. Therefore this file isolates only the "large tensor" branch: it reuses the common
``@triton.jit`` kernel and cache utilities, but with floor-specific BLOCK starting points, p4 switches
and an fp32 promotion strategy, avoiding in-place edits to the shared ``_foreach_unary.py`` that would affect sibling
operators.

Small tensors and non-float / non-contiguous shapes fall back directly to the common ``apply_unary``, with dtype policy
and
exception semantics unchanged; the main compute path is always a Triton kernel.
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.foreach import (
    _device_meta,
    _launch_executor_kernel,
    _pack_ok,
    foreach_unary_kernel,
    foreach_unary_kernel_p4,
    same_dtype,
    tl_dtype,
)

from . import _foreach_unary as shared_unary

logger = logging.getLogger(__name__)

# --- TUNING PARAMS START ---
# Rewritten per iteration by the driver; all launch-time config, semantics unchanged.
LARGE_BLOCK_START = 65536
LARGE_TENSOR_STRATEGY = 2
PACK4_MODE = 0
FP32_PROMOTE = 1
USE_SHARED_PATH = 0
FALLBACK_TO_PUBLIC = 0
# --- TUNING PARAMS END ---


@triton.jit
def _floor_promote_fn(x):
    """Same as the common floor_func: promote to fp32 before rounding, then cast back to the original dtype on store."""
    return tl.floor(x.to(tl.float32)).to(x.dtype)


@triton.jit
def _floor_native_fn(x):
    """Round low precision directly; used to check whether the GCU backend accepts tl.floor on fp16/bf16."""
    return tl.floor(x)


def _floor_op():
    """Build a unary table row used only by the large-shape path, without modifying shared UNARY_OPS."""
    fn = _floor_promote_fn if FP32_PROMOTE else _floor_native_fn
    return shared_unary.UnaryOp(
        "floor", fn, same_dtype, shared_unary.NO_BOOL_NO_COMPLEX
    )


def _launch_large_block_group(op, tensors, inplace):
    """Increase BLOCK to push the second dimension of the 2D grid back under the 255 limit, keeping the common cache and
    vectorization.

    The metadata layout is identical to the common executor, so ``_device_meta``
    (with memoisation) and ``_launch_executor_kernel`` (with launch replay) can be reused directly.
    """
    dtype = tensors[0].dtype
    out_dtype = op.out_dtype_fn(dtype)
    if inplace:
        if out_dtype != dtype:
            raise RuntimeError(
                f"result type {str(out_dtype).replace('torch.', '').capitalize()}"
                " can't be cast to the desired output type "
                f"{str(dtype).replace('torch.', '').capitalize()}"
            )
        destinations = tensors
    else:
        destinations = [torch.empty_like(t, dtype=out_dtype) for t in tensors]

    numels = [t.numel() for t in tensors]
    max_numel = max(numels)
    tensor_count = len(tensors)
    block = LARGE_BLOCK_START
    while -(-max_numel // block) > shared_unary.GCU400_GRID_Y_LIMIT:
        block *= 2

    pointer_values = (
        [t.data_ptr() for t in tensors] + [d.data_ptr() for d in destinations] + numels
    )
    meta = _device_meta(pointer_values, tensors[0].device)
    grid = (tensor_count, -(-max_numel // block))
    pack_ok = PACK4_MODE and _pack_ok(
        (dtype, destinations[0].dtype), max_numel, pointer_values
    )
    kernel = foreach_unary_kernel_p4 if pack_ok else foreach_unary_kernel
    _launch_executor_kernel(
        kernel,
        grid,
        tensor_count,
        meta,
        op.fn,
        tl_dtype(dtype),
        tl_dtype(destinations[0].dtype),
        block,
    )
    return None if inplace else destinations


def _run_floor(self, inplace):
    """Small shapes and unsupported forms use the common executor; large shapes use the dedicated parameter branch."""
    if FALLBACK_TO_PUBLIC or USE_SHARED_PATH:
        return shared_unary.apply_unary("floor", self, inplace=inplace)
    try:
        needs_chunking = shared_unary._needs_chunking(self)
    except AttributeError:
        # Non-Tensor elements: fall back to the common executor to match ATen's TypeError/RuntimeError.
        return shared_unary.apply_unary("floor", self, inplace=inplace)
    if not needs_chunking:
        return shared_unary.apply_unary("floor", self, inplace=inplace)
    if LARGE_TENSOR_STRATEGY == 2:
        return _launch_large_block_group(_floor_op(), self, inplace)
    return shared_unary._apply_unary_chunked(_floor_op(), self, inplace)


def _foreach_floor(self):
    logger.debug("GEMS_GCU400 _FOREACH_FLOOR")
    return _run_floor(self, False)


def _foreach_floor_(self):
    logger.debug("GEMS_GCU400 _FOREACH_FLOOR_")
    return _run_floor(self, True)
