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

import logging
from contextlib import contextmanager

import torch
import triton
import triton.language as tl
from triton.backends.tsingmicro.precision import (
    restore_precision_mode,
    set_precision_mode,
)

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)

# the uplimit f32 can present the precision of i32
_F32_PRECISION_NUMEL_THRESHOLD = 2**24
_MAX_GRID = 65535


def _normalize_shapes(inp: torch.Tensor, sizes):
    """Align inp rank to len(sizes) by prepending 1s (torch.repeat 语义)."""
    sizes = tuple(int(s) for s in sizes)
    in_shape = list(inp.shape)
    assert len(sizes) >= len(in_shape), (
        "Number of dimensions of repeat dims can not be smaller than "
        "number of dimensions of tensor"
    )
    if len(sizes) > len(in_shape):
        diff = len(sizes) - len(in_shape)
        in_shape = [1] * diff + in_shape
        x = inp.reshape(in_shape)
    else:
        x = inp
    for r in sizes:
        assert r >= 0, (
            f"the number of repetitions per dimension out of range "
            f"(expected to >= 0) but got {r}"
        )
    out_shape = tuple(s * r for s, r in zip(in_shape, sizes))
    return x, sizes, tuple(in_shape), out_shape


def _can_expand_repeat(in_shape, sizes) -> bool:
    for s, r in zip(in_shape, sizes):
        if r > 1 and s != 1:
            return False
    return True


def _outer_nrep_inner(in_shape, sizes):
    idxs = [i for i, r in enumerate(sizes) if r > 1]
    if not idxs:
        return None
    first, last = idxs[0], idxs[-1]
    for i in range(first, last + 1):
        if in_shape[i] != 1:
            return None
    outer = 1
    for s in in_shape[:first]:
        outer *= s
    nrep = 1
    for r in sizes[first : last + 1]:
        nrep *= r
    inner = 1
    for s in in_shape[last + 1 :]:
        inner *= s
    return outer, nrep, inner


def _prod(xs):
    p = 1
    for v in xs:
        p *= int(v)
    return p


@contextmanager
def _maybe_i64_index(numel: int):
    if numel <= _F32_PRECISION_NUMEL_THRESHOLD:
        yield
        return
    precision_token = set_precision_mode("1")
    try:
        yield
    finally:
        restore_precision_mode(precision_token)


@libentry()
@triton.jit
def _repeat_memcpy_kernel(
    in_ptr,
    out_ptr,
    nrep,
    inner,
    n_chunks,
    num_tasks,
    tiles_per_cta,
    BLOCK: tl.constexpr,
    one_tile_per_cta: tl.constexpr,
):
    pid = tle.program_id(0)
    lane = tl.arange(0, BLOCK)

    if one_tile_per_cta:
        chunk = pid % n_chunks
        o = pid // n_chunks
        offs = chunk * BLOCK + lane
        mask = offs < inner
        v = tl.load(in_ptr + o * inner + offs, mask=mask)
        out_base = o * nrep * inner
        for r in range(nrep):
            tl.store(out_ptr + out_base + r * inner + offs, v, mask=mask)
    else:
        num_ctas = tle.num_programs(0)
        for j in range(0, tiles_per_cta):
            task = pid + j * num_ctas
            in_range = task < num_tasks
            task_c = tl.minimum(task, num_tasks - 1)
            chunk = task_c % n_chunks
            o = task_c // n_chunks
            offs = chunk * BLOCK + lane
            mask = in_range & (offs < inner)
            v = tl.load(in_ptr + o * inner + offs, mask=mask)
            out_base = o * nrep * inner
            for r in range(nrep):
                tl.store(out_ptr + out_base + r * inner + offs, v, mask=mask)


def _repeat_by_memcpy(x: torch.Tensor, out_shape, outer, nrep, inner):
    if not x.is_contiguous():
        x = x.contiguous()
    x2 = x.reshape(outer, inner)
    out = torch.empty(out_shape, device=x.device, dtype=x.dtype)
    out2 = out.reshape(outer, nrep, inner)

    if inner >= 65536:
        block = 4096
    elif inner >= 4096:
        block = 2048
    else:
        block = 1024
    block = min(block, triton.next_power_of_2(max(inner, 1)))
    n_chunks = triton.cdiv(max(inner, 1), block)
    num_tasks = max(outer * n_chunks, 1)
    num_ctas = min(_MAX_GRID, num_tasks)
    tiles_per_cta = triton.cdiv(num_tasks, num_ctas)
    one_tile_per_cta = tiles_per_cta == 1

    with torch_device_fn.device(x.device.index), _maybe_i64_index(outer * nrep * inner):
        _repeat_memcpy_kernel[(num_ctas,)](
            x2,
            out2,
            nrep,
            inner,
            n_chunks,
            num_tasks,
            tiles_per_cta,
            BLOCK=block,
            one_tile_per_cta=one_tile_per_cta,
            num_warps=1,
            num_stages=1,
        )
    return out


def _repeat_by_successive_dims(x, sizes, in_shape, out_shape):
    """Tile each dim with contiguous row memcpy (inner → outer).

    torch.repeat along dim i by r copies the whole (shape[i:] ) block r times,
    which is exactly (outer=prod(shape[:i]), nrep=r, inner=prod(shape[i:])).
    """
    if not x.is_contiguous():
        x = x.contiguous()
    x = x.reshape(in_shape)
    cur = list(in_shape)
    for dim in reversed(range(len(sizes))):
        r = sizes[dim]
        if r == 1:
            continue
        outer = _prod(cur[:dim])
        inner = _prod(cur[dim:])
        cur[dim] = cur[dim] * r
        logger.debug(
            "GEMS_TSINGMICRO REPEAT dim-memcpy dim=%s r=%s outer=%s inner=%s -> %s",
            dim,
            r,
            outer,
            inner,
            tuple(cur),
        )
        x = _repeat_by_memcpy(x, tuple(cur), outer, r, inner)
    return x.reshape(out_shape)


def repeat(inp, sizes):
    logger.debug("GEMS_TSINGMICRO REPEAT")
    x, sizes, in_shape, out_shape = _normalize_shapes(inp, sizes)

    if any(d == 0 for d in out_shape):
        return inp.new_empty(out_shape)

    if all(r == 1 for r in sizes):
        return x.clone()

    if _can_expand_repeat(in_shape, sizes):
        oi = _outer_nrep_inner(in_shape, sizes)
        if oi is not None:
            outer, nrep, inner = oi
            if outer > 0 and inner > 0 and nrep > 0:
                logger.debug(
                    "GEMS_TSINGMICRO REPEAT fast-path memcpy outer=%s nrep=%s inner=%s out=%s",
                    outer,
                    nrep,
                    inner,
                    out_shape,
                )
                return _repeat_by_memcpy(x, out_shape, outer, nrep, inner)

        logger.debug(
            "GEMS_TSINGMICRO REPEAT fast-path expand+contiguous in=%s sizes=%s out=%s",
            in_shape,
            sizes,
            out_shape,
        )
        return x.expand(out_shape).contiguous()

    logger.debug(
        "GEMS_TSINGMICRO REPEAT dim-memcpy in=%s sizes=%s out=%s",
        in_shape,
        sizes,
        out_shape,
    )
    return _repeat_by_successive_dims(x, sizes, in_shape, out_shape)
