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

import torch
import triton
import triton.language as tl

try:
    import triton.experimental.tle.language as tle

    _HAS_SDNN = True
except ImportError:
    _HAS_SDNN = False

logger = logging.getLogger(__name__)

_SDNN_CONFIG = {
    torch.float16: (16384, 8, 2),
    torch.float32: (16384, 2, 2),
}


@triton.jit(do_not_specialize=["scale"])
def native_dropout_backward_sdnn_kernel(
    grad_ptr,
    mask_ptr,
    out_ptr,
    scale,
    numel,
    DTYPE: tl.constexpr,
    TILE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    CAPACITY: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * (TILE * NBLOCKS)
    c = tl.arange(0, TILE)

    g_buf = tle.dsa.alloc([TILE], DTYPE, tle.dsa.UNI_SRAM, num_buffers=CAPACITY)
    m_buf = tle.dsa.alloc([TILE], tl.int8, tle.dsa.UNI_SRAM, num_buffers=CAPACITY)
    g_pipe = tle.pipe(capacity=CAPACITY, scope="core", name="g", tile=g_buf)
    m_pipe = tle.pipe(capacity=CAPACITY, scope="core", name="m", tile=m_buf)
    g_writer, g_reader = g_pipe.writer(), g_pipe.reader()
    m_writer, m_reader = m_pipe.writer(), m_pipe.reader()

    s = scale.to(DTYPE)
    nblocks = tl.minimum(NBLOCKS, (numel - base + TILE - 1) // TILE)
    for i in tl.range(0, nblocks):
        off = base + i * TILE
        tail = tl.minimum(numel - off, TILE)

        g_slot = g_writer.acquire(i)
        tle.dsa.copy(grad_ptr + off + c, g_slot.tile, sizes=[tail])
        g_writer.commit(i)
        m_slot = m_writer.acquire(i)
        tle.dsa.copy(mask_ptr + off + c, m_slot.tile, sizes=[tail])
        m_writer.commit(i)

        g_ready = g_reader.wait(i)
        m_ready = m_reader.wait(i)
        g = tle.dsa.to_tensor(g_ready.slot.tile)
        m = tle.dsa.to_tensor(m_ready.slot.tile)
        res = (m.to(DTYPE) * s) * g
        tle.dsa.copy(res, out_ptr + off + c, sizes=[tail])

        g_reader.release(i)
        m_reader.release(i)


_BLOCK = 65536
_NUM_WARPS = 32


@triton.jit
def native_dropout_backward_kernel(
    grad, mask, scale, out, N, BLOCK: tl.constexpr, NEED_MASK: tl.constexpr
):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    if NEED_MASK:
        idx_mask = off < N
        m = tl.load(mask + off, mask=idx_mask, other=0)
        g = tl.load(grad + off, mask=idx_mask, other=0.0)
        r = tl.where(m != 0, g * scale.to(g.dtype), tl.zeros_like(g))
        tl.store(out + off, r, mask=idx_mask)
    else:
        m = tl.load(mask + off)
        g = tl.load(grad + off)
        r = tl.where(m != 0, g * scale.to(g.dtype), tl.zeros_like(g))
        tl.store(out + off, r)


def native_dropout_backward(grad_output, mask, scale):
    """aten::native_dropout_backward specialized for XPU.

    out = grad * (mask != 0) * scale. fp16/fp32 take the SDNN coprocessor
    tier (descriptor-DMA + unified-precision multiply); bf16 (whose SDNN
    pipeline does not verify) takes the unified-precision LSU tier (mask
    feeds the select predicate only, the fp32 scale is cast once).
    """
    logger.debug("GEMS_KUNLUNXIN NATIVE_DROPOUT_BACKWARD")
    num_tasks = grad_output.numel()
    if num_tasks == 0:
        return grad_output.clone()
    grad_output = grad_output.contiguous()
    grad_input = torch.empty_like(grad_output)
    mask = mask.contiguous()
    N = num_tasks

    if _HAS_SDNN and grad_output.dtype in _SDNN_CONFIG:
        try:
            tile, nblocks, capacity = _SDNN_CONFIG[grad_output.dtype]
            native_dropout_backward_sdnn_kernel[(triton.cdiv(N, tile * nblocks),)](
                grad_output.view(-1),
                mask.view(-1).view(torch.int8),
                grad_input.view(-1),
                scale,
                N,
                DTYPE=(
                    tl.float16 if grad_output.dtype == torch.float16 else tl.float32
                ),
                TILE=tile,
                NBLOCKS=nblocks,
                CAPACITY=capacity,
                is_sdnn=True,
                num_stages=2,
            )
            return grad_input
        except Exception as e:  # pragma: no cover - SDNN unsupported device
            logger.warning(
                "GEMS_KUNLUNXIN native_dropout_backward SDNN tier failed, "
                "falling back to LSU: %s",
                e,
            )

    native_dropout_backward_kernel[(triton.cdiv(N, _BLOCK),)](
        grad_output.view(-1),
        mask.view(-1).view(torch.int8),
        scale,
        grad_input.view(-1),
        N,
        BLOCK=_BLOCK,
        NEED_MASK=N % _BLOCK != 0,
        num_warps=_NUM_WARPS,
    )
    return grad_input
