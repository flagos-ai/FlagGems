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

"""GCU400 专属 ``new_zeros`` host 侧分派。

公共通用实现只把 ``device.type == "cuda"`` 视为可被 Triton kernel 直接写，
GCU 上因此改走 ``scratch = torch.empty(..., device="cuda")`` + ``copy_`` 的
中转路径，额外付出一次分配、一遍写零和一遍拷贝。这里保留公共实现里已经
编译缓存过的 ``_zero_bytes`` / ``_zero_strided`` Triton kernel，仅修正 host
侧设备判定，直接在目标 buffer 上发射 kernel；算子语义与函数签名不变。
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.new_zeros import _zero_bytes, _zero_strided

logger = logging.getLogger(__name__)


@triton.jit
def _zero_words(p, N, BLOCK: tl.constexpr):
    """按 32-bit word 粒度清零：相比 uint8 store 减少 4 倍写指令。"""
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    tl.store(p + offs, tl.zeros([BLOCK], p.dtype.element_ty), mask=offs < N)


# --- TUNING PARAMS START ---
DIRECT_DEVICE = 1
USE_WORDS = 1
BLOCK_BYTES = 8192
BLOCK_WORDS = 16384
NUM_WARPS = 4
SMALL_THRESHOLD_BYTES = 0
SMALL_BLOCK_BYTES = 8192
BLOCK_ELEMS_STRIDED = 1024
STRIDED_WARPS = 4
SKIP_PINNED_CHECK = 1
# --- TUNING PARAMS END ---


def _as_size(size):
    """把 ``size`` 归一化为 int 元组，语义与通用实现一致。"""
    if isinstance(size, torch.Size):
        return tuple(int(e) for e in size)
    if isinstance(size, (list, tuple)):
        out = []
        for e in size:
            if isinstance(e, bool):
                out.append(int(e))
            else:
                out.append(e)
        return tuple(out)
    if isinstance(size, torch.Tensor):
        return tuple(int(e) for e in size.tolist())
    raise TypeError(
        "new_zeros(): argument 'size' must be a sequence of integers, "
        f"not {type(size).__name__}"
    )


def _empty(size_t, dtype, layout, device, pin_memory):
    kwargs = {"dtype": dtype, "device": device}
    if layout is not None:
        kwargs["layout"] = layout
    if pin_memory:
        kwargs["pin_memory"] = True
    return torch.empty(size_t, **kwargs)


def _fill_contiguous(buf):
    """连续 buffer：优先按 32-bit word 发射，字节数非 4 倍数时回退字节路径。"""
    flat = buf.reshape(-1)
    byte_view = flat.view(torch.uint8)
    num_bytes = byte_view.numel()
    if USE_WORDS and num_bytes % 4 == 0 and num_bytes > 0:
        word_view = byte_view.view(torch.int32)
        num_words = word_view.numel()
        block = BLOCK_WORDS
        if SMALL_THRESHOLD_BYTES and num_bytes <= SMALL_THRESHOLD_BYTES:
            block = max(1, SMALL_BLOCK_BYTES // 4)
        grid = (triton.cdiv(num_words, block),)
        _zero_words[grid](word_view, num_words, block, num_warps=NUM_WARPS)
        return
    block = BLOCK_BYTES
    if SMALL_THRESHOLD_BYTES and num_bytes <= SMALL_THRESHOLD_BYTES:
        block = SMALL_BLOCK_BYTES
    grid = (triton.cdiv(num_bytes, block),)
    _zero_bytes[grid](byte_view, num_bytes, block, num_warps=NUM_WARPS)


def _fill_strided(buf):
    view = buf
    if buf.stride(-1) == 1 and buf.dtype not in (torch.uint8, torch.int8):
        try:
            view = buf.view(torch.uint8)
        except RuntimeError:
            view = buf
    shape = tuple(int(x) for x in view.shape)
    strides = tuple(int(x) for x in view.stride())
    num_elems = view.numel()
    grid = (triton.cdiv(num_elems, BLOCK_ELEMS_STRIDED),)
    _zero_strided[grid](
        view,
        num_elems,
        len(shape),
        shape,
        strides,
        BLOCK_ELEMS_STRIDED,
        num_warps=STRIDED_WARPS,
    )


def _fill_device(buf):
    if buf.is_contiguous():
        _fill_contiguous(buf)
    else:
        _fill_strided(buf)


def _fill_any(t):
    """清零一个 dense buffer（GCU 设备 / UVA pinned / CPU 回退）。"""
    if t.numel() == 0 or t.device.type == "meta":
        return
    if (DIRECT_DEVICE and t.device.type == "gcu") or t.device.type == "cuda":
        # GCU 设备内存可直接被 Triton kernel 寻址，无需中转 scratch。
        _fill_device(t)
    elif not SKIP_PINNED_CHECK and t.is_pinned():
        _fill_device(t)
    else:
        # pageable host 存储无法被设备 kernel 直接写；零值仍由设备 kernel 产生，
        # 这里只是把已算好的零 buffer 搬回 host 结果。
        scratch = torch.empty(tuple(t.shape), dtype=t.dtype, device="cuda")
        _fill_device(scratch)
        t.copy_(scratch)


def _zero_buffer(buf):
    if buf.layout == torch.sparse_csr or buf.layout == torch.sparse_csc:
        aux = (
            buf.crow_indices() if buf.layout == torch.sparse_csr else buf.ccol_indices()
        )
        _fill_any(aux)
        return
    if buf.layout != torch.strided:
        return
    _fill_any(buf)


def new_zeros(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    logger.debug("GEMS_GCU400 NEW_ZEROS")
    size_t = _as_size(size)

    if dtype is None:
        dtype = self.dtype
    if layout is None:
        layout = self.layout
    if device is None:
        device = self.device
    if pin_memory is None:
        try:
            pin_memory = bool(self.is_pinned())
        except Exception:  # noqa: BLE001 - sparse/meta sources have no pin state
            pin_memory = False

    if out is None:
        buf = _empty(size_t, dtype, layout, device, pin_memory)
    else:
        buf = out
        if tuple(buf.shape) != size_t:
            buf = buf.resize_(size_t)

    _zero_buffer(buf)
    return buf


def new_zeros_out(
    self, size, *, dtype=None, layout=None, device=None, pin_memory=None, out=None
):
    """``aten::new_zeros.out``：清零调用方的 buffer 并返回其别名。"""
    logger.debug("GEMS_GCU400 NEW_ZEROS_OUT")
    return new_zeros(
        self,
        size,
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=pin_memory,
        out=out,
    )
