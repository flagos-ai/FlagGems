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

from flag_gems.ops._foreach_binary import (
    _foreach_addcdiv__Scalar as default_foreach_addcdiv__Scalar,
)
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# Moore Threads hardware does not support fp64.
_SUPPORTED_DTYPES = {torch.float16, torch.bfloat16, torch.float32}

_KMAX = 16
_BLOCK = 1024
_NUM_WARPS = 4
_BLOCK_MASK = _BLOCK - 1
# Large entries: fewer, fatter CTAs (4x fewer blocks) with more elements in
# flight per thread, which is the usual win for a pure 4-stream elementwise
# kernel; tiny entries keep the small block to stay launch-friendly.
_BLOCK_BIG = 4096
_BLOCK_BIG16 = 8192
_WARPS_BIG = 8
_BIG_N = 65536


def _config(n, itemsize=4):
    if n >= _BIG_N:
        # Match bytes-in-flight per thread across dtypes: 16-bit elements get
        # twice as many per block, so every lane still holds 64 B per stream.
        return (_BLOCK_BIG16 if itemsize <= 2 else _BLOCK_BIG), _WARPS_BIG
    return _BLOCK, _NUM_WARPS


# (dtype, numel, n_entries, plain) -> (CompiledKernel runner or None, arg mode).
# Only compiled-kernel handles are cached: no tensor values, no input-dependent
# intermediates.  A None runner means "always take the regular launch path".
_FAST = {}


@triton.jit
def _addcdiv_entry_kernel(
    s_ptr,
    a_ptr,
    b_ptr,
    value,
    n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements

    s = tl.load(s_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=1.0).to(tl.float32)

    out = s + value * (a / b)
    tl.store(s_ptr + offs, out, mask=mask)


@triton.jit
def _addcdiv_fused_kernel(
    s0,
    a0,
    b0,
    s1,
    a1,
    b1,
    s2,
    a2,
    b2,
    s3,
    a3,
    b3,
    s4,
    a4,
    b4,
    s5,
    a5,
    b5,
    s6,
    a6,
    b6,
    s7,
    a7,
    b7,
    s8,
    a8,
    b8,
    s9,
    a9,
    b9,
    s10,
    a10,
    b10,
    s11,
    a11,
    b11,
    s12,
    a12,
    b12,
    s13,
    a13,
    b13,
    s14,
    a14,
    b14,
    s15,
    a15,
    b15,
    value,
    n,
    BLOCK: tl.constexpr,
):
    """One launch over up to 16 same-dtype, same-numel list entries.

    program_id(1) selects the entry, so the selected base pointer stays a real
    typed pointer argument (16-byte vectorized accesses); an address loaded from
    a metadata table would be scalarized on this backend.
    """
    pid = tl.program_id(0)
    idx = tl.program_id(1)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n

    sp = s0
    sp = tl.where(idx == 1, s1, sp)
    sp = tl.where(idx == 2, s2, sp)
    sp = tl.where(idx == 3, s3, sp)
    sp = tl.where(idx == 4, s4, sp)
    sp = tl.where(idx == 5, s5, sp)
    sp = tl.where(idx == 6, s6, sp)
    sp = tl.where(idx == 7, s7, sp)
    sp = tl.where(idx == 8, s8, sp)
    sp = tl.where(idx == 9, s9, sp)
    sp = tl.where(idx == 10, s10, sp)
    sp = tl.where(idx == 11, s11, sp)
    sp = tl.where(idx == 12, s12, sp)
    sp = tl.where(idx == 13, s13, sp)
    sp = tl.where(idx == 14, s14, sp)
    sp = tl.where(idx == 15, s15, sp)

    ap = a0
    ap = tl.where(idx == 1, a1, ap)
    ap = tl.where(idx == 2, a2, ap)
    ap = tl.where(idx == 3, a3, ap)
    ap = tl.where(idx == 4, a4, ap)
    ap = tl.where(idx == 5, a5, ap)
    ap = tl.where(idx == 6, a6, ap)
    ap = tl.where(idx == 7, a7, ap)
    ap = tl.where(idx == 8, a8, ap)
    ap = tl.where(idx == 9, a9, ap)
    ap = tl.where(idx == 10, a10, ap)
    ap = tl.where(idx == 11, a11, ap)
    ap = tl.where(idx == 12, a12, ap)
    ap = tl.where(idx == 13, a13, ap)
    ap = tl.where(idx == 14, a14, ap)
    ap = tl.where(idx == 15, a15, ap)

    bp = b0
    bp = tl.where(idx == 1, b1, bp)
    bp = tl.where(idx == 2, b2, bp)
    bp = tl.where(idx == 3, b3, bp)
    bp = tl.where(idx == 4, b4, bp)
    bp = tl.where(idx == 5, b5, bp)
    bp = tl.where(idx == 6, b6, bp)
    bp = tl.where(idx == 7, b7, bp)
    bp = tl.where(idx == 8, b8, bp)
    bp = tl.where(idx == 9, b9, bp)
    bp = tl.where(idx == 10, b10, bp)
    bp = tl.where(idx == 11, b11, bp)
    bp = tl.where(idx == 12, b12, bp)
    bp = tl.where(idx == 13, b13, bp)
    bp = tl.where(idx == 14, b14, bp)
    bp = tl.where(idx == 15, b15, bp)

    s = tl.load(sp + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(ap + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bp + offs, mask=mask, other=1.0).to(tl.float32)

    out = s + value * (a / b)
    tl.store(sp + offs, out, mask=mask)


def _launch_fused(chunk, n, v, plain):
    k = len(chunk)
    pad = chunk[0]
    if k == _KMAX:
        cols = chunk
    else:
        cols = list(chunk)
        cols.extend([pad] * (_KMAX - k))

    args = [t for triple in cols for t in triple]
    args.append(v)
    args.append(n)

    block, warps = _config(n, chunk[0][0].element_size())
    blocks = (n + block - 1) // block
    if plain:
        key = (chunk[0][0].dtype, n, k)
        ent = _FAST.get(key)
        if ent is not None:
            runner, mode = ent
            if runner is not None:
                try:
                    if mode == 0:
                        runner(*args)
                    else:
                        runner(*args, block)
                    return
                except TypeError:
                    if mode == 0:
                        try:
                            runner(*args, block)
                            _FAST[key] = (runner, 1)
                            return
                        except TypeError:
                            pass
                    _FAST[key] = (None, 0)
                except Exception:
                    _FAST[key] = (None, 0)

    ck = _addcdiv_fused_kernel[(blocks, k)](*args, BLOCK=block, num_warps=warps)
    if plain and key not in _FAST:
        try:
            _FAST[key] = (ck[(blocks, k, 1)], 0)
        except Exception:
            _FAST[key] = (None, 0)


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def _list_supported(seq):
    """A foreach call is specialized only if every tensor in it qualifies."""
    if seq is None:
        return True
    if isinstance(seq, torch.Tensor):
        return _use_triton_kernel(seq)
    for t in seq:
        if isinstance(t, torch.Tensor) and not _use_triton_kernel(t):
            return False
    return True


def _foreach_addcdiv__Scalar(self, tensor1, tensor2, value=1):
    logger.debug("GEMS_MTHREADS _FOREACH_ADDCDIV__SCALAR")

    if not _list_supported(self):
        return default_foreach_addcdiv__Scalar(self, tensor1, tensor2, value)

    with torch_device_fn.device(
        self[0].device
        if isinstance(self, (list, tuple)) and self
        else torch.device("musa")
    ):
        return _launch_foreach_addcdiv__Scalar(self, tensor1, tensor2, value)


def _launch_foreach_addcdiv__Scalar(self, tensor1, tensor2, value=1):
    """_foreach_addcdiv_.Scalar: self_i += value * (tensor1_i / tensor2_i), in place."""
    if not self:
        return None

    v = float(value)
    groups = {}
    setdefault = groups.setdefault
    plain = True
    for s, a, b in zip(self, tensor1, tensor2):
        n = s.numel()
        if n == 0:
            continue
        if s._base is not None or a._base is not None or b._base is not None:
            # Views can carry an unaligned data pointer; keep them on the
            # regularly specialized launch path.
            plain = False
        setdefault((s.dtype, n), []).append((s, a, b))

    for (_dtype, n), ents in groups.items():
        if len(ents) > 1:
            for i in range(0, len(ents), _KMAX):
                _launch_fused(ents[i : i + _KMAX], n, v, plain)
        else:
            s, a, b = ents[0]
            block, warps = _config(n, s.element_size())
            _addcdiv_entry_kernel[((n + block - 1) // block,)](
                s, a, b, v, n, BLOCK=block, num_warps=warps
            )
    return None
