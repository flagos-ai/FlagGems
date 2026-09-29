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

logger = logging.getLogger(__name__)


def _ormqr_apply_kernel(
    input_ptr,
    tau_ptr,
    other_ptr,
    out_ptr,
    m,
    n,
    k,
    k2,
    input_batch_stride,
    tau_batch_stride,
    input_row_stride,
    input_col_stride,
    other_batch_stride,
    other_row_stride,
    other_col_stride,
    out_batch_stride,
    out_row_stride,
    out_col_stride,
    FORWARD: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    M_BLOCK: tl.constexpr,
    BC: tl.constexpr,
):
    """Apply the product of Householder reflectors stored in packed geqrf form.

    For each reflector i (i = 0..k-1, k = min(m, n)):
      H_i = I - tau_i * v_i v_i^T,  v_i[i] = 1,  v_i[j] = input[j, i] for j > i.
    The kernel applies op(Q), Q = H_0 ... H_{k-1}, along the m-axis of `other`:
      FORWARD=True  -> apply H_0, H_1, ..., H_{k-1}   (op(Q) = Q^T on the left)
      FORWARD=False -> apply H_{k-1}, ..., H_0        (op(Q) = Q on the left)

    TRANSPOSED=False: other/out have logical shape (B, m, k2) (left multiply);
      tile (M_BLOCK, BC): rows = m-axis, cols = k2-block.
    TRANSPOSED=True: other/out have logical shape (B, k2, m) (right multiply);
      tile (BC, M_BLOCK): rows = m-axis (last), cols = k2-block (first).
    """
    pid_c = tl.program_id(0)
    pid_b = tl.program_id(1)

    rows = tl.arange(0, M_BLOCK)
    row_mask = rows < m
    cols = pid_c * BC + tl.arange(0, BC)
    col_mask = cols < k2

    in_base = input_ptr + pid_b * input_batch_stride
    ta_base = tau_ptr + pid_b * tau_batch_stride
    ot_base = other_ptr + pid_b * other_batch_stride
    ou_base = out_ptr + pid_b * out_batch_stride

    if TRANSPOSED:
        offs = cols[:, None] * other_row_stride + rows[None, :] * other_col_stride
        tmask = col_mask[:, None] & row_mask[None, :]
        tile = tl.load(ot_base + offs, mask=tmask, other=0.0)
        for ii in tl.range(0, k):
            i = ii if FORWARD else (k - 1 - ii)
            v = tl.load(
                in_base + rows * input_row_stride + i * input_col_stride,
                mask=(rows > i) & row_mask,
                other=0.0,
            )
            v = tl.where(rows == i, 1.0, v)
            s = tl.sum(v[None, :] * tile, axis=1)
            tau_i = tl.load(ta_base + i)
            tile = tile - tau_i * s[:, None] * v[None, :]
        tl.store(ou_base + offs, tile, mask=tmask)
    else:
        offs = rows[:, None] * other_row_stride + cols[None, :] * other_col_stride
        tmask = row_mask[:, None] & col_mask[None, :]
        tile = tl.load(ot_base + offs, mask=tmask, other=0.0)
        for ii in tl.range(0, k):
            i = ii if FORWARD else (k - 1 - ii)
            v = tl.load(
                in_base + rows * input_row_stride + i * input_col_stride,
                mask=(rows > i) & row_mask,
                other=0.0,
            )
            v = tl.where(rows == i, 1.0, v)
            s = tl.sum(v[:, None] * tile, axis=0)
            tau_i = tl.load(ta_base + i)
            tile = tile - tau_i * v[:, None] * s[None, :]
        tl.store(ou_base + offs, tile, mask=tmask)


def _launch(input, tau, other, out, m, n, k, k2, forward, transposed, batch):
    M_BLOCK = triton.next_power_of_2(m)
    item_size = input.element_size()
    # Two-sided BC policy on the direct reflector kernel:
    #  - upper bound from the register budget: larger BC means fewer programs
    #    re-gathering the same strided Householder columns (L2 traffic scales
    #    as k2/bc);
    #  - upper bound k2//16 keeps at least ~16 programs across the k2 axis so
    #    small shapes do not collapse to grid=1;
    #  - floor 512//M_BLOCK avoids pathological tiny tiles.
    bc_reg = 131072 // (M_BLOCK * item_size)
    bc = min(bc_reg, k2 // 16)
    bc = max(bc, 512 // M_BLOCK)
    bc = min(bc, k2)
    # fp32 with m <= 2048: bc=32 halved the program count on [1024,1024] and
    # regressed it 2.40ms -> 3.09ms vs the round-1 bc=16 (64 programs); cap 16.
    if item_size == 4 and M_BLOCK <= 2048:
        bc = min(bc, 16)
    # Small squares (m <= 512): bc=4 with 8 warps (grid = k2/4 = 64-128 programs)
    # beat larger bc in the config sweep for [128,128] and [256,256].
    if item_size == 4 and M_BLOCK <= 512:
        bc = min(bc, 4)
    # fp64 m == 1024: bc=4/nw=4 (grid 256) beat bc=16/nw=8 in the sweep.
    if item_size == 8 and 256 <= M_BLOCK <= 1024 and k2 <= 4096:
        bc = min(bc, 4)
    # EXPERIMENT: fp64 m == 4096 with k2 <= 4096: bc=8 (grid 512 instead of
    # 1024) halves the v-gather program redundancy; test if [4096,4096] fp64 is
    # partially L2-bound. (Harness: 760ms vs 347ms current — expected worse.)
    if item_size == 8 and M_BLOCK == 4096 and k2 <= 4096:
        bc = min(bc, 8)
    bc = max(1, triton.next_power_of_2(bc))
    # Wide-other workloads: prefer a register-safe bc with 4 warps over the
    # generic bc with 8 warps (fewer spills, fewer programs re-gathering v).
    wide64 = (item_size == 8) and (k2 >= 8192)
    wide32 = (item_size == 4) and (k2 >= 8192)
    if wide64:
        bc = min(bc, 8)
    if wide32:
        bc = min(bc, 16)
    grid = (triton.cdiv(k2, bc), batch)
    tile_elems = M_BLOCK * bc
    # Device limit: 512 threads/program (warp size 64) -> at most 8 warps.
    small32 = (item_size == 4) and (M_BLOCK <= 512) and (bc <= 4) and (k2 <= 4096)
    mid64 = (item_size == 8) and (256 <= M_BLOCK <= 1024) and (k2 <= 4096) and (bc <= 4)
    if wide64 or wide32:
        num_warps = 4
    elif small32:
        num_warps = 8
    elif mid64:
        num_warps = 4
    else:
        num_warps = 8 if tile_elems > 4096 else 4
    _ormqr_apply_kernel[grid](
        input,
        tau,
        other,
        out,
        m,
        n,
        k,
        k2,
        m * n,
        k,
        input.stride(-2),
        input.stride(-1),
        other.stride(0) if other.ndim == 3 else m * k2,
        other.stride(-2),
        other.stride(-1),
        out.stride(0) if out.ndim == 3 else m * k2,
        out.stride(-2),
        out.stride(-1),
        FORWARD=forward,
        TRANSPOSED=transposed,
        M_BLOCK=M_BLOCK,
        BC=bc,
        num_warps=num_warps,
    )


def ormqr(input, tau, other, left=True, transpose=False):
    """ormqr: apply the product of Householder reflectors (packed geqrf form in
    `input` with scalars `tau`) to `other`.

    Verified semantics (torch.ormqr):
      left=True : out = Q @ other        (transpose=False)
                  out = Q^T @ other      (transpose=True)
      left=False: out = other @ Q        (transpose=False)
                  out = other @ Q^T      (transpose=True)
    Q = H_0 H_1 ... H_{k-1}, H_i = I - tau_i v_i v_i^T, v_i[i] = 1,
    v_i[j] = input[j, i] for j > i, k = min(m, n).
    """
    m, n = input.shape[-2], input.shape[-1]
    k = min(m, n)

    if input.ndim > 2:
        batch = 1
        for s in input.shape[:-2]:
            batch *= s
        input = input.reshape(batch, m, n)
        tau = tau.reshape(batch, k)
    else:
        batch = 1

    orig_shape = other.shape
    if left:
        k2 = other.shape[-1]
        if other.ndim > 2:
            other = other.reshape(batch, m, k2)
        out = torch.empty(orig_shape, dtype=other.dtype, device=other.device)
        out_v = out
        if other.ndim > 2:
            out_v = out.reshape(batch, m, k2)
        forward = transpose
        transposed = False
    else:
        k2 = other.shape[-2]
        if other.ndim > 2:
            other = other.reshape(batch, k2, m)
        out = torch.empty(orig_shape, dtype=other.dtype, device=other.device)
        out_v = out
        if other.ndim > 2:
            out_v = out.reshape(batch, k2, m)
        forward = not transpose
        transposed = True

    _launch(input, tau, other, out_v, m, n, k, k2, forward, transposed, batch)
    return out
