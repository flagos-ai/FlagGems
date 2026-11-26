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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _mm_silu_mul_kernel(
    A, B, C, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    gate_ptrs = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn
    up_ptrs = gate_ptrs + N * stride_bn
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
        gate_acc = tl.dot(a, gate, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(a, up, up_acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        gate_ptrs += BLOCK_K * stride_bk
        up_ptrs += BLOCK_K * stride_bk
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(c_ptrs, result.to(C.dtype.element_ty), mask=mask)


@libentry()
@triton.jit
def _tile_packed_silu_mul_kernel(
    A, GATE, UP, C, M, N, K,
    stride_am, stride_ak, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rk = tl.arange(0, BLOCK_K)
    rn_local = tl.arange(0, BLOCK_N)
    tile_offsets = rk[:, None] + rn_local[None, :] * BLOCK_K
    a_ptrs = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    gate_ptrs = GATE + pid_n * BLOCK_K * BLOCK_N + tile_offsets
    up_ptrs = UP + pid_n * BLOCK_K * BLOCK_N + tile_offsets
    gate_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    up_acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        gate = tl.load(gate_ptrs)
        up = tl.load(up_ptrs)
        gate_acc = tl.dot(a, gate, gate_acc, out_dtype=tl.float32)
        up_acc = tl.dot(a, up, up_acc, out_dtype=tl.float32)
        a_ptrs += BLOCK_K * stride_ak
        gate_ptrs += grid_n * BLOCK_K * BLOCK_N
        up_ptrs += grid_n * BLOCK_K * BLOCK_N
    result = gate_acc * tl.sigmoid(gate_acc) * up_acc
    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(c_ptrs, result.to(C.dtype.element_ty), mask=mask)


_TILE_PACKED = {}


def _tile_pack_slice(x):
    k, n = x.shape
    return x.contiguous().view(k // 32, 32, n // 128, 128).permute(
        0, 2, 3, 1
    ).contiguous()


def _tile_packed_weights(b):
    key = (b.data_ptr(), b.device, b.dtype, b.shape, b.stride())
    if key not in _TILE_PACKED:
        n = b.shape[1] // 2
        gate = _tile_pack_slice(b[:, :n])
        up = _tile_pack_slice(b[:, n:])
        _TILE_PACKED[key] = (b, gate, up)
    return _TILE_PACKED[key][1:]


def mm_silu_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    assert b.shape[0] == K and b.shape[1] % 2 == 0
    N = b.shape[1] // 2
    out = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M <= 128:
        block_m, warps, stages = 64, 8, 2
    else:
        block_m, warps, stages = 256, 8, 2
    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, 128),)
    with torch_device_fn.device(a.device):
        if K % 32 == 0 and N % 128 == 0:
            gate, up = _tile_packed_weights(b)
            _tile_packed_silu_mul_kernel[grid](
                a, gate, up, out, M, N, K,
                a.stride(0), a.stride(1), out.stride(0), out.stride(1),
                BLOCK_M=block_m, BLOCK_N=128, BLOCK_K=32,
                num_warps=warps, num_stages=stages,
            )
        else:
            _mm_silu_mul_kernel[grid](
                a, b, out, M, N, K,
                a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                out.stride(0), out.stride(1),
                BLOCK_M=block_m, BLOCK_N=128, BLOCK_K=32,
                num_warps=warps, num_stages=stages,
            )
    return out
