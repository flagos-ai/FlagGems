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

from typing import Tuple

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit(do_not_specialize=["eps"])
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    task = tl.program_id(1)

    if task == 0:
        size = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        row_out = q_out_ptr + token_idx * q_out_stride
        weight_ptr = q_weight_ptr
    else:
        size = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        row_out = kv_out_ptr + token_idx * kv_out_stride
        weight_ptr = kv_weight_ptr

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < size
    x = tl.load(row_in + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / size
    rrms = tl.rsqrt(var + eps)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + offs, y.to(row_out.dtype.element_ty), mask=mask)


def fused_q_kv_rmsnorm(
    qr: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert qr.ndim == 2 and kv.ndim == 2
    assert qr.shape[0] == kv.shape[0]
    assert qr.stride(-1) == 1 and kv.stride(-1) == 1
    assert q_weight.is_contiguous() and kv_weight.is_contiguous()

    q_size = qr.shape[1]
    kv_size = kv.shape[1]
    num_tokens = qr.shape[0]
    qr_out = torch.empty_like(qr)
    kv_out = torch.empty_like(kv)
    if num_tokens == 0:
        return qr_out, kv_out

    block_size = triton.next_power_of_2(max(q_size, kv_size))
    with torch_device_fn.device(qr.device):
        _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
            qr,
            qr_out,
            q_weight,
            qr.stride(0),
            qr_out.stride(0),
            kv,
            kv_out,
            kv_weight,
            kv.stride(0),
            kv_out.stride(0),
            eps,
            Q_SIZE=q_size,
            KV_SIZE=kv_size,
            BLOCK_SIZE=block_size,
        )
    return qr_out, kv_out


__all__ = ["fused_q_kv_rmsnorm"]
