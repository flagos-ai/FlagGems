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

from flag_gems.fused.FLA import chunk_gated_delta_rule_fwd
from flag_gems.fused.FLA.chunk_gated_delta_direct import (
    can_use_chunk_gated_delta_rule_direct,
    chunk_gated_delta_rule_direct_fwd,
)
from flag_gems.utils import libentry


@libentry()
@triton.jit
def _l2_normalize_last_dim_kernel(
    x,
    out,
    n_rows: tl.constexpr,
    H: tl.constexpr,
    K: tl.constexpr,
    stride_x_b: tl.constexpr,
    stride_x_t: tl.constexpr,
    stride_x_h: tl.constexpr,
    stride_x_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    mask = offs < K

    h = row % H
    row_bt = row // H
    t = row_bt % n_rows
    b = row_bt // n_rows
    x_base = x + b * stride_x_b + t * stride_x_t + h * stride_x_h
    values = tl.load(x_base + offs * stride_x_k, mask=mask, other=0.0).to(tl.float32)
    inv_norm = 1.0 / tl.maximum(tl.sqrt(tl.sum(values * values, axis=0)), 1e-6)
    tl.store(out + row * K + offs, values * inv_norm, mask=mask)


def _as_seq_first(
    x: torch.Tensor,
    *,
    name: str,
    head_first: bool,
    expected_ndim: int,
) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.ndim != expected_ndim:
        raise ValueError(f"{name} must be {expected_ndim}D, got shape {tuple(x.shape)}")
    if head_first:
        return x.transpose(1, 2)
    return x


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> None:
    B, T, Hg, K = q.shape
    Bk, Tk, Hk, Kk = k.shape
    Bv, Tv, H, V = v.shape

    tensors = {"k": k, "v": v, "beta": beta, "g": g}
    for name, tensor in tensors.items():
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same device as q")
        if tensor.dtype != q.dtype:
            raise ValueError(f"{name} must have the same dtype as q")

    if (Bk, Tk, Hk, Kk) != (B, T, Hg, K):
        raise ValueError(
            "q and k must have matching [B, T, Hq, K] shapes after layout conversion"
        )
    if (Bv, Tv) != (B, T):
        raise ValueError("v must have matching B and T dimensions with q/k")
    if H % Hg != 0:
        raise ValueError("the q/k head count must divide the v head count")
    if beta.shape != (B, T, H):
        raise ValueError(
            f"beta must have shape {(B, T, H)} after layout conversion, got {tuple(beta.shape)}"
        )
    if g.shape != (B, T, H):
        raise ValueError(
            f"g must have shape {(B, T, H)} after layout conversion, got {tuple(g.shape)}"
        )
    if cu_seqlens is not None:
        if not isinstance(cu_seqlens, torch.Tensor):
            raise TypeError("cu_seqlens must be a torch.Tensor")
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D tensor")
        if cu_seqlens.dtype != torch.long:
            raise ValueError("cu_seqlens must have dtype torch.long")
        if cu_seqlens.device != q.device:
            raise ValueError("cu_seqlens must be on the same device as q")
        if B != 1:
            raise ValueError("cu_seqlens packed varlen inputs must use B=1")

    if initial_state is not None:
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        if initial_state.dtype != q.dtype:
            raise ValueError("initial_state must have the same dtype as q")
        expected_n = B if cu_seqlens is None else cu_seqlens.numel() - 1
        expected_shape = (expected_n, H, K, V)
        if initial_state.shape != expected_shape:
            raise ValueError(
                f"initial_state must have shape {expected_shape}, got {tuple(initial_state.shape)}"
            )


def _direct_contiguous(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _l2_normalize_last_dim(x: torch.Tensor) -> torch.Tensor:
    B, T, H, K = x.shape
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    block_k = triton.next_power_of_2(K)
    _l2_normalize_last_dim_kernel[(B * T * H,)](
        x=x,
        out=out,
        n_rows=T,
        H=H,
        K=K,
        stride_x_b=x.stride(0),
        stride_x_t=x.stride(1),
        stride_x_h=x.stride(2),
        stride_x_k=x.stride(3),
        BLOCK_K=block_k,
    )
    return out


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    BT: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = True,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Public wrapper for the chunk gated delta rule forward operator.

    Inputs follow common FLA layouts:
    - ``head_first=True``: q/k/v are ``[B, H, T, D]`` and beta/g are ``[B, H, T]``.
    - ``head_first=False``: q/k/v are ``[B, T, H, D]`` and beta/g are ``[B, T, H]``.

    q/k may use fewer heads than v when the q/k head count divides the v head count.
    """
    if BT != 64:
        raise ValueError("chunk_gated_delta_rule currently supports only BT=64")

    q_seq = _as_seq_first(q, name="q", head_first=head_first, expected_ndim=4)
    k_seq = _as_seq_first(k, name="k", head_first=head_first, expected_ndim=4)
    v_seq = _as_seq_first(v, name="v", head_first=head_first, expected_ndim=4)
    beta_seq = _as_seq_first(beta, name="beta", head_first=head_first, expected_ndim=3)
    g_seq = _as_seq_first(g, name="g", head_first=head_first, expected_ndim=3)

    _validate_inputs(q_seq, k_seq, v_seq, beta_seq, g_seq, initial_state, cu_seqlens)

    if scale is None:
        scale = k_seq.shape[-1] ** -0.5

    B, T, Hg, K = q_seq.shape
    H, V = v_seq.shape[2], v_seq.shape[3]
    if (
        initial_state is None
        and cu_seqlens is None
        and T <= 128
        and K <= 128
        and V <= 128
        and H % Hg == 0
    ):
        q_direct = _direct_contiguous(q_seq)
        k_direct = _direct_contiguous(k_seq)
        v_direct = _direct_contiguous(v_seq)
        g_direct = _direct_contiguous(g_seq)
        beta_direct = _direct_contiguous(beta_seq)
        if can_use_chunk_gated_delta_rule_direct(
            q=q_direct,
            k=k_direct,
            v=v_direct,
            g=g_direct,
            beta=beta_direct,
            initial_state=None,
            cu_seqlens=None,
        ):
            o, final_state = chunk_gated_delta_rule_direct_fwd(
                q=q_direct,
                k=k_direct,
                v=v_direct,
                g=g_direct,
                beta=beta_direct,
                scale=float(scale),
                initial_state=None,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            if head_first:
                o = o.transpose(1, 2)
            return o, final_state

    if use_qk_l2norm_in_kernel:
        q_seq = _l2_normalize_last_dim(q_seq)
        k_seq = _l2_normalize_last_dim(k_seq)

    _, o, _, final_state, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q_seq,
        k=k_seq,
        v=v_seq,
        g=g_seq,
        beta=beta_seq,
        scale=float(scale),
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    if head_first:
        o = o.transpose(1, 2)
    return o, final_state
