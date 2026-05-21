import inspect
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

import flag_gems


os.environ.setdefault("VLLM_CONFIGURE_LOGGING", "0")


try:
    from flag_gems.runtime.backend._nvidia.hopper.ops.flash_api_v3 import (
        is_fa3_supported as _backend_is_fa3_supported,
    )
except ImportError:
    _backend_is_fa3_supported = None
except Exception:
    _backend_is_fa3_supported = None


try:
    from vllm.vllm_flash_attn.flash_attn_interface import (
        flash_attn_varlen_func as vllm_fa_varlen,
    )

    HAS_VLLM_FA = True
    _vllm_fa_params = set(inspect.signature(vllm_fa_varlen).parameters.keys())
    VLLM_FA_HAS_FA_VERSION = "fa_version" in _vllm_fa_params
    VLLM_FA_HAS_BLOCK_TABLE = "block_table" in _vllm_fa_params
    VLLM_FA_HAS_SEQUSED_K = "seqused_k" in _vllm_fa_params
except ImportError:
    vllm_fa_varlen = None
    HAS_VLLM_FA = False
    VLLM_FA_HAS_FA_VERSION = False
    VLLM_FA_HAS_BLOCK_TABLE = False
    VLLM_FA_HAS_SEQUSED_K = False
except Exception:
    vllm_fa_varlen = None
    HAS_VLLM_FA = False
    VLLM_FA_HAS_FA_VERSION = False
    VLLM_FA_HAS_BLOCK_TABLE = False
    VLLM_FA_HAS_SEQUSED_K = False


@dataclass(frozen=True)
class Shape:
    name: str
    seq_lens: List[Tuple[int, int]]
    nh_q: int
    nh_k: int
    head_dim: int
    causal: bool
    paged: bool = False
    block_size: int = 16
    overcommit: float = 1.5


@dataclass
class Tensors:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: Optional[torch.Tensor]
    seqused_k: torch.Tensor
    block_table: Optional[torch.Tensor]
    max_seqlen_q: int
    max_seqlen_k: int


def is_fa3_supported() -> bool:
    if _backend_is_fa3_supported is not None:
        return _backend_is_fa3_supported()
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 9


def accuracy_shapes() -> List[Shape]:
    return [
        Shape("dense_prefill_mha", [(64, 64), (48, 48)], 4, 4, 64, False),
        Shape("dense_causal_gqa", [(64, 64), (1, 128), (17, 96)], 8, 2, 128, True),
        Shape("dense_decode_gqa", [(1, 128), (1, 256), (1, 384)], 8, 2, 128, True),
        Shape(
            "paged_decode_gqa",
            [(1, 128), (1, 192), (1, 256), (1, 320)],
            8,
            2,
            128,
            True,
            paged=True,
            block_size=16,
        ),
    ]


def benchmark_shapes() -> List[Shape]:
    return [
        Shape("prefill_b4_s2k_d128_mha", [(2048, 2048)] * 4, 32, 32, 128, True),
        Shape("prefill_b4_s4k_d128_mha", [(4096, 4096)] * 4, 32, 32, 128, True),
        Shape("prefill_b4_s4k_d128_gqa4", [(4096, 4096)] * 4, 32, 8, 128, True),
        Shape("decode_b16_kv1k_d128_gqa4", [(1, 1024)] * 16, 32, 8, 128, True),
        Shape(
            "varlen_mixed_d128_gqa4",
            [(2048, 2048), (1, 4096), (1024, 1024), (1, 8192)],
            32,
            8,
            128,
            True,
        ),
        Shape(
            "paged_decode_b16_kvmix_bs16_d128_gqa4",
            [(1, 1024 + 256 * i) for i in range(16)],
            32,
            8,
            128,
            True,
            paged=True,
            block_size=16,
        ),
    ]


def make_varlen(shape: Shape, dtype: torch.dtype, device: str, seed: int = 0) -> Tensors:
    if shape.paged:
        return _make_paged_varlen(shape, dtype, device, seed)
    return _make_dense_varlen(shape, dtype, device, seed)


def _make_dense_varlen(
    shape: Shape, dtype: torch.dtype, device: str, seed: int
) -> Tensors:
    gen = torch.Generator(device=device).manual_seed(seed)
    cu_q = [0]
    cu_k = [0]
    for q_len, k_len in shape.seq_lens:
        cu_q.append(cu_q[-1] + q_len)
        cu_k.append(cu_k[-1] + k_len)

    q = torch.randn(
        (cu_q[-1], shape.nh_q, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    k = torch.randn(
        (cu_k[-1], shape.nh_k, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    v = torch.randn(
        (cu_k[-1], shape.nh_k, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5

    return Tensors(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=device),
        seqused_k=torch.tensor(
            [k_len for _, k_len in shape.seq_lens],
            dtype=torch.int32,
            device=device,
        ),
        block_table=None,
        max_seqlen_q=max(q_len for q_len, _ in shape.seq_lens),
        max_seqlen_k=max(k_len for _, k_len in shape.seq_lens),
    )


def _make_paged_varlen(
    shape: Shape, dtype: torch.dtype, device: str, seed: int
) -> Tensors:
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed + 1)
    block_size = shape.block_size

    cu_q = [0]
    for q_len, _ in shape.seq_lens:
        cu_q.append(cu_q[-1] + q_len)
    q = torch.randn(
        (cu_q[-1], shape.nh_q, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5

    blocks_per_req = [
        (k_len + block_size - 1) // block_size for _, k_len in shape.seq_lens
    ]
    max_blocks_per_req = max(blocks_per_req)
    total_virtual_blocks = sum(blocks_per_req)
    num_physical_blocks = max(1, int(total_virtual_blocks * shape.overcommit))

    perm = torch.randperm(num_physical_blocks, generator=cpu_gen)[:total_virtual_blocks]
    block_table = torch.zeros(
        (len(shape.seq_lens), max_blocks_per_req), dtype=torch.int32, device=device
    )
    offset = 0
    for req_idx, num_blocks in enumerate(blocks_per_req):
        block_table[req_idx, :num_blocks] = perm[
            offset : offset + num_blocks
        ].to(dtype=torch.int32, device=device)
        offset += num_blocks

    k_cache = torch.randn(
        (num_physical_blocks, block_size, shape.nh_k, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    v_cache = torch.randn(
        (num_physical_blocks, block_size, shape.nh_k, shape.head_dim),
        dtype=dtype,
        device=device,
        generator=gen,
    ) * 0.5
    seqused_k = torch.tensor(
        [k_len for _, k_len in shape.seq_lens], dtype=torch.int32, device=device
    )

    return Tensors(
        q=q,
        k=k_cache,
        v=v_cache,
        cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=device),
        cu_seqlens_k=None,
        seqused_k=seqused_k,
        block_table=block_table,
        max_seqlen_q=max(q_len for q_len, _ in shape.seq_lens),
        max_seqlen_k=max(k_len for _, k_len in shape.seq_lens),
    )


def run_flag_gems(tensors: Tensors, shape: Shape, fa_version: int = 3) -> torch.Tensor:
    kwargs = {
        "max_seqlen_q": tensors.max_seqlen_q,
        "cu_seqlens_q": tensors.cu_seqlens_q,
        "max_seqlen_k": tensors.max_seqlen_k,
        "softmax_scale": 1.0 / math.sqrt(shape.head_dim),
        "causal": shape.causal,
        "fa_version": fa_version,
    }
    if shape.paged:
        kwargs["seqused_k"] = tensors.seqused_k
        kwargs["block_table"] = tensors.block_table
    else:
        kwargs["cu_seqlens_k"] = tensors.cu_seqlens_k

    return flag_gems.flash_attn_varlen_func(tensors.q, tensors.k, tensors.v, **kwargs)


def run_vllm_fa(tensors: Tensors, shape: Shape, fa_version: int = 3) -> torch.Tensor:
    if not HAS_VLLM_FA:
        raise NotImplementedError("vLLM flash-attention is unavailable")

    extra = {
        "softmax_scale": 1.0 / math.sqrt(shape.head_dim),
        "causal": shape.causal,
    }
    if VLLM_FA_HAS_FA_VERSION:
        extra["fa_version"] = fa_version

    if shape.paged:
        if not VLLM_FA_HAS_BLOCK_TABLE:
            raise NotImplementedError("vLLM flash-attention lacks block_table support")
        if not VLLM_FA_HAS_SEQUSED_K:
            raise NotImplementedError("vLLM flash-attention lacks seqused_k support")
        out = vllm_fa_varlen(
            tensors.q,
            tensors.k,
            tensors.v,
            max_seqlen_q=tensors.max_seqlen_q,
            cu_seqlens_q=tensors.cu_seqlens_q,
            max_seqlen_k=tensors.max_seqlen_k,
            seqused_k=tensors.seqused_k,
            block_table=tensors.block_table,
            **extra,
        )
    else:
        out = vllm_fa_varlen(
            tensors.q,
            tensors.k,
            tensors.v,
            max_seqlen_q=tensors.max_seqlen_q,
            cu_seqlens_q=tensors.cu_seqlens_q,
            max_seqlen_k=tensors.max_seqlen_k,
            cu_seqlens_k=tensors.cu_seqlens_k,
            **extra,
        )
    return out[0] if isinstance(out, tuple) else out


def eager_reference(tensors: Tensors, shape: Shape) -> torch.Tensor:
    scale = 1.0 / math.sqrt(shape.head_dim)
    cu_q = tensors.cu_seqlens_q
    if shape.paged:
        seqs_k, seqs_v = _gather_paged_to_dense(
            tensors.k, tensors.v, tensors.block_table, tensors.seqused_k
        )
    else:
        seqs_k = []
        seqs_v = []
        for batch_idx in range(cu_q.numel() - 1):
            k_begin = tensors.cu_seqlens_k[batch_idx]
            k_end = tensors.cu_seqlens_k[batch_idx + 1]
            seqs_k.append(tensors.k[k_begin:k_end])
            seqs_v.append(tensors.v[k_begin:k_end])

    outputs = []
    for batch_idx in range(cu_q.numel() - 1):
        q_begin = cu_q[batch_idx]
        q_end = cu_q[batch_idx + 1]
        q_b = tensors.q[q_begin:q_end]
        k_b = seqs_k[batch_idx]
        v_b = seqs_v[batch_idx]
        q_len, num_q_heads, _ = q_b.shape
        k_len = k_b.size(0)

        if k_b.size(1) != num_q_heads:
            repeat = num_q_heads // k_b.size(1)
            k_b = k_b.repeat_interleave(repeat, dim=1)
            v_b = v_b.repeat_interleave(repeat, dim=1)

        q_heads = q_b.transpose(0, 1).float()
        k_heads = k_b.transpose(0, 1).float()
        v_heads = v_b.transpose(0, 1).float()
        scores = (q_heads @ k_heads.transpose(-1, -2)) * scale
        if shape.causal:
            mask = torch.ones(
                q_len, k_len, dtype=torch.bool, device=tensors.q.device
            ).tril(diagonal=k_len - q_len)
            scores = scores.masked_fill(~mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append((probs @ v_heads).transpose(0, 1).to(tensors.q.dtype))

    return torch.cat(outputs, dim=0)


def build_reference(
    tensors: Tensors,
    shape: Shape,
    fa_version: int = 3,
    prefer_vllm: bool = True,
) -> Tuple[torch.Tensor, str]:
    if prefer_vllm and HAS_VLLM_FA:
        try:
            with torch.inference_mode():
                return run_vllm_fa(tensors, shape, fa_version), "vllm_fa"
        except NotImplementedError:
            pass
    with torch.inference_mode():
        return eager_reference(tensors, shape), "eager_fp32"


def _gather_paged_to_dense(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seqused_k: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    block_size = k_cache.size(1)
    seqs_k = []
    seqs_v = []
    for req_idx in range(seqused_k.numel()):
        k_len = int(seqused_k[req_idx].item())
        num_blocks = (k_len + block_size - 1) // block_size
        block_ids = block_table[req_idx, :num_blocks].to(torch.long)
        k_flat = k_cache.index_select(0, block_ids).reshape(
            -1, k_cache.size(2), k_cache.size(3)
        )
        v_flat = v_cache.index_select(0, block_ids).reshape(
            -1, v_cache.size(2), v_cache.size(3)
        )
        seqs_k.append(k_flat[:k_len])
        seqs_v.append(v_flat[:k_len])
    return seqs_k, seqs_v


def tolerances(dtype: torch.dtype, max_k: int, ref_kind: str) -> Tuple[float, float]:
    if ref_kind == "vllm_fa":
        atol = 3e-3 if dtype == torch.bfloat16 else 1.5e-3
        rtol = 3e-3 if dtype == torch.bfloat16 else 1.5e-3
    else:
        atol = 1e-2 if dtype == torch.bfloat16 else 5e-3
        rtol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    if max_k >= 8192:
        atol *= 2.0
        rtol *= 2.0
    return atol, rtol


def attn_flops(shape: Shape) -> float:
    flops = 0.0
    for q_len, k_len in shape.seq_lens:
        flops += 4.0 * q_len * k_len * shape.head_dim * shape.nh_q
    if shape.causal:
        flops *= 0.5
    return flops


def output_tensor(out: torch.Tensor) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out


def max_mean_abs(out: torch.Tensor, ref: torch.Tensor) -> Tuple[float, float]:
    diff = (out.float() - ref.float()).abs()
    return diff.max().item(), diff.mean().item()


def dispatch_source() -> str:
    return inspect.getsourcefile(flag_gems.flash_attn_varlen_func) or ""


def dispatches_to_hopper() -> bool:
    return "runtime/backend/_nvidia/hopper/ops" in dispatch_source().replace("\\", "/")
