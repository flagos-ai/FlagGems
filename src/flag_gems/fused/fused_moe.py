# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from the vLLM project (https://github.com/vllm-project/vllm).
# Source files under vllm/model_executor/layers/:
#   fused_moe/fused_moe.py      – Triton kernels, dispatch, fused_experts_impl
#   fused_moe/activation.py     – MoEActivation enum, apply_moe_activation
#   fused_moe/utils.py          – _fp8_quantize, _int8_quantize, moe_kernel_quantize_input
#   fused_moe/config.py         – _get_config_dtype_str
#   quantization/utils/mxfp4_utils.py   – dequant_mxfp4
#   quantization/utils/mxfp6_utils.py   – dequant_mxfp6
#   quantization/utils/ocp_mx_utils.py  – OCP_MX_BLOCK_SIZE


import functools
import logging
import os
from enum import Enum
from typing import Any, Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import yaml

from flag_gems.fused.moe_align_block_size import moe_align_block_size
from flag_gems.fused.moe_sum import moe_sum
from flag_gems.runtime import device, torch_device_fn
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

# OCP MX quantization helpers (requires amd-quark)

OCP_MX_BLOCK_SIZE = 32


@functools.lru_cache(maxsize=1)
def get_embedded_moe_configs():
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "utils", "configs", "fused_moe_config.yaml"
    )
    if not os.path.exists(config_path):
        return {}, {}
    with open(config_path, "r") as f:
        # JSON keys are strings, values are dicts where keys are M and values are configs
        data = yaml.safe_load(f)

        fallback = data.get("_FALLBACK", {})

        # We need to convert the innermost keys (which are stringified integers for M) back to integers.
        # Ensure we map the lists back to config dicts.
        keys_order = [
            "BLOCK_SIZE_M",
            "BLOCK_SIZE_N",
            "BLOCK_SIZE_K",
            "GROUP_SIZE_M",
            "num_warps",
            "num_stages",
        ]
        parsed_data = {}
        for dev, configs in data.items():
            if dev == "_FALLBACK":
                continue
            parsed_data[dev] = {}
            for k, m_dict in configs.items():
                parsed_dict = {}
                for m, v in m_dict.items():
                    if isinstance(v, list):
                        parsed_dict[int(m)] = dict(zip(keys_order, v))
                    else:
                        parsed_dict[int(m)] = v
                parsed_data[dev][k] = parsed_dict

        return parsed_data, fallback


def dequant_mxfp4(
    x: torch.Tensor,
    scale: torch.Tensor,
    float_dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize MXFP4 tensor via quark.torch.kernel.mx.dq_mxfp4."""
    try:
        from quark.torch.kernel import mx
    except ImportError as err:
        raise ImportError("amd-quark is required for MX-FP4") from err

    return mx.dq_mxfp4(x, scale, float_dtype)


def dequant_mxfp6(
    x: torch.Tensor,
    scale: torch.Tensor,
    float_dtype: torch.dtype,
    quant_dtype: str,
) -> torch.Tensor:
    """Dequantize MXFP6 tensor via quark hw_emulation."""
    try:
        from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
            dequantize_fp4_fp6_per_group,
        )
        from quark.torch.utils.pack import create_pack_method
    except ImportError as err:
        raise ImportError("amd-quark is required for MX-FP6") from err

    pack_method = create_pack_method(None, dtype=quant_dtype)
    unpacked_x = pack_method.unpack(x, reorder=False)

    scale = 2 ** (scale.view(torch.uint8).to(torch.int16) - 127).to(float_dtype)

    return dequantize_fp4_fp6_per_group(
        unpacked_x,
        scale,
        axis=-1,
        group_size=OCP_MX_BLOCK_SIZE,
        quant_dtype=quant_dtype,
    ).to(float_dtype)


# Activation quantization helpers


@functools.lru_cache(maxsize=1)
def _get_device_name() -> str:
    """Return the normalised device name (spaces replaced by underscores).

    Matches the naming convention used by vLLM for its per-device config files.
    H800 falls back to H100_80GB_HBM3 (same SM 9.0 architecture).
    """
    try:
        name = torch_device_fn.get_device_name().replace(" ", "_")
    except AttributeError:
        name = device.name
    # Normalise the H200 product family to a single key, following vLLM.
    if "H200" in name.split("_"):
        name = "NVIDIA_H200"
    # H800 has the same SM 9.0 as H100; use H100 configs as fallback.
    embedded_configs, fallback_mapping = get_embedded_moe_configs()
    if name in embedded_configs:
        return name
    # Fallback mapping for devices whose tuning profiles are equivalent.
    fallback = fallback_mapping.get(name)
    if fallback and fallback in embedded_configs:
        logger.info("Device %s not in config table, falling back to %s", name, fallback)
        return fallback
    return name


def get_moe_configs(
    E: int,
    N: int,
    dtype: str | None,
    block_n: int | None = None,
    block_k: int | None = None,
) -> dict[int, Any] | None:
    """
    Return optimized configurations for the fused MoE kernel.

    Looks up pre-tuned configs from the embedded table (ported from vLLM)
    for the current GPU device. Returns None if no matching config is found.
    """
    device_name = _get_device_name()
    embedded_configs, _ = get_embedded_moe_configs()
    device_table = embedded_configs.get(device_name)
    if device_table is None:
        logger.warning(
            "No embedded MoE configs for device %s. Will use default config.",
            device_name,
        )
        return None

    _block_n = block_n if block_n else 0
    _block_k = block_k if block_k else 0
    key = f"{E},{N},{dtype},{_block_n},{_block_k}"
    configs = device_table.get(key)
    if configs is not None:
        logger.info("Using embedded MoE config for device=%s, key=%s", device_name, key)
        return configs
    logger.warning(
        "No embedded MoE config for device=%s, key=%s. Will use default config.",
        device_name,
        key,
    )
    return None


def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: str | None,
    M: int,
    E: int,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    override_config: Optional[dict[str, Any]] = None

    is_hopper = torch.cuda.is_available() and torch.cuda.get_device_capability() == (
        9,
        0,
    )
    if (
        is_hopper
        and dtype == "fp8_w8a8"
        and block_shape is not None
        and len(block_shape) == 2
    ):
        # Use heuristic config like hpc-ops
        avg_tokens_per_expert = M * top_k // E
        is_large_m = M >= 16384
        if avg_tokens_per_expert <= 16:
            block_size_m = 16
        elif avg_tokens_per_expert <= 32:
            block_size_m = 32
        elif avg_tokens_per_expert <= 64 or not is_large_m:
            block_size_m = 64
        else:
            block_size_m = 128

        config = {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 8 if (is_large_m and avg_tokens_per_expert > 16) else 1,
            "num_warps": 8 if (is_large_m and block_size_m > 32) else 4,
            "num_stages": 5 if is_large_m else 3,
            "SWAP_AB": False,
        }
        override_config = config

    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
        if dtype == "int4_w4a16":
            N = N * 2
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
        configs = get_moe_configs(E, N, dtype, block_n, block_k)

        if configs:
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype, block_shape)
    return config


def _get_config_quant_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    ocp_mx_scheme: str | None,
) -> None | torch.dtype | str:
    """Map quantization flags to the corresponding dtype."""
    if use_fp8_w8a8:
        return torch.float8_e4m3fn
    elif use_int8_w8a8:
        return torch.int8
    elif ocp_mx_scheme == "w_mxfp4_a_mxfp4":
        return "mxfp4"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e3m2", "w_mxfp6_e3m2_a_mxfp6_e3m2"}:
        return "mxfp6_e3m2"
    elif ocp_mx_scheme in {"w_mxfp4_a_mxfp6_e2m3", "w_mxfp6_e2m3_a_mxfp6_e2m3"}:
        return "mxfp6_e2m3"
    elif ocp_mx_scheme in {"w_mxfp4", "w_mxfp6_e3m2", "w_mxfp6_e2m3"}:
        return torch.bfloat16
    elif ocp_mx_scheme in {"w_mxfp4_a_fp8", "w_mxfp6_e3m2_a_fp8", "w_mxfp6_e2m3_a_fp8"}:
        return torch.float8_e4m3fn

    return None


def get_moe_wna16_block_config(
    config: dict[str, int],
    use_moe_wna16_cuda: bool,
    num_valid_tokens: int,
    size_k: int,
    size_n: int,
    num_experts: int,
    group_size: int,
    real_top_k: int,
    block_size_m: int,
):
    if "BLOCK_SIZE_N" in config and "BLOCK_SIZE_K" in config:
        return {}
    if not use_moe_wna16_cuda:
        if num_valid_tokens // real_top_k == 1:
            return {"BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64}
        else:
            return {"BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}
    else:
        block_size_n = 128
        block_size_k = 128
        if block_size_k <= group_size:
            block_size_k = group_size

        num_n_blocks = size_k // block_size_k
        num_k_blocks = size_n // block_size_k
        num_m_blocks = (
            num_valid_tokens + block_size_m - 1
        ) / block_size_m + num_experts
        if num_valid_tokens // real_top_k <= block_size_m:
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        num_blocks = num_m_blocks * num_n_blocks * num_k_blocks

        if size_k % 256 == 0 and num_blocks >= 256 and block_size_k < 256:
            block_size_k = 256
            num_blocks = num_blocks // (256 // block_size_k)

        if (
            num_m_blocks <= 16
            and size_k % (block_size_k * 2) == 0
            and size_k % (block_size_k * 2) == 0
            and block_size_k <= 512
            and num_blocks >= 512
        ):
            block_size_k = block_size_k * 2
            num_blocks = num_blocks // 2

        if num_blocks > 1024:
            block_size_n = 256
            num_n_blocks = num_n_blocks // 2
            num_blocks = num_blocks // 2

        if size_n <= 1024 and num_blocks >= 1024:
            block_size_n = 1024

        block_size_k = _ensure_block_size_k_divisible(size_k, block_size_k, group_size)

        return {"BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k}


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: str | None,
    block_shape: list[int] | None = None,
) -> dict[str, int]:
    """Default Triton config for fused MoE kernel.

    Heuristic selection aligned with vLLM v0.17.0 defaults, tuned on H20/H100.
    Key insight: for high-expert-count MoE (e.g. DeepSeek-V3 E=256), each
    expert sees very few tokens, so small BLOCK_SIZE_M (16) is critical.
    """
    if dtype == "fp8_w8a8" and block_shape is not None:
        config = {
            "BLOCK_SIZE_M": 16 if M <= 64 else 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 1 if M <= 16 else 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    else:
        # tokens_per_expert drives block_m: use M//E (not M*topk//E) to
        # estimate the actual per-expert token count after routing.
        tokens_per_expert = M // max(E, 1)

        if tokens_per_expert <= 2:
            block_m = 16
        elif tokens_per_expert <= 4:
            block_m = 32
        elif tokens_per_expert <= 16:
            block_m = 64
        else:
            block_m = 128

        # Tile sizing
        if N >= 4096:
            block_n = 128 if M <= 128 else 256
        elif N >= 1024:
            block_n = 64 if M <= 64 else 128
        else:
            block_n = 64 if M <= 64 else 128

        if dtype == "fp8_w8a8":
            block_k = 128
        elif M <= 64:
            block_k = 128
        else:
            block_k = 64

        if tokens_per_expert > 128:
            group_m = 16
        elif tokens_per_expert > 32:
            group_m = 8
        else:
            group_m = 1

        # Prefer 4 warps for small tiles; only use 8 for large M
        num_warps = 4 if M <= 128 else 8
        num_stages = 3

        smem_per_stage = (block_m * block_k + block_k * block_n) * 2
        while num_stages > 2 and smem_per_stage * num_stages > 200_000:
            num_stages -= 1

        config = {
            "BLOCK_SIZE_M": block_m,
            "BLOCK_SIZE_N": block_n,
            "BLOCK_SIZE_K": block_k,
            "GROUP_SIZE_M": group_m,
            "num_warps": num_warps,
            "num_stages": num_stages,
        }
    return config


def _get_config_dtype_str(
    dtype: Optional[torch.dtype] = None,
    use_fp8_w8a8: bool = False,
    use_fp8_w8a16: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
) -> str | None:
    """Return dtype string for kernel config lookup."""
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_fp8_w8a16:
        return "fp8_w8a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif ocp_mx_scheme is not None:
        return None
    elif dtype == torch.float:
        return "float32"
    return None


# MoE activation enum


class MoEActivation(Enum):
    """Activation functions for MoE layers."""

    # Gated: gate * activation(up), input [..., 2*d] -> output [..., d]
    SILU = "silu"
    GELU = "gelu"
    RELU2 = "relu2"
    SWIGLUOAI = "swigluoai"
    SWIGLUSTEP = "swiglustep"

    # Non-gated: input [..., d] -> output [..., d]
    SILU_NO_MUL = "silu_no_mul"
    GELU_NO_MUL = "gelu_no_mul"
    RELU2_NO_MUL = "relu2_no_mul"

    @property
    def is_gated(self) -> bool:
        return not self.value.endswith("_no_mul")

    def without_mul(self) -> "MoEActivation":
        """Return the non-gated variant."""
        _without_mul: dict[MoEActivation, MoEActivation] = {
            MoEActivation.SILU: MoEActivation.SILU_NO_MUL,
            MoEActivation.GELU: MoEActivation.GELU_NO_MUL,
            MoEActivation.RELU2: MoEActivation.RELU2_NO_MUL,
        }
        return _without_mul.get(self, self)

    @classmethod
    def from_str(cls, s: str) -> "MoEActivation":
        for member in cls:
            if member.value == s:
                return member
        valid = [m.value for m in cls]
        raise ValueError(f"Unknown MoE activation: {s!r}. Valid activations: {valid}")

    @staticmethod
    def adjust_N_for_activation(N: int, activation: "MoEActivation") -> int:
        """Return N for non-gated, N // 2 for gated activations."""
        return N if not activation.is_gated else N // 2


def apply_moe_activation(
    activation: MoEActivation,
    output: torch.Tensor,
    input: torch.Tensor,
) -> torch.Tensor:
    """Apply MoE activation (pure PyTorch / FlagGems Triton)."""
    assert input.dim() == 2, "Input must be 2D"
    assert output.dim() == 2, "Output must be 2D"
    if activation.is_gated:
        assert output.size(-1) * 2 == input.size(-1), (
            f"{activation.value} expects 2x ratio: "
            f"{output.size(-1) * 2} vs {input.size(-1)}"
        )
    else:
        assert output.size(-1) == input.size(-1), (
            f"{activation.value} expects equal sizes: "
            f"{output.size(-1)} vs {input.size(-1)}"
        )

    if activation in (MoEActivation.SILU, MoEActivation.SWIGLUOAI):
        N = output.size(-1)
        x, y = input[:, :N], input[:, N:]
        _silu_and_mul_kernel(x, y, out0=output)
    elif activation == MoEActivation.GELU:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(F.gelu(gate) * up)
    elif activation == MoEActivation.SWIGLUSTEP:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(torch.sigmoid(gate) * up)
    elif activation == MoEActivation.RELU2:
        N = output.size(-1)
        gate, up = input[:, :N], input[:, N:]
        output.copy_(F.relu(gate).square() * up)

    elif activation == MoEActivation.SILU_NO_MUL:
        output.copy_(F.silu(input))
    elif activation == MoEActivation.GELU_NO_MUL:
        output.copy_(F.gelu(input))
    elif activation == MoEActivation.RELU2_NO_MUL:
        F.relu(input, inplace=True)
        torch.square(input, out=output)
    else:
        raise ValueError(f"Unsupported FusedMoe activation: {activation}")

    return output


def _fp8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP8 E4M3 quantization: per-tensor, per-token, or block-wise."""
    fp8_dtype = torch.float8_e4m3fn
    finfo = torch.finfo(fp8_dtype)
    fp8_max = finfo.max
    fp8_min = finfo.min
    eps = 1e-10

    if block_shape is not None:
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        if A.ndim == 2 and A.stride(-1) == 1:
            from flag_gems.ops.per_token_group_quant_fp8 import (
                per_token_group_quant_fp8,
            )

            return per_token_group_quant_fp8(
                A,
                group_size=block_k,
                eps=eps,
                dtype=fp8_dtype,
                column_major_scales=False,
                scale_ue8m0=False,
            )
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = (
            A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        )
        scale = amax / fp8_max
        A_q = (A_groups.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / fp8_max
        min_scale = torch.tensor(
            1.0 / (fp8_max * 512.0), dtype=torch.float32, device=A.device
        )
        scale = scale.clamp(min=min_scale)
        A_q = (A_flat.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        if A_scale is not None:
            scale = (
                A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
            )
            A_q = (A.float() / scale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, A_scale
        else:
            amax = A.abs().amax().clamp(min=eps).to(torch.float32)
            scale = amax / fp8_max
            iscale = 1.0 / scale
            A_q = (A.float() * iscale).clamp(fp8_min, fp8_max).to(fp8_dtype)
            return A_q, scale.view(1)


def _int8_quantize(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    per_act_token: bool,
    block_shape: Optional[list[int]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """INT8 quantization: per-tensor, per-token, or block-wise."""
    iinfo = torch.iinfo(torch.int8)
    int8_max = iinfo.max
    int8_min = iinfo.min
    eps = 1e-10

    if block_shape is not None:
        assert not per_act_token
        assert len(block_shape) == 2
        block_k = block_shape[1]
        assert A.size(-1) % block_k == 0
        orig_shape = A.shape
        A_flat = A.reshape(-1, A.size(-1))
        M, K = A_flat.shape
        A_groups = A_flat.reshape(M * (K // block_k), block_k)
        amax = (
            A_groups.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        )
        scale = amax / int8_max
        A_q = (
            (A_groups.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        )
        A_q = A_q.reshape(orig_shape)
        scale = scale.reshape(M, K // block_k)
        return A_q, scale

    elif per_act_token:
        A_flat = A.reshape(-1, A.size(-1))
        amax = A_flat.abs().amax(dim=-1, keepdim=True).clamp(min=eps).to(torch.float32)
        scale = amax / int8_max
        A_q = (A_flat.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        A_q = A_q.reshape(A.shape)
        scale = scale.reshape(A.shape[:-1] + (1,))
        return A_q, scale

    else:
        assert A_scale is not None, "int8 per-tensor requires A_scale"
        scale = A_scale.float().view(1, 1) if A_scale.numel() == 1 else A_scale.float()
        A_q = (A.float() / scale).round().clamp(int8_min, int8_max).to(torch.int8)
        return A_q, A_scale


def moe_kernel_quantize_input(
    A: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    quant_dtype: None | torch.dtype | str,
    per_act_token_quant: bool,
    block_shape: Optional[list[int]] = None,
    ocp_mx_scheme: str | None = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Quantize MoE input activations before GEMM."""
    if ocp_mx_scheme is not None:
        if ocp_mx_scheme in {"w_mxfp4", "w_mxfp4_a_mxfp4"}:
            pass
        elif ocp_mx_scheme.endswith("a_fp8"):
            qA, qA_scale = _fp8_quantize(A, A_scale, per_act_token=False)
            A = (qA.float() * qA_scale.float()).to(A.dtype)
            return A, None

    if quant_dtype is None:
        return A, A_scale
    elif quant_dtype == torch.float8_e4m3fn:
        return _fp8_quantize(A, A_scale, per_act_token_quant, block_shape)
    elif quant_dtype == torch.int8:
        return _int8_quantize(A, A_scale, per_act_token_quant, block_shape)
    else:
        return A, A_scale


def _ensure_block_size_k_divisible(
    size_k: int, block_size_k: int, group_size: int
) -> int:
    """Find largest block_size_k that divides size_k and is divisible by group_size."""
    if size_k % block_size_k == 0 and block_size_k % group_size == 0:
        return block_size_k

    max_search = min(block_size_k, size_k)
    start = (max_search // group_size) * group_size
    for candidate in range(start, group_size - 1, -group_size):
        if size_k % candidate == 0:
            return candidate

    if size_k % group_size == 0:
        return group_size

    return size_k


@pointwise_dynamic(promotion_methods=[(0, 1, "DEFAULT")])
@triton.jit
def _silu_and_mul_kernel(x, y):
    x_fp32 = x.to(tl.float32)
    x_silu = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    return x_silu * y


@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """Fused MoE kernel for GPTQ/AWQ (WNA16) quantized weights."""
    # Map pid to C block (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for first blocks of A and B
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # Cast to int64 to prevent overflow in stride*offset products
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] // 2) * stride_bk
            + offs_bn[None, :] * stride_bn
        )
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # Accumulate C block in fp32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
            other=0.0,
        )
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = (
            b_scale_ptr
            + off_experts * stride_bse
            + offs_bn[None, :] * stride_bsn
            + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
        )
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + (offs_bn[None, :] // 2) * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = (b_zp >> b_zp_shifter) & 0xF
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = (
                b_zp_ptr
                + off_experts * stride_bze
                + offs_bn[None, :] * stride_bzn
                + offs_k_true * stride_bzk
            )
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # Write back output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    naive_block_assignment: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SWAP_AB: tl.constexpr,
    K_DIVISIBLE_BY_BLOCK_K: tl.constexpr,
    FUSE_SILU: tl.constexpr,
):
    """Fused MoE kernel: token × expert GEMM with quantization support and optional SiLU fusion."""
    # Map pid to C block (grouped ordering for L2 reuse)
    pid = tl.program_id(axis=0)
    # Adjust N for FUSE_SILU. If fused, the actual output dimension is N // 2
    N_out = N // 2 if FUSE_SILU else N
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N_out, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for first blocks of A and B
    offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    if not naive_block_assignment:
        offs_token_id = pid_m * BLOCK_SIZE_M + offs
        offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    else:
        offs_token = tl.where(
            offs == 0,
            pid_m,  # first element = pid_m
            num_valid_tokens,  # remaining elements = constant
        )
    offs_token = offs_token.to(tl.int64)  # prevent int32 overflow

    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N_out
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_base = a_ptr + (offs_token[:, None] // top_k * stride_am)

    if FUSE_SILU:
        offs_bn_gate = offs_bn
        offs_bn_up = offs_bn + N_out

        b_expert_base = b_ptr + off_experts * stride_be
        b_ptrs_gate = b_expert_base + (
            offs_k[:, None] * stride_bk + offs_bn_gate[None, :] * stride_bn
        )
        b_ptrs_up = b_expert_base + (
            offs_k[:, None] * stride_bk + offs_bn_up[None, :] * stride_bn
        )

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:  # block-wise
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                # Use scalar scale load for hardware broadcast when block size fits within quantization group.
                if BLOCK_SIZE_N <= group_n:
                    offs_bsn_gate_idx = (pid_n * BLOCK_SIZE_N) % N_out // group_n
                    offs_bsn_up_idx = (
                        (pid_n * BLOCK_SIZE_N) % N_out + N_out
                    ) // group_n
                else:
                    offs_bsn_gate_idx = offs_bn_gate // group_n
                    offs_bsn_up_idx = offs_bn_up // group_n
                b_scale_gate_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bsn_gate_idx * stride_bsn
                )
                b_scale_up_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bsn_up_idx * stride_bsn
                )
            elif per_channel_quant:  # channel-wise
                b_scale_gate_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn_gate[None, :] * stride_bsn
                )
                b_scale_gate = tl.load(b_scale_gate_ptrs)
                b_scale_up_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn_up[None, :] * stride_bsn
                )
                b_scale_up = tl.load(b_scale_up_ptrs)
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
            else:  # tensor-wise
                a_scale = tl.load(a_scale_ptr)
                b_scale_gate = tl.load(b_scale_ptr + off_experts)
                b_scale_up = b_scale_gate

        # Pass 1: Sequential execution of gate projection to minimize peak register pressure.
        a_ptrs = a_base + offs_k[None, :] * stride_ak
        acc_gate = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Eliminate masking overhead when K is perfectly aligned with BLOCK_SIZE_K.
            if K_DIVISIBLE_BY_BLOCK_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b_gate = tl.load(b_ptrs_gate)
            else:
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b_gate = tl.load(
                    b_ptrs_gate, mask=offs_k[:, None] < k_remaining, other=0.0
                )

            if use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    b_scale_val = tl.load(b_scale_gate_ptrs + offs_ks * stride_bsk)

                    # Pre-compute combined scale to reduce arithmetic overhead via the associative property.
                    if BLOCK_SIZE_N <= group_n:
                        combined_scale = a_scale[:, None] * b_scale_val
                    else:
                        combined_scale = a_scale[:, None] * b_scale_val[None, :]
                    acc_gate += tl.dot(a, b_gate) * combined_scale
                else:
                    if use_fp8_w8a8:
                        acc_gate = tl.dot(a, b_gate, acc=acc_gate)
                    else:
                        acc_gate += tl.dot(a, b_gate)
            else:
                acc_gate += tl.dot(a, b_gate)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs_gate += BLOCK_SIZE_K * stride_bk

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                pass
            elif per_channel_quant:
                acc_gate = acc_gate * a_scale * b_scale_gate
            else:
                acc_gate = acc_gate * a_scale * b_scale_gate

        # Pass 2: Sequential up projection; operand A is reloaded with high L1 hit rate.
        a_ptrs = a_base + offs_k[None, :] * stride_ak
        acc_up = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Apply mask elimination during the up projection stage.
            if K_DIVISIBLE_BY_BLOCK_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b_up = tl.load(b_ptrs_up)
            else:
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b_up = tl.load(b_ptrs_up, mask=offs_k[:, None] < k_remaining, other=0.0)

            if use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    b_scale_val = tl.load(b_scale_up_ptrs + offs_ks * stride_bsk)

                    # Apply pre-computed scale merging to reduce multiplication overhead.
                    if BLOCK_SIZE_N <= group_n:
                        combined_scale = a_scale[:, None] * b_scale_val
                    else:
                        combined_scale = a_scale[:, None] * b_scale_val[None, :]
                    acc_up += tl.dot(a, b_up) * combined_scale
                else:
                    if use_fp8_w8a8:
                        acc_up = tl.dot(a, b_up, acc=acc_up)
                    else:
                        acc_up += tl.dot(a, b_up)
            else:
                acc_up += tl.dot(a, b_up)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs_up += BLOCK_SIZE_K * stride_bk

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:
                pass
            elif per_channel_quant:
                acc_up = acc_up * a_scale * b_scale_up
            else:
                acc_up = acc_up * a_scale * b_scale_up

        # SiLU activation fusion
        accumulator = tl.fdiv(acc_gate, (1.0 + tl.exp(-acc_gate))) * acc_up

    else:
        if off_experts == -1:
            # Expert not in current EP rank, write zeros
            write_zeros_to_output(
                c_ptr,
                stride_cm,
                stride_cn,
                pid_n,
                N_out,
                offs_token,
                token_mask,
                BLOCK_SIZE_M,
                BLOCK_SIZE_N,
                compute_type,
            )
            return
        a_ptrs = a_base + offs_k[None, :] * stride_ak
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        )

        if use_int8_w8a16:
            b_scale_ptrs = (
                b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn
            )
            b_scale = tl.load(b_scale_ptrs)

        if use_fp8_w8a8 or use_int8_w8a8:
            if group_k > 0 and group_n > 0:  # block-wise
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                # Use scalar scale load for hardware broadcast when block size fits within quantization group.
                if BLOCK_SIZE_N <= group_n:
                    offs_bsn = (pid_n * BLOCK_SIZE_N) % N_out // group_n
                else:
                    offs_bsn = offs_bn // group_n
                b_scale_ptrs = (
                    b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn
                )
            elif per_channel_quant:  # channel-wise
                b_scale_ptrs = (
                    b_scale_ptr
                    + off_experts * stride_bse
                    + offs_bn[None, :] * stride_bsn
                )
                b_scale = tl.load(b_scale_ptrs)
                a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
                a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:, None]
            else:  # tensor-wise
                a_scale = tl.load(a_scale_ptr)
                b_scale = tl.load(b_scale_ptr + off_experts)

        if HAS_BIAS:
            bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
            bias = tl.load(bias_ptrs, mask=(offs_bn < N_out), other=0.0)

        # Accumulate C block in fp32
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        if SWAP_AB:
            accumulator_nm = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)

        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Eliminate masking overhead when K is perfectly aligned with BLOCK_SIZE_K.
            if K_DIVISIBLE_BY_BLOCK_K:
                a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
                b = tl.load(b_ptrs)
            else:
                k_remaining = K - k * BLOCK_SIZE_K
                a = tl.load(
                    a_ptrs,
                    mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
                    other=0.0,
                )
                b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)

            if use_int8_w8a16:
                accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
            elif use_fp8_w8a8 or use_int8_w8a8:
                if group_k > 0 and group_n > 0:
                    k_start = k * BLOCK_SIZE_K
                    offs_ks = k_start // group_k
                    a_scale = tl.load(
                        a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0
                    )
                    if SWAP_AB:
                        b_scale_val = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                        if BLOCK_SIZE_N <= group_n:
                            combined_scale_nm = b_scale_val * a_scale[None, :]
                        else:
                            combined_scale_nm = b_scale_val[:, None] * a_scale[None, :]
                        accumulator_nm += (
                            tl.dot(tl.trans(b), tl.trans(a)) * combined_scale_nm
                        )
                    else:
                        b_scale_val = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
                        # Pre-compute combined scale to reduce arithmetic overhead via the associative property.
                        if BLOCK_SIZE_N <= group_n:
                            combined_scale = a_scale[:, None] * b_scale_val
                        else:
                            combined_scale = a_scale[:, None] * b_scale_val[None, :]
                        accumulator += tl.dot(a, b) * combined_scale
                else:
                    if use_fp8_w8a8:
                        if SWAP_AB:
                            accumulator_nm = tl.dot(
                                tl.trans(b), tl.trans(a), acc=accumulator_nm
                            )
                        else:
                            accumulator = tl.dot(a, b, acc=accumulator)
                    else:
                        if SWAP_AB:
                            accumulator_nm += tl.dot(tl.trans(b), tl.trans(a))
                        else:
                            accumulator += tl.dot(a, b)
            else:
                if SWAP_AB:
                    accumulator_nm += tl.dot(tl.trans(b), tl.trans(a))
                else:
                    accumulator += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        if SWAP_AB:
            accumulator = tl.trans(accumulator_nm)

        # Dequantization
        if use_int8_w8a16:
            accumulator = accumulator * b_scale
        elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
            accumulator = accumulator * a_scale * b_scale

        if HAS_BIAS:
            accumulator += bias[None, :]

    # Router weight multiplication (must be in fp32)
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token,
            mask=token_mask,
            other=0,
        )
        accumulator *= moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    # Write back output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N_out)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def invoke_fused_moe_wna16_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor | None,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    block_shape: list[int] | None,
):
    assert B_scale is not None and B_scale.ndim == 3
    assert B_zp is None or B_zp.ndim == 3
    assert block_shape is not None and block_shape[0] == 0

    M = A.size(0)
    num_tokens = M * top_k

    EM = sorted_token_ids.size(0)
    if A.size(0) < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique,
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"])
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),
    )
    config = config.copy()
    config.update(
        get_moe_wna16_block_config(
            config=config,
            use_moe_wna16_cuda=False,
            num_valid_tokens=num_tokens,
            size_k=A.size(1),
            size_n=B.size(1),
            num_experts=B.size(1),
            group_size=block_shape[1],
            real_top_k=top_k,
            block_size_m=config["BLOCK_SIZE_M"],
        )
    )

    fused_moe_kernel_gptq_awq[grid](
        A,
        B,
        C,
        B_scale,
        B_zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),
        A.size(1),
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        B_zp.stride(0) if B_zp is not None else 0,
        B_zp.stride(2) if B_zp is not None else 0,
        B_zp.stride(1) if B_zp is not None else 0,
        block_k_diviable=A.size(1) % config["BLOCK_SIZE_K"] == 0,
        group_size=block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        has_zp=B_zp is not None,
        use_int4_w4a16=use_int4_w4a16,
        use_int8_w8a16=use_int8_w8a16,
        **config,
    )


def invoke_fused_moe_triton_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    block_shape: Optional[list[int]] = None,
    B_bias: torch.Tensor | None = None,
    FUSE_SILU: bool = False,
) -> None:
    """Launch the fused_moe_kernel Triton kernel."""
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert block_shape is None or triton.cdiv(
            B.size(-2), block_shape[0]
        ) == B_scale.size(-2)
        assert block_shape is None or triton.cdiv(
            B.size(-1), block_shape[1]
        ) == B_scale.size(-1)
    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.size(0)
    num_tokens = M * top_k
    if sorted_token_ids is not None:
        EM = sorted_token_ids.size(0)
        if A.size(0) < config["BLOCK_SIZE_M"]:
            EM = min(
                sorted_token_ids.size(0), A.size(0) * top_k * config["BLOCK_SIZE_M"]
            )
    else:
        EM = num_tokens * config["BLOCK_SIZE_M"]

    #  FUSE_SILU means B.size(1) contains both Gate and Up. N is halved.
    actual_N = B.size(1) // 2 if FUSE_SILU else B.size(1)
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"])
        * triton.cdiv(actual_N, META["BLOCK_SIZE_N"]),
    )
    HAS_BIAS = B_bias is not None

    config = config.copy()
    config["SPLIT_K"] = 1
    BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
    if block_shape is not None:
        BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0], block_shape[1]))

    swap_AB = config.pop("SWAP_AB", False)
    # Force disable SWAP_AB in fusion mode
    if FUSE_SILU:
        swap_AB = False

    fused_moe_kernel[grid](
        A,
        B,
        C,
        B_bias,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.size(1),  # N
        B.size(2),  # K
        EM,
        num_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        A_scale.stride(0) if A_scale is not None and A_scale.ndim == 2 else 0,
        A_scale.stride(1) if A_scale is not None and A_scale.ndim == 2 else 0,
        B_scale.stride(0) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_scale.stride(2) if B_scale is not None and B_scale.ndim == 3 else 0,
        B_scale.stride(1) if B_scale is not None and B_scale.ndim >= 2 else 0,
        B_bias.stride(0) if B_bias is not None else 0,
        B_bias.stride(1) if B_bias is not None else 0,
        0 if block_shape is None else block_shape[0],
        0 if block_shape is None else block_shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        per_channel_quant=per_channel_quant,
        naive_block_assignment=(sorted_token_ids is None),
        HAS_BIAS=HAS_BIAS,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SWAP_AB=swap_AB,
        K_DIVISIBLE_BY_BLOCK_K=(B.size(2) % BLOCK_SIZE_K == 0),
        FUSE_SILU=FUSE_SILU,
        **config,
    )


def dispatch_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: Optional[torch.Tensor],
    B_scale: Optional[torch.Tensor],
    B_zp: Optional[torch.Tensor],
    topk_weights: Optional[torch.Tensor],
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: dict[str, Any],
    compute_type: tl.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    use_int4_w4a16: bool,
    per_channel_quant: bool,
    block_shape: Optional[list[int]] = None,
    B_bias: Optional[torch.Tensor] = None,
    FUSE_SILU: bool = False,
) -> None:
    """Dispatch to the appropriate fused MoE kernel based on quantization flags."""
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids is None or sorted_token_ids.stride(0) == 1

    # M = A.size(0)
    # num_tokens = M * top_k

    if False:
        # TODO: Other precision-specific implementations
        # use_fp8_w8a8,
        # use_int8_w8a8,
        # use_int8_w8a16,
        # use_int4_w4a16,
        pass
    if (use_int8_w8a16 or use_int4_w4a16) and (
        block_shape is not None and block_shape[1] > 0
    ):
        assert B_bias is None
        invoke_fused_moe_wna16_triton_kernel(
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_int8_w8a16,
            use_int4_w4a16,
            block_shape,
        )
    else:
        invoke_fused_moe_triton_kernel(
            A,
            B,
            C,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            mul_routed_weight,
            top_k,
            config,
            compute_type,
            use_fp8_w8a8,
            use_int8_w8a8,
            use_int8_w8a16,
            use_int4_w4a16,
            per_channel_quant,
            block_shape,
            B_bias,
            FUSE_SILU=FUSE_SILU,
        )


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: torch.Tensor | None = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logger.debug("GEMS FUSED MOE")
    assert (
        activation == "silu"
    ), f"Only 'silu' activation is supported, got {activation}"

    activation_enum = MoEActivation.from_str(activation)

    # Check constraints
    if use_int4_w4a16:
        # INT4 stored unpacked in INT8 containers (full K dim)
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"
    elif ocp_mx_scheme is not None:
        if ocp_mx_scheme.startswith("w_mxfp4"):
            assert hidden_states.size(1) == w1.size(2) * 2, "hidden size mismatch"
        elif ocp_mx_scheme.startswith("w_mxfp6"):
            assert (
                hidden_states.size(1) == (w1.size(2) * 4) // 3
            ), "hidden size mismatch"
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")
    else:
        assert hidden_states.size(1) == w1.size(
            2
        ), f"Hidden size mismatch {hidden_states.size(1)} != {w1.size(2)}"

    assert topk_weights.size() == topk_ids.size(), "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]

    num_tokens = hidden_states.size(0)
    E, N, _ = w1.size()
    K = w2.size(1)
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.size(1)

    CHUNK_SIZE: int = 32 * 1024
    M = min(num_tokens, CHUNK_SIZE)

    config_dtype = _get_config_dtype_str(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        ocp_mx_scheme=ocp_mx_scheme,
        dtype=hidden_states.dtype,
    )

    quant_dtype = _get_config_quant_dtype(
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        ocp_mx_scheme=ocp_mx_scheme,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.size(),
        w2.size(),
        top_k_num,
        config_dtype,
        block_shape=block_shape,
        E=E,
    )

    config = get_config_func(M)

    # cache1 and cache3 share memory (non-overlapping lifetime)
    cache13 = torch.empty(
        M * top_k_num * max(N, K),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache13[: M * top_k_num * N].view(M, top_k_num, N)
    intermediate_cache3 = cache13[: M * top_k_num * K].view(M, top_k_num, K)

    # cache2 needs separate memory (concurrent with cache1)
    activation_out_dim = MoEActivation.adjust_N_for_activation(N, activation_enum)
    intermediate_cache2 = torch.empty(
        (M * top_k_num, activation_out_dim),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    out_hidden_states = hidden_states if inplace else torch.empty_like(hidden_states)

    if ocp_mx_scheme is not None:
        # Dequantize OCP MX weights (TODO: skip on platforms with native MX)
        if ocp_mx_scheme.startswith("w_mxfp4"):
            w1 = dequant_mxfp4(w1, w1_scale, hidden_states.dtype)
            w1_scale = None
            w2 = dequant_mxfp4(w2, w2_scale, hidden_states.dtype)
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e3m2"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e3m2", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        elif ocp_mx_scheme.startswith("w_mxfp6_e2m3"):
            w1 = dequant_mxfp6(
                w1, w1_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w1_scale = None
            w2 = dequant_mxfp6(
                w2, w2_scale, quant_dtype="fp6_e2m3", float_dtype=hidden_states.dtype
            )
            w2_scale = None
        else:
            raise NotImplementedError(f"Unsupported ocp_mx_scheme={ocp_mx_scheme}")

    # Dequant INT8/INT4 weights (Triton can't do mixed-dtype dot)
    if use_int8_w8a16 or use_int4_w4a16:
        w1 = w1.to(hidden_states.dtype) * w1_scale.unsqueeze(-1).to(hidden_states.dtype)
        w1_scale = None
        w2 = w2.to(hidden_states.dtype) * w2_scale.unsqueeze(-1).to(hidden_states.dtype)
        w2_scale = None
        use_int8_w8a16 = False
        use_int4_w4a16 = False

    # Check if we can safely fuse the activation with the first GEMM pass
    can_use_fused_silu = (
        activation_enum in (MoEActivation.SILU, MoEActivation.SWIGLUOAI)
        and w1_bias is None
        and expert_map is None  # Fused kernel doesn't handle EP -1 experts
    )

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (
            chunk * CHUNK_SIZE,
            min((chunk + 1) * CHUNK_SIZE, num_tokens),
        )
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.size()

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust cache size for last chunk
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[
                : tokens_in_chunk * topk_ids.size(1)
            ]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        SPARSITY_FACTOR = 4
        naive_block_assignment = (
            expert_map is None
            and tokens_in_chunk * top_k_num * SPARSITY_FACTOR <= global_num_experts
            and not (
                (use_int8_w8a16 or use_int4_w4a16)
                and block_shape is not None
                and block_shape[1] > 0
            )
        )

        if not naive_block_assignment:
            sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
                curr_topk_ids,
                config["BLOCK_SIZE_M"],
                global_num_experts,
                expert_map,
                # ignore_invalid_experts=True,
            )
        else:
            max_num_tokens_padded = topk_ids.numel() * config["BLOCK_SIZE_M"]
            expert_ids = curr_topk_ids.view(-1)
            num_tokens_post_padded = torch.empty(
                (1), dtype=torch.int32, device=topk_ids.device
            )
            num_tokens_post_padded.fill_(max_num_tokens_padded)
            sorted_token_ids = None

        # 1. Extract a unified boolean flag for fusion
        do_fuse_silu = can_use_fused_silu and not naive_block_assignment

        # 2. Dynamically determine the differing parameters based on the fusion flag
        if do_fuse_silu:
            # Output goes directly to cache 2 with adjusted dimensions
            out_cache = intermediate_cache2.view(
                tokens_in_chunk, top_k_num, activation_out_dim
            )

            # Fused kernel weight handling depends on apply_router_weight_on_input
            if apply_router_weight_on_input:
                weights_arg = curr_topk_weights
            else:
                weights_arg = None
        else:
            # Standard path outputs to cache 1
            out_cache = intermediate_cache1
            # Standard path always passes the weights
            weights_arg = curr_topk_weights

        # 3. Unified dispatch call to eliminate redundant code blocks
        dispatch_fused_moe_kernel(
            qcurr_hidden_states,
            w1,
            out_cache,  # Dynamically assigned output buffer
            a1q_scale,
            w1_scale,
            w1_zp,
            weights_arg,  # Dynamically assigned weights argument
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            top_k_num,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w1_bias,
            FUSE_SILU=do_fuse_silu,  # Master switch for the kernel
        )

        # 4. Apply activation separately if the fused path was not taken
        if not do_fuse_silu:
            apply_moe_activation(
                activation_enum, intermediate_cache2, intermediate_cache1.view(-1, N)
            )

        qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape,
            ocp_mx_scheme=ocp_mx_scheme,
        )

        if expert_map is not None:
            intermediate_cache3.zero_()

        dispatch_fused_moe_kernel(
            qintermediate_cache2,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            w2_zp,
            curr_topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
            B_bias=w2_bias,
            FUSE_SILU=False,
        )

        moe_sum(
            intermediate_cache3.view(*intermediate_cache3.size()),
            out_hidden_states[begin_chunk_idx:end_chunk_idx],
        )

    return out_hidden_states


def inplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> None:
    """
    In-place fused MoE: writes output directly into ``hidden_states``.

    Same semantics as ``fused_experts_impl(..., inplace=True)``.
    Returns None (the result is stored in ``hidden_states``).
    """
    fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=True,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )


def outplace_fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Out-of-place fused MoE: allocates and returns a new output tensor.

    Same semantics as ``fused_experts_impl(..., inplace=False)``.
    """
    return fused_experts_impl(
        hidden_states,
        w1,
        w2,
        topk_weights,
        topk_ids,
        inplace=False,
        activation=activation,
        apply_router_weight_on_input=apply_router_weight_on_input,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_int4_w4a16=use_int4_w4a16,
        per_channel_quant=per_channel_quant,
        global_num_experts=global_num_experts,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
    )
