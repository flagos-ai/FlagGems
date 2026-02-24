import random
from itertools import product
from math import ceil
from typing import Optional

import pytest
import torch

import flag_gems

from .conftest import QUICK_MODE

random.seed(42)


def is_vllm_available():
    try:
        import vllm  # noqa: 401

        return True
    except ImportError:
        return False


VLLM_AVAILABLE = is_vllm_available()


def is_hopper_available():
    if flag_gems.device != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability()
    sm_version_num = major * 10 + minor
    return sm_version_num >= 90 and sm_version_num < 100


HOPPER_AVAILABLE = is_hopper_available()


def to_int8(tensor: torch.Tensor):
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)


def to_fp8(tensor: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    return torch.round(tensor.clamp(min=finfo.min, max=finfo.max)).to(
        dtype=torch.float8_e4m3fn
    )


class CutlassScaledMMTestKit:
    num_test_cases = 16 if QUICK_MODE else 32

    @staticmethod
    def _get_all_combinations():
        # these shapes come from the test file of op `cutlass_scaled_mm` of vLLM
        mnk = [
            (1, 256, 128),
            (1, 16384, 1024),
            (1, 24576, 496),
            (16, 256, 496),
            (16, 16384, 128),
            (16, 24576, 4096),
            (32, 8192, 4096),
            (32, 16384, 4096),
            (33, 1024, 1024),
            (33, 8192, 128),
            (64, 2048, 496),
            (64, 16384, 1024),
            (100, 8192, 496),
            (128, 32768, 4096),
            (256, 4096, 4096),
            (512, 256, 1024),
            (512, 8192, 4096),
            (512, 16384, 128),
            (512, 24576, 128),
        ]
        scale_shape_types = ["scalar", "vector", "matrix"]
        if_use_bias = [True, False]
        dtypes = [(torch.int8, torch.float16), (torch.float8_e4m3fn, torch.bfloat16)]

        combinations = product(
            mnk, scale_shape_types, scale_shape_types, if_use_bias, dtypes
        )
        return combinations

    @classmethod
    def _rand_sample(cls, all_params):
        random.shuffle(all_params)
        return all_params[: cls.num_test_cases]

    @classmethod
    def get_test_params(cls):
        combinations = cls._get_all_combinations()

        all_params = []
        for (
            (M, N, K),
            a_scale_category,
            b_scale_category,
            bias,
            (in_dtype, out_dtype),
        ) in combinations:
            is_scalar_or_vector_dequant = a_scale_category in [
                "scalar",
                "vector",
            ] and b_scale_category in ["scalar", "vector"]
            is_block_dequant = (
                a_scale_category == "matrix" and b_scale_category == "matrix"
            )

            if not (is_scalar_or_vector_dequant or is_block_dequant):
                continue

            if is_block_dequant and (bias is not None or M % 4 != 0):
                continue

            param = {
                "M": M,
                "N": N,
                "K": K,
                "a_scale_category": a_scale_category,
                "b_scale_category": b_scale_category,
                "use_bias": bias,
                "in_dtype": in_dtype,
                "out_dtype": out_dtype,
            }
            all_params.append(param)

        return cls._rand_sample(all_params)

    @staticmethod
    def get_scale_shape(M, N, K, category, is_a_scale=True):
        if category == "scalar":
            return (1,)
        elif category == "vector":
            if is_a_scale:
                return (M,)
            else:
                return (N,)
        else:
            if is_a_scale:
                return (M, ceil(K / 128))
            else:
                return (ceil(K / 128), ceil(N / 128))

    @staticmethod
    def baseline_scaled_mm(
        a: torch.Tensor,
        b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        out_dtype: torch.dtype,
        bias: Optional[torch.Tensor] = None,
    ):
        def group_broadcast(t: torch.Tensor, shape):
            for i, s in enumerate(shape):
                if t.shape[i] != s and t.shape[i] != 1:
                    assert s % t.shape[i] == 0
                    t = (
                        t.unsqueeze(i + 1)
                        .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                        .flatten(i, i + 1)
                    )
            return t

        scale_a_full = group_broadcast(scale_a, a.shape)
        scale_b_full = group_broadcast(scale_b, b.shape)

        a_f32 = a.to(torch.float32)
        b_f32 = b.to(torch.float32)

        lhs = scale_a_full * a_f32
        rhs = scale_b_full * b_f32

        output = torch.mm(lhs, rhs).to(out_dtype)

        if bias is not None:
            output = output + bias

        return output


@pytest.mark.skipif(
    not (VLLM_AVAILABLE and HOPPER_AVAILABLE),
    reason="requires vLLM and NVIDIA Hopper architecture",
)
@pytest.mark.cutlass_scaled_mm
@pytest.mark.parametrize("p", CutlassScaledMMTestKit.get_test_params())
def test_cutlass_scaled_mm(p):
    kit = CutlassScaledMMTestKit

    M, N, K = p["M"], p["N"], p["K"]
    in_dtype = p["in_dtype"]
    out_dtype = p["out_dtype"]
    a_scale_category = p["a_scale_category"]
    b_scale_category = p["b_scale_category"]

    if in_dtype == torch.int8:
        a = to_int8(torch.randn((M, K), device=flag_gems.device))
        b = to_int8(
            torch.randn((K, N), device=flag_gems.device).t().contiguous().t() * 5
        )
    else:
        a = to_fp8(torch.randn((M, K), device=flag_gems.device))
        b = to_fp8(torch.randn((K, N), device=flag_gems.device).t().contiguous().t())

    a_scale_shape = kit.get_scale_shape(M, N, K, a_scale_category)
    b_scale_shape = kit.get_scale_shape(M, N, K, b_scale_category, False)

    scale_a = torch.randn(a_scale_shape, device=flag_gems.device, dtype=torch.float32)
    scale_b = torch.randn(b_scale_shape, device=flag_gems.device, dtype=torch.float32)

    scale_a = scale_a.contiguous()
    # convert scale_b to col-major
    # (for scalar/vector scale_b, this's a identical transformation)
    scale_b = scale_b.t().contiguous().t()

    bias = None
    if p["use_bias"]:
        bias = torch.randn((N,), device=flag_gems.device, dtype=out_dtype)

    c = torch.empty((M, N), device=flag_gems.device, dtype=out_dtype)

    flag_gems.cutlass_scaled_mm(c, a, b, scale_a, scale_b, bias)

    output_ref = kit.baseline_scaled_mm(
        a, b, scale_a.view(-1, 1), scale_b.view(1, -1), out_dtype, bias
    )

    if in_dtype == torch.int8:
        rtol, atol = 1e-1, 1.0
    else:
        rtol, atol = 5e-1, 1.5e-1

    torch.testing.assert_close(c, output_ref, rtol=rtol, atol=atol)


# ============ cp_gather_indexer_k_quant_cache tests ============
class CpGatherIndexerKQuantCacheTestKit:
    num_test_cases = 8 if QUICK_MODE else 16

    @staticmethod
    def _get_all_combinations():
        # Test configurations: (batch_size, max_seq_len, head_dim, block_size)
        configs = [
            (1, 128, 64, 16),
            (2, 256, 64, 16),
            (4, 512, 128, 32),
            (8, 256, 64, 16),
            (16, 128, 64, 32),
            (32, 64, 128, 16),
            (4, 1024, 64, 64),
            (8, 512, 128, 32),
        ]
        return configs

    @classmethod
    def get_test_params(cls):
        all_configs = cls._get_all_combinations()
        random.shuffle(all_configs)
        return all_configs[: cls.num_test_cases]


@pytest.mark.cp_gather_indexer_k_quant_cache
@pytest.mark.parametrize(
    "batch_size,max_seq_len,head_dim,block_size",
    CpGatherIndexerKQuantCacheTestKit.get_test_params(),
)
def test_cp_gather_indexer_k_quant_cache(batch_size, max_seq_len, head_dim, block_size):
    from flag_gems.fused.cp_gather_indexer_k_quant_cache import (
        cp_gather_indexer_k_quant_cache,
    )

    device = flag_gems.device

    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(
        block_size, max_seq_len + 1, (batch_size,), device=device, dtype=torch.int32
    )

    # Create cumulative sequence lengths
    cu_seq_lens = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seq_lens[1:] = torch.cumsum(seq_lens, dim=0)
    num_tokens = int(cu_seq_lens[-1].item())

    # Calculate number of blocks needed
    num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq

    # cache_stride = head_dim + 4 (4 bytes for float32 scale per token)
    cache_stride = head_dim + 4

    # Create kv_cache: [num_blocks, block_size, cache_stride]
    # FP8 data for K values, float32 scale packed at the end
    kv_cache = torch.zeros(
        total_blocks, block_size, cache_stride, dtype=torch.float8_e4m3fn, device=device
    )

    # Fill kv_cache with test data
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        for pos in range(seq_len):
            block_idx = b * num_blocks_per_seq + pos // block_size
            block_offset = pos % block_size
            # Fill K values with position-dependent data
            kv_cache[block_idx, block_offset, :head_dim] = to_fp8(
                torch.randn(head_dim, device=device) * 0.1
            )
            # Fill scale (last 4 bytes as float32)
            scale_val = torch.tensor(
                [0.5 + pos * 0.01], dtype=torch.float32, device=device
            )
            kv_cache[block_idx, block_offset, head_dim:] = scale_val.view(
                torch.float8_e4m3fn
            )

    # Create block_table: [batch_size, num_blocks_per_seq]
    block_table = torch.arange(total_blocks, device=device, dtype=torch.int32).view(
        batch_size, num_blocks_per_seq
    )

    # Create output tensors
    dst_k = torch.empty(num_tokens, head_dim, dtype=torch.float8_e4m3fn, device=device)
    dst_scale = torch.empty(num_tokens, 1, dtype=torch.float8_e4m3fn, device=device)

    # Call the function
    cp_gather_indexer_k_quant_cache(
        kv_cache, dst_k, dst_scale, block_table, cu_seq_lens
    )

    # Verify results by checking that data was gathered correctly
    dst_scale_f32 = dst_scale.view(-1).view(torch.float32)

    token_idx = 0
    for b in range(batch_size):
        seq_len = int(seq_lens[b].item())
        for pos in range(seq_len):
            block_idx = b * num_blocks_per_seq + pos // block_size
            block_offset = pos % block_size

            # Check K values match
            expected_k = kv_cache[block_idx, block_offset, :head_dim]
            actual_k = dst_k[token_idx]
            assert torch.equal(
                actual_k, expected_k
            ), f"K mismatch at batch {b}, pos {pos}"

            # Check scale matches
            expected_scale = kv_cache[block_idx, block_offset, head_dim:].view(
                torch.float32
            )
            actual_scale = dst_scale_f32[token_idx : token_idx + 1]
            torch.testing.assert_close(
                actual_scale, expected_scale, rtol=1e-5, atol=1e-5
            )

            token_idx += 1
