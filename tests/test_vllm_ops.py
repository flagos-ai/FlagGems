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


# ============================================================================
# top_k_per_row_decode tests (converted from vLLM)
# ============================================================================

# Test parameters for top_k_per_row_decode
TOP_K_VALUES = [2048, 3000]
TOP_K_BATCH_SIZE = [1, 2, 2048]
TOP_K_NEXT_N = [1, 8]
TOP_K_DATA_GENERATION = ["random", "10LSBits"]


def create_random_logits(
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    dtype: torch.dtype,
    seed: int,
    clean_logits: bool,
    data_generation: str,
) -> torch.Tensor:
    """Create random logits tensor for testing."""
    torch.manual_seed(seed)
    device = flag_gems.device
    # Generate logits with some structure to make testing more meaningful
    if data_generation == "random":
        logits = torch.randn(
            row_starts.shape[0], max(row_ends), dtype=dtype, device=device
        )
    elif data_generation == "10LSBits":
        top_22_bits_mask = 0xFFFFFC00
        last_10_bits_mask = 0x000003FF
        fixed_top_22_bits = 0x3F900000
        # Generate random bits for the last 10 bits
        random_bottom_bits = torch.randint(
            0,
            2**10,
            (row_starts.shape[0], max(row_ends)),
            dtype=torch.int32,
            device=device,
        )
        # Combine: fixed top 22 bits with random last 10 bits
        logits_bits = (fixed_top_22_bits & top_22_bits_mask) | (
            random_bottom_bits & last_10_bits_mask
        )
        logits = logits_bits.view(dtype)
    else:
        raise ValueError(f"Unknown data_generation: {data_generation}")

    if clean_logits:
        for i, end in enumerate(row_ends):
            logits[i, int(end) :] = float("-inf")
    return logits


def compare_top_k_results(
    logits: torch.Tensor,
    gems_indices: torch.Tensor,
    torch_indices: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    top_k: int,
    tolerance: float = 1e-5,
) -> bool:
    """
    Compare results from FlagGems top_k_per_row with torch.topk.
    Both results should contain the same top-k elements (values, not necessarily indices).
    """
    num_rows = gems_indices.shape[0]

    for row_idx in range(num_rows):
        # Get valid elements using row boundaries
        row_start = row_starts[row_idx].item()
        row_end = row_ends[row_idx].item()
        row_length = row_end - row_start
        num_valid = min(top_k, row_length)
        gems_row_indices = gems_indices[row_idx][:num_valid].cpu()
        torch_row_indices = torch_indices[row_idx][:num_valid].cpu()

        # Compare the sets of indices first
        gems_set = set(gems_row_indices.tolist())
        torch_set = set(torch_row_indices.tolist())
        if gems_set == torch_set:
            continue

        # Any difference in elements, compare the values
        logits_row = logits[row_idx]
        gems_row_values = [logits_row[i] for i in gems_row_indices]
        torch_row_values = [logits_row[i] for i in torch_row_indices]

        gems_only_values, torch_only_values = [], []
        for idx in gems_set - torch_set:
            gems_pos = (gems_row_indices == idx).nonzero(as_tuple=True)[0]
            gems_only_values.append(gems_row_values[gems_pos[0]])

        for idx in torch_set - gems_set:
            torch_pos = (torch_row_indices == idx).nonzero(as_tuple=True)[0]
            torch_only_values.append(torch_row_values[torch_pos[0]])

        if len(gems_only_values) != len(torch_only_values):
            return False
        if not torch.allclose(
            torch.tensor(gems_only_values),
            torch.tensor(torch_only_values),
            rtol=tolerance,
            atol=tolerance,
        ):
            return False

    return True


def _run_top_k_per_row_decode_test(
    top_k: int,
    batch_size: int,
    next_n: int,
    vocab_size: int,
    clean_logits: bool,
    data_generation: str,
) -> None:
    """
    Helper function to run top_k_per_row_decode test with given parameters.
    """
    from flag_gems.fused.top_k_per_row_decode import top_k_per_row_decode

    device = flag_gems.device

    # Create test data
    num_rows = batch_size * next_n
    seq_lens = torch.randint(
        low=next_n,
        high=vocab_size,
        size=(batch_size,),
        dtype=torch.int32,
        device=device,
    )
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device=device)
    row_indices = torch.arange(num_rows, device=device) // next_n
    next_n_offset = torch.arange(num_rows, device=device) % next_n
    row_ends = seq_lens[row_indices] - next_n + next_n_offset + 1
    logits = create_random_logits(
        row_starts, row_ends, torch.float32, 42, clean_logits, data_generation
    )

    # Create output tensors
    indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)

    # Run FlagGems implementation
    top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        num_rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
    )

    torch.cuda.synchronize() if device == "cuda" else None

    # Run reference implementation
    torch_indices = torch.empty((num_rows, top_k), dtype=torch.int32, device=device)
    for i in range(num_rows):
        row_end = int(row_ends[i])
        k_i = min(top_k, row_end)
        idx = logits[i, :row_end].topk(k_i, dim=-1)[1]
        torch_indices[i, :k_i] = idx

    # Compare results
    assert compare_top_k_results(
        logits, indices, torch_indices, row_starts, row_ends, top_k
    ), "FlagGems top_k_per_row_decode results don't match torch.topk"


@pytest.mark.skipif(flag_gems.device != "cuda", reason="This test requires CUDA")
@pytest.mark.top_k_per_row_decode
@pytest.mark.parametrize("top_k", TOP_K_VALUES if not QUICK_MODE else [2048])
@pytest.mark.parametrize("batch_size", TOP_K_BATCH_SIZE if not QUICK_MODE else [1, 32])
@pytest.mark.parametrize("next_n", TOP_K_NEXT_N if not QUICK_MODE else [1])
@pytest.mark.parametrize("clean_logits", [True, False] if not QUICK_MODE else [True])
@pytest.mark.parametrize(
    "data_generation", TOP_K_DATA_GENERATION if not QUICK_MODE else ["random"]
)
@torch.inference_mode()
def test_top_k_per_row_decode(
    top_k: int,
    batch_size: int,
    next_n: int,
    clean_logits: bool,
    data_generation: str,
) -> None:
    """
    Test top_k_per_row_decode with various parameter combinations.
    Converted from vLLM's test_top_k_per_row.py.
    """
    torch.manual_seed(0)
    vocab_size = 20000
    _run_top_k_per_row_decode_test(
        top_k, batch_size, next_n, vocab_size, clean_logits, data_generation
    )


@pytest.mark.skipif(flag_gems.device != "cuda", reason="This test requires CUDA")
@pytest.mark.top_k_per_row_decode
@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_top_k_per_row_decode_large_vocab_size(clean_logits: bool) -> None:
    """
    Test top_k_per_row_decode with large vocabulary size (300K).
    """
    torch.manual_seed(0)
    top_k = 2048
    batch_size = 2
    next_n = 2
    vocab_size = 300000
    data_generation = "random"
    _run_top_k_per_row_decode_test(
        top_k, batch_size, next_n, vocab_size, clean_logits, data_generation
    )


@pytest.mark.skipif(flag_gems.device != "cuda", reason="This test requires CUDA")
@pytest.mark.top_k_per_row_decode
@pytest.mark.parametrize("clean_logits", [True, False])
@torch.inference_mode()
def test_deepseek_hybrid_topk_short_sequences(clean_logits: bool) -> None:
    """
    Test top_k_per_row_decode for short sequences (< 8192).
    This is the portion of test_deepseek_hybrid_topk that uses top_k_per_row_decode.
    (Long sequences use large_context_topk which is not in scope.)
    """
    from flag_gems.fused.top_k_per_row_decode import top_k_per_row_decode

    device = flag_gems.device
    top_k = 2048

    # Test case: Short sequences (< 8192)
    batch_size_short = 4
    next_n = 1
    num_rows_short = batch_size_short * next_n

    # Create sequences with max length < 8192
    torch.manual_seed(42)
    seq_lens_short = torch.randint(
        4000, 8000, (batch_size_short,), dtype=torch.int32, device=device
    )

    row_starts_short = torch.zeros(num_rows_short, dtype=torch.int32, device=device)
    row_indices_short = torch.arange(num_rows_short, device=device) // next_n
    next_n_offset_short = torch.arange(num_rows_short, device=device) % next_n
    row_ends_short = (
        seq_lens_short[row_indices_short] - next_n + next_n_offset_short + 1
    )

    logits_short = create_random_logits(
        row_starts_short, row_ends_short, torch.float32, 42, clean_logits, "random"
    )

    indices_gems = torch.empty(
        (num_rows_short, top_k), dtype=torch.int32, device=device
    )

    # Use FlagGems kernel for short sequences
    top_k_per_row_decode(
        logits_short,
        next_n,
        seq_lens_short,
        indices_gems,
        num_rows_short,
        logits_short.stride(0),
        logits_short.stride(1),
        top_k,
    )

    # Reference implementation
    torch_indices_short = torch.empty(
        (num_rows_short, top_k), dtype=torch.int32, device=device
    )
    for i in range(num_rows_short):
        row_end = int(row_ends_short[i])
        k_i = min(top_k, row_end)
        idx = logits_short[i, :row_end].topk(k_i, dim=-1)[1]
        torch_indices_short[i, :k_i] = idx

    assert compare_top_k_results(
        logits_short,
        indices_gems,
        torch_indices_short,
        row_starts_short,
        row_ends_short,
        top_k,
    ), "top_k_per_row_decode kernel (short sequences) doesn't match torch.topk"
