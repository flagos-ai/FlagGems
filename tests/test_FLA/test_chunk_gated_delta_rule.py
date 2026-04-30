"""
Test suite for chunk_gated_delta_rule operator.

This test module validates correctness by comparing FlagGems implementation
against the reference implementation from GatedDeltaNet.
"""

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from flag_gems.fused import chunk_gated_delta_rule

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(project_root)
sys.path.append(project_root)

# Manually create GatedDeltaNet module hierarchy to work around circular import
gated_delta_net = types.ModuleType("GatedDeltaNet")
gated_delta_net.__path__ = [os.path.join(project_root, "GatedDeltaNet")]
sys.modules["GatedDeltaNet"] = gated_delta_net

lit_gpt = types.ModuleType("GatedDeltaNet.lit_gpt")
lit_gpt.__path__ = [os.path.join(project_root, "GatedDeltaNet", "lit_gpt")]
sys.modules["GatedDeltaNet.lit_gpt"] = lit_gpt

# Try to import both reference implementations
GATEDDELTANET_AVAILABLE = False
FLA_AVAILABLE = False
reference_gateddnet = None
reference_fla = None

try:
    from GatedDeltaNet.lit_gpt.gated_delta_rule_ops.chunk import (
        chunk_gated_delta_rule as reference_gateddnet,
    )

    GATEDDELTANET_AVAILABLE = True
    print("GatedDeltaNet reference implementation available.")
except (ImportError, ModuleNotFoundError) as e:
    print(f"GatedDeltaNet not available: {e}")

try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule as reference_fla

    FLA_AVAILABLE = True
    print("FLA reference implementation available.")
except (ImportError, ModuleNotFoundError) as e:
    print(f"FLA not available: {e}")


REFERENCE_AVAILABLE = GATEDDELTANET_AVAILABLE or FLA_AVAILABLE

BACKENDS_TO_TEST = []
# Determine which backend(s) to test based on availability
if GATEDDELTANET_AVAILABLE:
    # When GatedDeltaNet is available, only test GatedDeltaNet backend
    BACKENDS_TO_TEST.append("gateddnet")
    print("GatedDeltaNet is available")
if FLA_AVAILABLE:
    # When only FLA is available, only test FLA backend
    BACKENDS_TO_TEST.append("fla")
    print("FLA is available")

# ============================================================================
# Precision Standards
# ============================================================================

FLOAT_DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

ATOL_DICT = {
    torch.float16: 1e-3,
    torch.float32: 1.3e-6,
    torch.bfloat16: 0.016,
}

RTOL = 1e-4


# ============================================================================
# Reference Implementation Selection
# ============================================================================


def get_reference_impl(dtype, backend="gateddnet"):
    """
    Get appropriate reference implementation based on dtype, backend, and availability.

    Returns:
        tuple: (reference_function, reference_type)
            - reference_function: The reference implementation function
            - reference_type: String indicating which implementation ("GatedDeltaNet", "FLA", or "FlagGems-Optimized")
    """

    # Select reference based on backend
    if backend == "fla" and FLA_AVAILABLE:
        return reference_fla, "FLA"
    elif backend == "gateddnet" and GATEDDELTANET_AVAILABLE:
        return reference_gateddnet, "GatedDeltaNet"

    # Fallback: prefer GatedDeltaNet over FLA for float16/bfloat16
    if GATEDDELTANET_AVAILABLE:
        return reference_gateddnet, "GatedDeltaNet"
    elif FLA_AVAILABLE:
        return reference_fla, "FLA"
    else:
        return None, None


def call_reference_impl(
    ref_func, ref_type, q, k, v, g, beta, BT, initial_state, output_final_state, scale
):
    """
    Call reference implementation with proper parameter order and format.

    Handles differences between GatedDeltaNet and FLA interfaces:
    - GatedDeltaNet: (q, k, v, beta, g, BT, ...) with head-first layout [B, H, T, K/V]
    - FLA: (q, k, v, g, beta, scale, ...) with time-first layout [B, T, H, K/V]

    IMPORTANT: The input tensors are in the format expected by the backend being tested,
    not necessarily the format expected by the reference implementation.
    We need to convert both input and output to match the reference implementation's expectations.
    """

    if ref_type == "GatedDeltaNet":
        ref_output, ref_final = ref_func(
            q,
            k,
            v,
            beta,
            g,
            BT=BT,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    elif ref_type == "FLA":
        # FLA uses (q, k, v, g, beta) order and scale parameter
        ref_output, ref_final = ref_func(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    else:
        raise ValueError(f"Unknown reference type: {ref_type}")

    return ref_output, ref_final


# ============================================================================
# Helper Functions
# ============================================================================


def assert_close(actual, expected, rtol=RTOL, atol=None, dtype=torch.float32):
    """Use torch.allclose to verify precision."""
    if atol is None:
        atol = ATOL_DICT.get(dtype, 1e-5)

    assert torch.allclose(
        actual, expected, rtol=rtol, atol=atol, equal_nan=True
    ), f"Results don't match: max diff={(actual - expected).abs().max().item()}"


def create_tensor(shape, dtype, device="cuda", requires_grad=False):
    """Create test tensor."""
    x = torch.randn(shape, dtype=dtype, device=device)
    if requires_grad:
        x.requires_grad_(True)
    return x


def create_test_tensors(backend, B, T, H, K, V, dtype):
    """
    Create test tensors in the format expected by each backend.

    This avoids stride differences from transpose operations that cause
    Triton autotuner to select different kernel configurations.
    """
    if backend == "gateddnet":
        # GagedDeltaNet backend: use head-first format [B, H, T, K/V]
        q = create_tensor((B, H, T, K), dtype)
        k = F.normalize(
            torch.randn((B, H, T, K), dtype=dtype, device="cuda"), p=2, dim=-1
        )
        v = create_tensor((B, H, T, V), dtype)
        beta = torch.rand((B, H, T), dtype=dtype, device="cuda").sigmoid()
        g = F.logsigmoid(torch.randn((B, H, T), dtype=dtype, device="cuda"))
    else:  # 'fla'
        # FLA backend: use time-first format [B, T, H, K/V]
        q = create_tensor((B, T, H, K), dtype)
        k = F.normalize(
            torch.randn((B, T, H, K), dtype=dtype, device="cuda"), p=2, dim=-1
        )
        v = create_tensor((B, T, H, V), dtype)
        beta = torch.rand((B, T, H), dtype=dtype, device="cuda").sigmoid()
        g = F.logsigmoid(torch.randn((B, T, H), dtype=dtype, device="cuda"))

    return q, k, v, beta, g


def create_normalized_tensor(shape, dtype, device="cuda", requires_grad=False):
    """Create L2-normalized test tensor (for keys as expected by gated_delta_rule)."""
    import torch.nn.functional as F

    x = torch.randn(shape, dtype=dtype, device=device)
    x = F.normalize(x, p=2, dim=-1)
    if requires_grad:
        x.requires_grad_(True)
    return x


# ============================================================================
# Comparison Tests with Reference Implementation
# ============================================================================


class TestChunkGatedDeltaRuleComparison:
    """Test chunk_gated_delta_rule against reference implementation."""

    @pytest.fixture(autouse=True)
    def cleanup_cuda(self):
        """Clear CUDA cache before each test to ensure clean state."""
        torch.cuda.empty_cache()
        yield
        torch.cuda.empty_cache()

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_matches_reference_small(self, dtype, backend):
        """Test forward output matches reference implementation - small inputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get appropriate reference implementation based on backend
        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V = 2, 128, 4, 64, 32
        BT = 64  # chunk size

        # Create identical inputs for both implementations
        # Note: The gated_delta_rule operator expects:
        # 1. L2-normalized keys (k)
        # 2. Gates (g) in log space, typically using logsigmoid to keep values bounded
        # 3. FlagGems-Optimized version uses (q, k, v, g, beta) order
        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone inputs for reference implementation
        q_ref = q.clone()
        k_ref = k.clone()
        v_ref = v.clone()
        beta_ref = beta.clone()
        g_ref = g.clone()

        # Reference implementation using unified wrapper
        try:
            ref_output, ref_final = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems implementation
        _, flag_output, _, flag_final, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape
        assert (
            flag_output.shape == ref_output.shape
        ), f"Output shape mismatch: flag={flag_output.shape}, ref={ref_output.shape}"
        assert flag_output.dtype == dtype, f"Output dtype mismatch: {flag_output.dtype}"

        # Verify precision (competition standard)
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

        # Verify final state if both return it
        if flag_final is not None and ref_final is not None:
            assert (
                flag_final.shape == ref_final.shape
            ), f"Final state shape mismatch: flag={flag_final.shape}, ref={ref_final.shape}"

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_matches_reference_medium(self, dtype, backend):
        """Test forward output matches reference - medium inputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get appropriate reference implementation based on backend
        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip("No reference implementation available")

        B, T, H, K, V = 4, 256, 8, 128, 64
        BT = 64

        # Note: The gated_delta_rule operator expects:
        # 1. L2-normalized keys (k)
        # 2. Gates (g) in log space, typically using logsigmoid to keep values bounded
        # 3. FlagGems-Optimized version uses (q, k, v, g, beta) order
        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation using unified wrapper
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_matches_reference_with_state(self, dtype, backend):
        """Test forward output matches reference with initial state."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Clear CUDA cache to ensure clean state
        torch.cuda.empty_cache()

        # Get appropriate reference implementation based on backend
        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip("No reference implementation available")

        B, T, H, K, V = 2, 128, 4, 64, 32
        BT = 64

        # Note: The gated_delta_rule operator expects:
        # 1. L2-normalized keys (k)
        # 2. Gates (g) in log space, typically using logsigmoid to keep values bounded
        # 3. FlagGems-Optimized version uses (q, k, v, g, beta) order
        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)
        initial_state = create_tensor((B, H, K, V), dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]
        initial_state_ref = initial_state.clone()

        # Reference implementation using unified wrapper
        try:
            ref_output, ref_final = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                initial_state_ref,
                True,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, flag_final, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=initial_state,
            output_final_state=True,
            backend=backend,
            BT=BT,
        )

        # Check if reference implementation produces NaN (known FLA bug with initial_state)
        # FLA backend with initial_state can produce NaN values - skip test in this case
        ref_has_nan = torch.isnan(ref_output).any()
        ref_final_has_nan = ref_final is not None and torch.isnan(ref_final).any()

        if ref_has_nan or ref_final_has_nan:
            pytest.skip(
                f"Reference implementation produces NaN values "
                f"(known FLA issue with initial_state: output_has_nan={ref_has_nan}, final_has_nan={ref_final_has_nan})"
            )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

        # Verify final state
        if flag_final is not None and ref_final is not None:
            assert flag_final.shape == ref_final.shape
            assert_close(
                flag_final, ref_final, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
            )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_matches_reference_large(self, dtype, backend):
        """Test forward output matches reference - large inputs."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get appropriate reference implementation based on backend
        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip("No reference implementation available")

        B, T, H, K, V = 4, 512, 8, 128, 64
        BT = 64

        # Note: The gated_delta_rule operator expects:
        # 1. L2-normalized keys (k)
        # 2. Gates (g) in log space, typically using logsigmoid to keep values bounded
        # 3. FlagGems-Optimized version uses (q, k, v, g, beta) order
        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation using unified wrapper
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize(
        "T,BT",
        [
            (1024, 64),  # Large sequence: T=1024, BT=64
            (2048, 64),  # Extra large sequence: T=2048, BT=64
        ],
    )
    def test_matches_reference_large_size(self, dtype, backend, T, BT):
        """Test forward output matches reference - large sizes (competition requirement)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Check GPU memory before large test
        gpu_memory_available = torch.cuda.get_device_properties(0).total_memory
        # Estimate memory: B*T*H*K/V * 4 bytes * ~8 tensors
        estimated_memory = 2 * T * 4 * 64 * 4 * 8

        if gpu_memory_available < estimated_memory * 2:
            pytest.skip(
                f"Insufficient GPU memory ({gpu_memory_available / 1024**3:.1f}GB) "
                f"for T={T} test (need ~{estimated_memory / 1024**3:.1f}GB)"
            )

        # Get appropriate reference implementation based on backend
        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, H, K, V = 2, 4, 64, 32

        # Note: The gated_delta_rule operator expects:
        # 1. L2-normalized keys (k)
        # 2. Gates (g) in log space, typically using logsigmoid to keep values bounded
        # 3. FlagGems-Optimized version uses (q, k, v, g, beta) order
        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation using unified wrapper
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip(f"Reference OOM during large test: {str(e)[:100]}")
            pytest.skip(f"Reference implementation failed: {e}")
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )


class TestChunkGatedDeltaRuleParameters:
    """Test chunk_gated_delta_rule with different parameter combinations."""

    @pytest.fixture(autouse=True)
    def cleanup_cuda(self):
        """Clear CUDA cache before each test to ensure clean state."""
        torch.cuda.empty_cache()
        yield
        torch.cuda.empty_cache()

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize("BT", [16, 32, 64, 128, 256])
    def test_different_bt_values(self, dtype, backend, BT):
        """Test forward output with different BT (chunk size) values."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get appropriate reference implementation based on backend
        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        # Choose T that is divisible by all BT values
        B, T, H, K, V = 2, 512, 4, 64, 32
        # Ensure T is divisible by BT
        if T % BT != 0:
            T = ((T // BT) + 1) * BT

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation using unified wrapper
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    @pytest.mark.parametrize(
        "K,V",
        [
            (32, 16),  # Small head dimensions
            (64, 32),  # Already tested in basic tests
            (128, 64),  # Already tested in basic tests
            (256, 128),  # Large head dimensions (max supported)
            (128, 32),  # Asymmetric: K > V
            (64, 128),  # Asymmetric: V > K
        ],
    )
    def test_different_kv_dimensions(self, dtype, backend, K, V):
        """Test forward output with different K and V dimension combinations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Get appropriate reference implementation based on backend
        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, BT = 2, 256, 4, 64

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation using unified wrapper
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )


class TestChunkGatedDeltaBoundaryCases:
    """Test chunk_gated_delta_rule with boundary and edge cases."""

    @pytest.fixture(autouse=True)
    def cleanup_cuda(self):
        """Clear CUDA cache before each test to ensure clean state."""
        torch.cuda.empty_cache()
        yield
        torch.cuda.empty_cache()

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_all_zeros(self, dtype, backend):
        """Test with all zero tensors."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V, BT = 2, 128, 4, 64, 32, 64

        # Create all-zero tensors
        q = torch.zeros((B, T, H, K), dtype=dtype, device="cuda")
        k = F.normalize(
            torch.zeros((B, T, H, K), dtype=dtype, device="cuda"), p=2, dim=-1
        )
        v = torch.zeros((B, T, H, V), dtype=dtype, device="cuda")
        beta = torch.zeros((B, T, H), dtype=dtype, device="cuda")
        g = torch.zeros((B, T, H), dtype=dtype, device="cuda")

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape
        assert flag_output.shape == ref_output.shape
        # For zero inputs, outputs should be close (but may not be exact due to normalization)
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_nan_in_q(self, dtype, backend):
        """Test with NaN values in query tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V, BT = 2, 128, 4, 64, 32, 64

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Insert NaN in query tensor
        q.view(-1)[0] = float("nan")

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems - expect NaN in output
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
        )

        # Check if outputs have NaN (both should propagate NaN)
        flag_has_nan = torch.isnan(flag_output).any()
        ref_has_nan = torch.isnan(ref_output).any()

        # Both should have NaN or both should not have NaN
        assert (
            flag_has_nan == ref_has_nan
        ), f"NaN handling mismatch: flag_gems has NaN={flag_has_nan}, reference has NaN={ref_has_nan}"

        # If both have NaN, verify non-NaN values match
        if not flag_has_nan and not ref_has_nan:
            atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
            assert_close(
                flag_output,
                ref_output,
                rtol=RTOL,
                atol=atol_for_dtype * 10,
                dtype=dtype,
            )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_inf_in_beta(self, dtype, backend):
        """Test with Inf values in beta tensor."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V, BT = 2, 128, 4, 64, 32, 64

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Insert Inf in beta tensor
        beta.view(-1)[0] = float("inf")

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape
        assert flag_output.shape == ref_output.shape

        # Check if outputs have Inf/NaN (both should handle similarly)
        flag_has_inf = torch.isinf(flag_output).any()
        ref_has_inf = torch.isinf(ref_output).any()

        assert (
            flag_has_inf == ref_has_inf
        ), f"Inf handling mismatch: flag_gems has Inf={flag_has_inf}, reference has Inf={ref_has_inf}"

        # If neither has Inf, verify values match
        if not flag_has_inf and not ref_has_inf:
            atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
            # Relax tolerance for edge case with inf in input
            assert_close(
                flag_output,
                ref_output,
                rtol=RTOL * 10,
                atol=atol_for_dtype * 100,
                dtype=dtype,
            )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_minimal_batch(self, dtype, backend):
        """Test with minimal batch size (B=1)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V, BT = 1, 128, 4, 64, 32, 64

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_single_head(self, dtype, backend):
        """Test with single head (H=1)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V, BT = 2, 128, 1, 64, 32, 64

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output,
            ref_output,
            rtol=RTOL,
            atol=atol_for_dtype,  # * atol_multiplier,
            dtype=dtype,
        )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_small_sequence(self, dtype, backend):
        """Test with small sequence length (T=16 < BT=64)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V, BT = 2, 16, 4, 64, 32, 64

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )

    @pytest.mark.chunk_gated_delta_rule
    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    @pytest.mark.parametrize("backend", BACKENDS_TO_TEST)
    def test_kv_dimension_1(self, dtype, backend):
        """Test with minimal K and V dimensions (K=1, V=1)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        ref_func, ref_type = get_reference_impl(dtype, backend)
        if ref_func is None:
            pytest.skip(f"No reference implementation available for backend={backend}")

        B, T, H, K, V, BT = 2, 128, 4, 1, 1, 64

        torch.manual_seed(42)
        q, k, v, beta, g = create_test_tensors(backend, B, T, H, K, V, dtype)

        # Clone for reference
        q_ref, k_ref, v_ref, beta_ref, g_ref = [x.clone() for x in [q, k, v, beta, g]]

        # Reference implementation
        try:
            ref_output, _ = call_reference_impl(
                ref_func,
                ref_type,
                q_ref,
                k_ref,
                v_ref,
                g_ref,
                beta_ref,
                BT,
                None,
                False,
                K**-0.5,
            )
        except Exception as e:
            pytest.skip(f"Reference implementation failed: {e}")

        # FlagGems
        _, flag_output, _, _, _, _, _ = chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            scale=K**-0.5,
            initial_state=None,
            output_final_state=False,
            backend=backend,
            BT=BT,
        )

        # Verify output shape and precision
        assert flag_output.shape == ref_output.shape
        atol_for_dtype = ATOL_DICT.get(dtype, 1e-5)
        assert_close(
            flag_output, ref_output, rtol=RTOL, atol=atol_for_dtype * 10, dtype=dtype
        )
