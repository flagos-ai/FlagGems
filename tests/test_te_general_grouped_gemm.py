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

"""
Correctness tests for te_general_grouped_gemm operator.

Tests compare FlagGems Triton implementation against:
1. PyTorch reference implementation (torch.matmul)
2. TransformerEngine tex.te_general_grouped_gemm (when available)
"""

import pytest
import torch

from flag_gems.fused import te_general_grouped_gemm, general_grouped_gemm

# Try to import TransformerEngine for comparison
try:
    import transformer_engine.pytorch.cpp_extensions as tex
    from transformer_engine.common.recipe import DelayedScaling

    HAS_TE = True
except ImportError:
    HAS_TE = False
    tex = None

# Test configurations
QUICK_MODE = False

if QUICK_MODE:
    NUM_GEMMS_LIST = [2, 4]
    SHAPES = [
        (128, 128, 64),
        (256, 256, 128),
    ]
    DTYPES = [torch.float16]
else:
    NUM_GEMMS_LIST = [1, 2, 4, 8]
    SHAPES = [
        (64, 64, 32),
        (128, 128, 64),
        (256, 256, 128),
        (512, 512, 256),
        (1024, 1024, 512),
        (100, 200, 150),
        (333, 444, 555),
        (128, 512, 256),
        (512, 128, 256),
    ]
    DTYPES = [torch.float16, torch.bfloat16]

LAYOUTS = ["TN", "NN", "NT"]


def torch_grouped_gemm_reference(
    A_list: list,
    B_list: list,
    transa: bool = True,
    transb: bool = False,
    bias_list: list = None,
    gelu: bool = False,
) -> list:
    """PyTorch reference implementation for grouped GEMM."""
    results = []
    for i, (A, B) in enumerate(zip(A_list, B_list)):
        A_mat = A.T if transa else A
        B_mat = B.T if transb else B
        out = torch.matmul(A_mat.float(), B_mat.float())

        if bias_list is not None and i < len(bias_list) and bias_list[i].numel() > 0:
            out = out + bias_list[i].float()

        if gelu:
            out = torch.nn.functional.gelu(out, approximate="tanh")

        results.append(out.to(A.dtype))
    return results


def create_test_tensors(
    num_gemms: int,
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    transa: bool = True,
    transb: bool = False,
    device: str = "cuda",
    with_bias: bool = False,
):
    """Create test tensors for grouped GEMM."""
    A_list = []
    B_list = []
    out_list = []
    bias_list = []

    for i in range(num_gemms):
        if transa:
            A = torch.randn(K, M, dtype=dtype, device=device) * 0.1
        else:
            A = torch.randn(M, K, dtype=dtype, device=device) * 0.1

        if transb:
            B = torch.randn(N, K, dtype=dtype, device=device) * 0.1
        else:
            B = torch.randn(K, N, dtype=dtype, device=device) * 0.1

        out = torch.empty(M, N, dtype=dtype, device=device)

        A_list.append(A)
        B_list.append(B)
        out_list.append(out)

        if with_bias:
            bias = torch.randn(N, dtype=dtype, device=device) * 0.1
            bias_list.append(bias)

    return A_list, B_list, out_list, bias_list


def assert_close(actual, expected, dtype, rtol=None, atol=None):
    """Assert that two tensors are close."""
    if rtol is None:
        rtol = 1e-2 if dtype == torch.float16 else 1e-2
    if atol is None:
        atol = 1e-2 if dtype == torch.float16 else 1e-2
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


class TestTeGeneralGroupedGemmCorrectness:
    """Correctness tests comparing against PyTorch reference."""

    @pytest.mark.parametrize("num_gemms", NUM_GEMMS_LIST)
    @pytest.mark.parametrize("M,N,K", SHAPES)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("layout", LAYOUTS)
    def test_basic_grouped_gemm(self, num_gemms, M, N, K, dtype, layout):
        """Test basic grouped GEMM without bias or GELU."""
        transa = layout[0] == "T"
        transb = layout[1] == "T"

        A_list, B_list, out_list, _ = create_test_tensors(
            num_gemms, M, N, K, dtype, transa, transb, device="cuda", with_bias=False
        )

        # Prepare empty tensors for API
        empty_tensor = torch.tensor([], device="cuda")
        empty_tensors = [empty_tensor] * num_gemms
        workspace = [torch.zeros(1, dtype=torch.uint8, device="cuda")]

        # FlagGems implementation
        te_general_grouped_gemm(
            A=A_list,
            transa=transa,
            B=B_list,
            transb=transb,
            D=out_list,
            D_type=dtype,
            m_splits=[],
            bias=empty_tensors,
            bias_type=torch.bfloat16,
            single_output=False,
            pre_gelu_out=empty_tensors,
            grad=False,
            workspace=workspace,
            workspaceSize=1,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        # PyTorch reference
        out_ref = torch_grouped_gemm_reference(A_list, B_list, transa, transb)

        for i in range(num_gemms):
            assert_close(out_list[i], out_ref[i], dtype)

    @pytest.mark.parametrize("num_gemms", NUM_GEMMS_LIST[:2])
    @pytest.mark.parametrize("M,N,K", SHAPES[:4])
    @pytest.mark.parametrize("dtype", DTYPES[:1])
    def test_grouped_gemm_with_bias(self, num_gemms, M, N, K, dtype):
        """Test grouped GEMM with bias."""
        transa, transb = True, False

        A_list, B_list, out_list, bias_list = create_test_tensors(
            num_gemms, M, N, K, dtype, transa, transb, device="cuda", with_bias=True
        )

        empty_tensor = torch.tensor([], device="cuda")
        empty_tensors = [empty_tensor] * num_gemms
        workspace = [torch.zeros(1, dtype=torch.uint8, device="cuda")]

        te_general_grouped_gemm(
            A=A_list,
            transa=transa,
            B=B_list,
            transb=transb,
            D=out_list,
            D_type=dtype,
            m_splits=[],
            bias=bias_list,
            bias_type=dtype,
            single_output=False,
            pre_gelu_out=empty_tensors,
            grad=False,
            workspace=workspace,
            workspaceSize=1,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        out_ref = torch_grouped_gemm_reference(A_list, B_list, transa, transb, bias_list)

        for i in range(num_gemms):
            assert_close(out_list[i], out_ref[i], dtype)

    @pytest.mark.parametrize("num_gemms", NUM_GEMMS_LIST[:2])
    @pytest.mark.parametrize("M,N,K", SHAPES[:4])
    @pytest.mark.parametrize("dtype", DTYPES[:1])
    def test_grouped_gemm_with_gelu(self, num_gemms, M, N, K, dtype):
        """Test grouped GEMM with GELU activation."""
        transa, transb = True, False

        A_list, B_list, out_list, bias_list = create_test_tensors(
            num_gemms, M, N, K, dtype, transa, transb, device="cuda", with_bias=True
        )

        pre_gelu_list = [torch.empty(M, N, dtype=dtype, device="cuda") for _ in range(num_gemms)]
        workspace = [torch.zeros(1, dtype=torch.uint8, device="cuda")]

        te_general_grouped_gemm(
            A=A_list,
            transa=transa,
            B=B_list,
            transb=transb,
            D=out_list,
            D_type=dtype,
            m_splits=[],
            bias=bias_list,
            bias_type=dtype,
            single_output=False,
            pre_gelu_out=pre_gelu_list,
            grad=False,
            workspace=workspace,
            workspaceSize=1,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        out_ref = torch_grouped_gemm_reference(A_list, B_list, transa, transb, bias_list, gelu=True)

        for i in range(num_gemms):
            assert_close(out_list[i], out_ref[i], dtype, rtol=0.05, atol=0.05)

    @pytest.mark.parametrize("num_gemms", NUM_GEMMS_LIST[:2])
    @pytest.mark.parametrize("M,N,K", SHAPES[:3])
    @pytest.mark.parametrize("dtype", DTYPES[:1])
    def test_single_output_mode(self, num_gemms, M, N, K, dtype):
        """Test single output mode."""
        transa, transb = True, False
        m_splits = [M + i * 32 for i in range(num_gemms)]

        A_list = []
        B_list = []
        for i, m in enumerate(m_splits):
            A = torch.randn(K, m, dtype=dtype, device="cuda") * 0.1
            B = torch.randn(K, N, dtype=dtype, device="cuda") * 0.1
            A_list.append(A)
            B_list.append(B)

        total_M = sum(m_splits)
        out_tensor = torch.empty((total_M, N), dtype=dtype, device="cuda")

        empty_tensor = torch.tensor([], device="cuda")
        empty_tensors = [empty_tensor] * num_gemms
        workspace = [torch.zeros(1, dtype=torch.uint8, device="cuda")]

        te_general_grouped_gemm(
            A=A_list,
            transa=transa,
            B=B_list,
            transb=transb,
            D=[out_tensor],
            D_type=dtype,
            m_splits=m_splits,
            bias=empty_tensors,
            bias_type=torch.bfloat16,
            single_output=True,
            pre_gelu_out=empty_tensors,
            grad=False,
            workspace=workspace,
            workspaceSize=1,
            accumulate=False,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        out_ref = torch_grouped_gemm_reference(A_list, B_list, transa, transb)

        start_idx = 0
        for i, m in enumerate(m_splits):
            assert_close(out_tensor[start_idx : start_idx + m], out_ref[i], dtype)
            start_idx += m

    @pytest.mark.parametrize("num_gemms", NUM_GEMMS_LIST[:2])
    @pytest.mark.parametrize("M,N,K", SHAPES[:3])
    @pytest.mark.parametrize("dtype", DTYPES[:1])
    def test_accumulate_mode(self, num_gemms, M, N, K, dtype):
        """Test accumulation mode."""
        transa, transb = True, False

        A_list, B_list, out_list, _ = create_test_tensors(
            num_gemms, M, N, K, dtype, transa, transb, device="cuda", with_bias=False
        )

        for out in out_list:
            out.copy_(torch.randn_like(out) * 0.1)
        out_original = [o.clone() for o in out_list]

        empty_tensor = torch.tensor([], device="cuda")
        empty_tensors = [empty_tensor] * num_gemms
        workspace = [torch.zeros(1, dtype=torch.uint8, device="cuda")]

        te_general_grouped_gemm(
            A=A_list,
            transa=transa,
            B=B_list,
            transb=transb,
            D=out_list,
            D_type=dtype,
            m_splits=[],
            bias=empty_tensors,
            bias_type=torch.bfloat16,
            single_output=False,
            pre_gelu_out=empty_tensors,
            grad=False,
            workspace=workspace,
            workspaceSize=1,
            accumulate=True,
            use_split_accumulator=False,
            math_sm_count=0,
        )

        out_ref = torch_grouped_gemm_reference(A_list, B_list, transa, transb)

        for i in range(num_gemms):
            expected = out_original[i].float() + out_ref[i].float()
            assert_close(out_list[i], expected.to(dtype), dtype)


# Note: Direct comparison with tex.te_general_grouped_gemm is complex due to
# internal tensor wrapper handling in TE. The correctness tests above compare
# against PyTorch reference which validates the mathematical correctness.
# Performance comparison with TE is done in the benchmark file.


if __name__ == "__main__":
    print("Running quick correctness test...")

    num_gemms = 4
    M, N, K = 256, 256, 128
    dtype = torch.float16
    transa, transb = True, False

    A_list, B_list, out_list, _ = create_test_tensors(
        num_gemms, M, N, K, dtype, transa, transb, device="cuda", with_bias=False
    )

    empty_tensor = torch.tensor([], device="cuda")
    empty_tensors = [empty_tensor] * num_gemms
    workspace = [torch.zeros(1, dtype=torch.uint8, device="cuda")]

    te_general_grouped_gemm(
        A=A_list,
        transa=transa,
        B=B_list,
        transb=transb,
        D=out_list,
        D_type=dtype,
        m_splits=[],
        bias=empty_tensors,
        bias_type=torch.bfloat16,
        single_output=False,
        pre_gelu_out=empty_tensors,
        grad=False,
        workspace=workspace,
        workspaceSize=1,
        accumulate=False,
        use_split_accumulator=False,
        math_sm_count=0,
    )

    out_ref = torch_grouped_gemm_reference(A_list, B_list, transa, transb)

    max_diff = 0
    for i in range(num_gemms):
        diff = (out_list[i].float() - out_ref[i].float()).abs().max().item()
        max_diff = max(max_diff, diff)
        print(f"GEMM {i}: max diff = {diff:.6f}")

    print(f"\nOverall max diff: {max_diff:.6f}")
    print("Test PASSED!" if max_diff < 0.1 else "Test FAILED!")
