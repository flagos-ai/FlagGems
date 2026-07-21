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
Test for te_general_grouped_gemm operator.
This test compares FlagGems implementation with TransformerEngine reference.
"""

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def get_te_reference():
    """Get TransformerEngine reference implementation if available."""
    try:
        import transformer_engine_torch as tex

        return tex.te_general_grouped_gemm
    except ImportError:
        pytest.skip("TransformerEngine not available")


def get_flaggems_impl():
    """Get FlagGems implementation."""
    from flag_gems.ops.te_general_grouped_gemm import te_general_grouped_gemm

    return te_general_grouped_gemm


class TestTeGeneralGroupedGemm:
    """Test cases for te_general_grouped_gemm operator."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    @pytest.mark.parametrize(
        "num_gemms,m,n,k",
        [
            (2, 32, 64, 128),
            (4, 64, 128, 256),
            (3, 128, 256, 512),
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("layout", ["TN", "NN"])
    def test_basic_grouped_gemm(self, num_gemms, m, n, k, dtype, layout):
        """
        Test basic grouped GEMM functionality.

        For TN layout: C = A^T @ B where A is (K, M), B is (K, N), C is (M, N)
        For NN layout: C = A @ B where A is (M, K), B is (K, N), C is (M, N)
        """
        flaggems_impl = get_flaggems_impl()

        # Create input tensors based on layout
        transa = layout[0] == "T"
        transb = layout[1] == "T"

        A_list = []
        B_list = []
        out_list = []

        for _ in range(num_gemms):
            if transa:
                A = torch.randn(k, m, dtype=dtype, device=self.device)
            else:
                A = torch.randn(m, k, dtype=dtype, device=self.device)

            if transb:
                B = torch.randn(n, k, dtype=dtype, device=self.device)
            else:
                B = torch.randn(k, n, dtype=dtype, device=self.device)

            out = torch.zeros(m, n, dtype=dtype, device=self.device)

            A_list.append(A)
            B_list.append(B)
            out_list.append(out)

        # Compute reference using torch.matmul
        ref_outputs = []
        for i in range(num_gemms):
            A_mat = A_list[i].T if transa else A_list[i]
            B_mat = B_list[i].T if transb else B_list[i]
            ref_out = torch.matmul(A_mat, B_mat)
            ref_outputs.append(ref_out)

        # Call FlagGems implementation
        flaggems_impl(
            A_list,
            transa,
            B_list,
            transb,
            out_list,
        )

        # Verify results
        for i in range(num_gemms):
            torch.testing.assert_close(
                out_list[i],
                ref_outputs[i],
                rtol=1e-2,
                atol=1e-2,
                msg=f"Mismatch at GEMM {i}",
            )

    @pytest.mark.parametrize(
        "shapes",
        [
            # (M, N, K) for each GEMM - varying M
            [(32, 64, 128), (48, 64, 128), (16, 64, 128)],
            # Varying all dimensions
            [(32, 64, 128), (64, 128, 256), (128, 32, 64)],
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_varying_shapes(self, shapes, dtype):
        """Test grouped GEMM with varying shapes across GEMMs."""
        flaggems_impl = get_flaggems_impl()

        # Use TN layout: A is (K, M), B is (K, N), C is (M, N)
        A_list = []
        B_list = []
        out_list = []
        ref_outputs = []

        for m, n, k in shapes:
            A = torch.randn(k, m, dtype=dtype, device=self.device)
            B = torch.randn(k, n, dtype=dtype, device=self.device)
            out = torch.zeros(m, n, dtype=dtype, device=self.device)

            A_list.append(A)
            B_list.append(B)
            out_list.append(out)

            # Reference: C = A^T @ B
            ref_outputs.append(torch.matmul(A.T, B))

        # Call FlagGems implementation (TN layout)
        flaggems_impl(
            A_list,
            True,  # transa
            B_list,
            False,  # transb
            out_list,
        )

        # Verify results
        for i in range(len(shapes)):
            torch.testing.assert_close(
                out_list[i],
                ref_outputs[i],
                rtol=1e-2,
                atol=1e-2,
                msg=f"Mismatch at GEMM {i}",
            )

    @pytest.mark.parametrize("accumulate", [False, True])
    def test_accumulate(self, accumulate):
        """Test accumulate flag."""
        flaggems_impl = get_flaggems_impl()
        dtype = torch.float16
        num_gemms = 2
        m, n, k = 32, 64, 128

        A_list = []
        B_list = []
        out_list = []
        initial_values = []

        for _ in range(num_gemms):
            A = torch.randn(k, m, dtype=dtype, device=self.device)
            B = torch.randn(k, n, dtype=dtype, device=self.device)
            initial = torch.randn(m, n, dtype=dtype, device=self.device)
            out = initial.clone()

            A_list.append(A)
            B_list.append(B)
            out_list.append(out)
            initial_values.append(initial)

        # Call FlagGems implementation (TN layout)
        flaggems_impl(
            A_list,
            True,  # transa
            B_list,
            False,  # transb
            out_list,
            accumulate=accumulate,
        )

        # Verify results
        for i in range(num_gemms):
            gemm_result = torch.matmul(A_list[i].T, B_list[i])
            if accumulate:
                expected = initial_values[i] + gemm_result
            else:
                expected = gemm_result

            torch.testing.assert_close(
                out_list[i],
                expected,
                rtol=1e-2,
                atol=1e-2,
                msg=f"Mismatch at GEMM {i} with accumulate={accumulate}",
            )

    def test_single_gemm(self):
        """Test with single GEMM (edge case)."""
        flaggems_impl = get_flaggems_impl()
        dtype = torch.float16
        m, n, k = 64, 128, 256

        A = torch.randn(k, m, dtype=dtype, device=self.device)
        B = torch.randn(k, n, dtype=dtype, device=self.device)
        out = torch.zeros(m, n, dtype=dtype, device=self.device)

        # Call FlagGems implementation (TN layout)
        flaggems_impl(
            [A],
            True,  # transa
            [B],
            False,  # transb
            [out],
        )

        # Reference
        ref_out = torch.matmul(A.T, B)

        torch.testing.assert_close(
            out,
            ref_out,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_empty_input(self):
        """Test with empty input list."""
        flaggems_impl = get_flaggems_impl()

        # Should handle empty lists gracefully
        result = flaggems_impl(
            [],  # empty A list
            True,
            [],  # empty B list
            False,
            [],  # empty out list
        )

        # Result should be empty or None
        assert result is None or result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
