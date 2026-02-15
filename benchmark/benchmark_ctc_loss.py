"""
Performance benchmark for ctc_loss operator.

This script benchmarks CTC loss operation using FlagGems benchmark framework.
CTC loss is used for sequence-to-sequence learning tasks like speech recognition.

Note:
PyTorch's torch.nn.functional.ctc_loss only supports float32 dtype.
Therefore, this benchmark only tests float32 performance.
Reference: https://pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark

# Import ctc_loss operator
from flag_gems.ops import ctc_loss

vendor_name = flag_gems.vendor_name


class CTCLossBenchmark(Benchmark):
    """
    Benchmark class for ctc_loss operations.
    """

    def __init__(
        self,
        op_name,
        torch_op,
        dtypes=None,
        is_backward=False,
        is_inplace=False,
        **kwargs,
    ):
        # Initialize parent class
        super().__init__(
            op_name,
            torch_op,
            dtypes,
            is_backward,
            is_inplace,
            **kwargs,
        )
        # Override shapes with ctc_loss specific shapes
        self.shapes = self.set_more_shapes()

    def set_shapes(self, shape_file_path=None):
        """
        Override set_shapes to prevent loading from shape file.
        CTC loss requires specific (T, N, C) shapes.
        """
        # Simply use shapes already set in __init__
        pass

    def set_more_metrics(self):
        """Add bandwidth metric for ctc_loss operations."""
        return ["gbps"]

    def get_gbps(self, args, latency):
        """
        Calculate effective bandwidth in GB/s.

        For ctc_loss: log_probs + targets + output
        """
        from flag_gems.utils import shape_utils

        log_probs = args[0]
        targets = args[1]
        # Output size varies based on reduction mode
        output_size = log_probs.shape[1]  # N for 'none' reduction

        io_amount = (
            shape_utils.size_in_bytes(log_probs)
            + shape_utils.size_in_bytes(targets)
            + output_size * log_probs.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        """Define additional shapes for ctc_loss operations."""
        # Small sizes
        small_shapes = [
            (10, 2, 5),  # T=10, N=2, C=5
            (20, 4, 10),  # T=20, N=4, C=10
        ]

        # Medium sizes
        medium_shapes = [
            (50, 8, 20),  # T=50, N=8, C=20
            (100, 16, 30),  # T=100, N=16, C=30
        ]

        # Large sizes
        large_shapes = [
            (200, 32, 50),  # T=200, N=32, C=50
            (500, 64, 100),  # T=500, N=64, C=100
        ]

        # Extra large sizes (to match test file coverage)
        extra_large_shapes = [
            (1024, 8, 50),  # T=1024, N=8, C=50 (matches test requirement)
            (512, 16, 100),  # T=512, N=16, C=100
            (256, 8, 200),  # T=256, N=8, C=200 (large vocabulary)
        ]

        return small_shapes + medium_shapes + large_shapes + extra_large_shapes

    def get_input_iter(self, cur_dtype):
        """Generate input tensors with various ctc_loss parameters."""
        for T, N, C in self.shapes:
            # Generate log probabilities
            log_probs = torch.randn(T, N, C, dtype=cur_dtype, device=self.device)
            log_probs = log_probs.log_softmax(2)

            # Determine target length (typically much shorter than input)
            S = min(T // 2, 50)  # Cap at 50 to avoid excessive computation

            # Generate targets
            targets = torch.randint(1, C, (N, S), dtype=torch.long, device=self.device)

            # Generate input lengths (can vary)
            input_lengths = torch.randint(
                T // 2, T + 1, (N,), dtype=torch.long, device=self.device
            )

            # Generate target lengths (can vary)
            target_lengths = torch.randint(
                1, S + 1, (N,), dtype=torch.long, device=self.device
            )

            # Test parameter combinations

            # 1. Default parameters (most common)
            yield log_probs, targets, input_lengths, target_lengths, {
                "blank": 0,
                "reduction": "mean",
                "zero_infinity": False,
            }

            # 2. Different reduction modes
            yield log_probs, targets, input_lengths, target_lengths, {
                "blank": 0,
                "reduction": "none",
                "zero_infinity": False,
            }
            yield log_probs, targets, input_lengths, target_lengths, {
                "blank": 0,
                "reduction": "sum",
                "zero_infinity": False,
            }

            # 3. Different blank values
            if C > 1:
                yield log_probs, targets, input_lengths, target_lengths, {
                    "blank": 1,
                    "reduction": "mean",
                    "zero_infinity": False,
                }

            # 4. zero_infinity enabled
            yield log_probs, targets, input_lengths, target_lengths, {
                "blank": 0,
                "reduction": "mean",
                "zero_infinity": True,
            }

            # 5. Scalar input_lengths and target_lengths (only valid when N=1)
            if N == 1:
                scalar_input_len = torch.tensor(T, dtype=torch.long, device=self.device)
                scalar_target_len = torch.tensor(
                    S, dtype=torch.long, device=self.device
                )
                yield log_probs, targets, scalar_input_len, scalar_target_len, {
                    "blank": 0,
                    "reduction": "mean",
                    "zero_infinity": False,
                }


@pytest.mark.ctc_loss
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_ctc_loss(dtype):
    """Benchmark ctc_loss forward operation."""
    bench = CTCLossBenchmark(
        op_name="ctc_loss",
        torch_op=torch.nn.functional.ctc_loss,
        dtypes=[dtype],
    )
    # Set gems operator explicitly
    bench.set_gems(ctc_loss)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run benchmark
    print("\n" + "=" * 80)
    print("CTC_LOSS PERFORMANCE BENCHMARK")
    print("=" * 80)
    test_perf_ctc_loss(torch.float32)
