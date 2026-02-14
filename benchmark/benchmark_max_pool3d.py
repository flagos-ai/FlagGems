"""
Performance benchmark for max_pool3d operator.

This script benchmarks the 3D max pooling operation using FlagGems benchmark framework.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

# Import max_pool3d operator
from flag_gems.ops import max_pool3d

vendor_name = flag_gems.vendor_name


class MaxPool3DBenchmark(Benchmark):
    """
    Benchmark class for 3D max pooling operations.
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
        # Override shapes with 3D pooling specific shapes
        self.shapes = self.set_more_shapes()

    def set_shapes(self, shape_file_path=None):
        """
        Override set_shapes to prevent loading from shape file.
        MaxPool3D requires specific 5D shapes, not the default 2D shapes.
        """
        # Simply use the shapes already set in __init__
        pass

    def set_more_metrics(self):
        """Add bandwidth metric for pooling operations."""
        return ["gbps"]

    def get_gbps(self, args, latency):
        """
        Calculate effective bandwidth in GB/s.

        For pooling: input + output (both read and written)
        """
        from flag_gems.utils import shape_utils

        inp = args[0]
        # Output size is typically smaller than input due to pooling
        # Approximate output size (varies by kernel size, stride, padding)
        io_amount = (
            shape_utils.size_in_bytes(inp) + shape_utils.size_in_bytes(inp) * 0.5
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        """Define additional shapes for 3D pooling operations."""
        # Small sizes
        small_shapes = [
            (1, 1, 4, 4, 4),  # N=1, C=1, D=4, H=4, W=4
            (2, 3, 8, 8, 8),  # N=2, C=3, D=8, H=8, W=8
        ]
        # Medium sizes
        medium_shapes = [
            (2, 16, 32, 32, 32),  # N=2, C=16, D=32, H=32, W=32
            (4, 64, 64, 64, 64),  # N=4, C=64, D=64, H=64, W=64
        ]
        # Large sizes (reduced to avoid OOM with large kernels)
        large_shapes = [
            (2, 128, 128, 128, 128),  # N=2, C=128, D=128, H=128, W=128
        ]
        return small_shapes + medium_shapes + large_shapes

    def get_input_iter(self, cur_dtype):
        """Generate input tensors with various pooling parameters."""
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)

            # Default configuration
            kernel_size = 2
            yield inp, kernel_size

            # Various kernel sizes
            yield inp, 3
            yield inp, 4

            # Tuple kernel size
            yield inp, (2, 3, 4)

            # With stride
            yield inp, {"kernel_size": 2, "stride": 2}
            yield inp, {"kernel_size": 3, "stride": 1}

            # With padding (PyTorch: padding <= (kernel_size - 1) // 2)
            yield inp, {"kernel_size": 3, "padding": 1}
            yield inp, {"kernel_size": 7, "padding": 3}

            # With dilation (only on larger shapes to avoid output size = 0)
            # Skip for small shapes like (1, 1, 4, 4, 4)
            if shape[-1] >= 32:  # Only use dilation on shapes with spatial dim >= 32
                yield inp, {"kernel_size": 5, "dilation": 2, "padding": 2}

            # With return_indices
            yield inp, {"kernel_size": 2, "return_indices": True}

            # Large kernel performance test (user关心的场景)
            # Only on larger shapes to avoid output size = 0
            if shape[-1] >= 32:
                yield inp, {"kernel_size": 8, "padding": 4}


@pytest.mark.max_pool3d
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_max_pool3d(dtype):
    """Benchmark max_pool3d forward operation."""
    bench = MaxPool3DBenchmark(
        op_name="max_pool3d",
        torch_op=torch.nn.functional.max_pool3d,
        dtypes=[dtype],
    )
    # Set gems operator explicitly
    bench.set_gems(max_pool3d)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run benchmark
    print("\n" + "=" * 60)
    print("MAX_POOL3D PERFORMANCE BENCHMARK")
    print("=" * 60)
    test_perf_max_pool3d(torch.float32)

    print("\n" + "=" * 60)
    print("MAX_POOL3D PERFORMANCE BENCHMARK (float16)")
    print("=" * 60)
    test_perf_max_pool3d(torch.float16)

    print("\n" + "=" * 60)
    print("MAX_POOL3D PERFORMANCE BENCHMARK (bfloat16)")
    print("=" * 60)
    test_perf_max_pool3d(torch.bfloat16)
