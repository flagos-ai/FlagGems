"""
Performance benchmark for grid_sample operator.

This script benchmarks the grid sample operation using FlagGems benchmark framework.
Grid sample is a spatial interpolation operation widely used in computer vision tasks.
"""

import pytest
import torch

import flag_gems
from benchmark.attri_util import FLOAT_DTYPES
from benchmark.performance_utils import Benchmark, generate_tensor_input

# Import grid_sample operator
from flag_gems.ops import grid_sample

vendor_name = flag_gems.vendor_name


class GridSampleBenchmark(Benchmark):
    """
    Benchmark class for grid_sample operations.
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
        # Override shapes with grid_sample specific shapes
        self.shapes = self.set_more_shapes()

    def set_shapes(self, shape_file_path=None):
        """
        Override set_shapes to prevent loading from shape file.
        Grid_sample requires specific 4D and 5D shapes.
        """
        # Simply use the shapes already set in __init__
        pass

    def set_more_metrics(self):
        """Add bandwidth metric for grid_sample operations."""
        return ["gbps"]

    def get_gbps(self, args, latency):
        """
        Calculate effective bandwidth in GB/s.

        For grid_sample: input + grid + output
        """
        from flag_gems.utils import shape_utils

        inp = args[0]
        grid = args[1]
        # Output size varies based on grid dimensions
        output_size = (
            grid.numel() * inp.shape[1]
        )  # N * H_out * W_out * C (for 4D) or similar for 5D
        io_amount = (
            shape_utils.size_in_bytes(inp)
            + shape_utils.size_in_bytes(grid)
            + output_size * inp.element_size()
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def set_more_shapes(self):
        """Define additional shapes for grid_sample operations."""
        # Small sizes (4D)
        small_4d_shapes = [
            (1, 3, 32, 32),  # N=1, C=3, H=32, W=32
            (2, 16, 32, 32),  # N=2, C=16, H=32, W=32
        ]

        # Small sizes (5D)
        small_5d_shapes = [
            (1, 3, 8, 8, 8),  # N=1, C=3, D=8, H=8, W=8
            (2, 4, 8, 8, 8),  # N=2, C=4, D=8, H=8, W=8
        ]

        # Medium sizes (4D)
        medium_4d_shapes = [
            (2, 32, 64, 64),  # N=2, C=32, H=64, W=64
            (4, 64, 64, 64),  # N=4, C=64, H=64, W=64
        ]

        # Medium sizes (5D)
        medium_5d_shapes = [
            (2, 8, 16, 16, 16),  # N=2, C=8, D=16, H=16, W=16
            (2, 16, 16, 16, 16),  # N=2, C=16, D=16, H=16, W=16
        ]

        # Large sizes (4D)
        large_4d_shapes = [
            (4, 128, 128, 128),  # N=4, C=128, H=128, W=128
        ]

        # Large sizes (5D)
        large_5d_shapes = [
            (2, 32, 32, 32, 32),  # N=2, C=32, D=32, H=32, W=32
        ]

        return (
            small_4d_shapes
            + small_5d_shapes
            + medium_4d_shapes
            + medium_5d_shapes
            + large_4d_shapes
            + large_5d_shapes
        )

    def get_input_iter(self, cur_dtype):
        """Generate input tensors with various grid_sample parameters."""
        for shape in self.shapes:
            inp = generate_tensor_input(shape, cur_dtype, self.device)

            # Determine if 4D or 5D
            is_5d = len(shape) == 5

            if is_5d:
                # 5D: (N, C, D_in, H_in, W_in) -> grid (N, D_out, H_out, W_out, 3)
                N, C, D_in, H_in, W_in = shape
                # Create output grid dimensions (can be different from input)
                D_out, H_out, W_out = D_in, H_in, W_in  # Same size

                # Generate random grid in valid range [-1, 1]
                grid = torch.randn(
                    N, D_out, H_out, W_out, 3, dtype=cur_dtype, device=self.device
                )
                grid = torch.clamp(
                    grid, -0.9, 0.9
                )  # Keep away from boundaries for main test
            else:
                # 4D: (N, C, H_in, W_in) -> grid (N, H_out, W_out, 2)
                N, C, H_in, W_in = shape
                H_out, W_out = H_in, W_in

                # Generate random grid in valid range [-1, 1]
                grid = torch.randn(
                    N, H_out, W_out, 2, dtype=cur_dtype, device=self.device
                )
                grid = torch.clamp(
                    grid, -0.9, 0.9
                )  # Keep away from boundaries for main test

            # Test mode combinations (focus on most common/useful cases)
            # Note: Yield format is (input, grid, {kwargs}) so grid is in args[1] for get_gbps

            # 1. Nearest neighbor (fastest, commonly used)
            yield inp, grid, {
                "mode": "nearest",
                "padding_mode": "zeros",
                "align_corners": False,
            }
            yield inp, grid, {
                "mode": "nearest",
                "padding_mode": "border",
                "align_corners": False,
            }
            yield inp, grid, {
                "mode": "nearest",
                "padding_mode": "reflection",
                "align_corners": False,
            }

            # 2. Bilinear / Trilinear (most common interpolation)
            yield inp, grid, {
                "mode": "bilinear",
                "padding_mode": "zeros",
                "align_corners": False,
            }
            yield inp, grid, {
                "mode": "bilinear",
                "padding_mode": "border",
                "align_corners": False,
            }
            yield inp, grid, {
                "mode": "bilinear",
                "padding_mode": "reflection",
                "align_corners": False,
            }

            # 3. Bicubic (4D only, higher quality)
            if not is_5d:
                yield inp, grid, {
                    "mode": "bicubic",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }
                yield inp, grid, {
                    "mode": "bicubic",
                    "padding_mode": "border",
                    "align_corners": False,
                }
                yield inp, grid, {
                    "mode": "bicubic",
                    "padding_mode": "reflection",
                    "align_corners": False,
                }

            # 4. Test with align_corners=True
            yield inp, grid, {
                "mode": "nearest",
                "padding_mode": "zeros",
                "align_corners": True,
            }
            yield inp, grid, {
                "mode": "bilinear",
                "padding_mode": "zeros",
                "align_corners": True,
            }
            if not is_5d:
                yield inp, grid, {
                    "mode": "bicubic",
                    "padding_mode": "zeros",
                    "align_corners": True,
                }

            # 5. Upsampling scenario (2x)
            if is_5d:
                D_out, H_out, W_out = D_in * 2, H_in * 2, W_in * 2
            else:
                H_out, W_out = H_in * 2, W_in * 2

            if is_5d:
                grid_upsample = torch.randn(
                    N, D_out, H_out, W_out, 3, dtype=cur_dtype, device=self.device
                )
                grid_upsample = torch.clamp(grid_upsample, -0.9, 0.9)
                yield inp, grid_upsample, {
                    "mode": "bilinear",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }
            else:
                grid_upsample = torch.randn(
                    N, H_out, W_out, 2, dtype=cur_dtype, device=self.device
                )
                grid_upsample = torch.clamp(grid_upsample, -0.9, 0.9)
                yield inp, grid_upsample, {
                    "mode": "bilinear",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }
                yield inp, grid_upsample, {
                    "mode": "bicubic",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }

            # 6. Downsampling scenario (2x)
            if is_5d:
                D_out, H_out, W_out = (
                    max(1, D_in // 2),
                    max(1, H_in // 2),
                    max(1, W_in // 2),
                )
            else:
                H_out, W_out = max(1, H_in // 2), max(1, W_in // 2)

            if is_5d:
                grid_downsample = torch.randn(
                    N, D_out, H_out, W_out, 3, dtype=cur_dtype, device=self.device
                )
                grid_downsample = torch.clamp(grid_downsample, -0.9, 0.9)
                yield inp, grid_downsample, {
                    "mode": "bilinear",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }
            else:
                grid_downsample = torch.randn(
                    N, H_out, W_out, 2, dtype=cur_dtype, device=self.device
                )
                grid_downsample = torch.clamp(grid_downsample, -0.9, 0.9)
                yield inp, grid_downsample, {
                    "mode": "bilinear",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }
                yield inp, grid_downsample, {
                    "mode": "bicubic",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }


@pytest.mark.grid_sample
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_perf_grid_sample(dtype):
    """Benchmark grid_sample forward operation."""
    bench = GridSampleBenchmark(
        op_name="grid_sample",
        torch_op=torch.nn.functional.grid_sample,
        dtypes=[dtype],
    )
    # Set gems operator explicitly
    bench.set_gems(grid_sample)
    bench.run()


if __name__ == "__main__":
    print(f"Using device: {flag_gems.device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Run benchmark
    print("\n" + "=" * 80)
    print("GRID_SAMPLE PERFORMANCE BENCHMARK")
    print("=" * 80)
    test_perf_grid_sample(torch.float32)

    print("\n" + "=" * 80)
    print("GRID_SAMPLE PERFORMANCE BENCHMARK (float16)")
    print("=" * 80)
    test_perf_grid_sample(torch.float16)

    print("\n" + "=" * 80)
    print("GRID_SAMPLE PERFORMANCE BENCHMARK (bfloat16)")
    print("=" * 80)
    test_perf_grid_sample(torch.bfloat16)
