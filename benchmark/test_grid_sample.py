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
Performance benchmark for grid_sample operator.

This script benchmarks the grid sample operation using FlagGems benchmark framework.
Grid sample is a spatial interpolation operation widely used in computer vision tasks.
"""

import pytest
import torch

import flag_gems
from flag_gems.utils import shape_utils

from . import base, consts, utils

vendor_name = flag_gems.vendor_name


class GridSampleBenchmark(base.Benchmark):
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
        # Reduced shapes to avoid CI benchmark timeout (original had 10 shapes,
        # each generating 8-14 mode combos × 3 dtypes = 300+ cases > 600s timeout)
        # Small sizes (4D)
        small_4d_shapes = [
            (1, 3, 32, 32),  # N=1, C=3, H=32, W=32
            # (2, 16, 32, 32),  # commented out to reduce CI timeout
        ]

        # Small sizes (5D)
        small_5d_shapes = [
            (1, 3, 8, 8, 8),  # N=1, C=3, D=8, H=8, W=8
            # (2, 4, 8, 8, 8),  # commented out to reduce CI timeout
        ]

        # Medium sizes (4D)
        medium_4d_shapes = [
            (2, 32, 64, 64),  # N=2, C=32, H=64, W=64
            # (4, 64, 64, 64),  # commented out to reduce CI timeout
        ]

        # Medium sizes (5D)
        medium_5d_shapes = [
            (2, 8, 16, 16, 16),  # N=2, C=8, D=16, H=16, W=16
            # (2, 16, 16, 16, 16),  # commented out to reduce CI timeout
        ]

        # Large sizes (4D) - commented out to reduce CI timeout
        large_4d_shapes = [
            # (4, 128, 128, 128),  # N=4, C=128, H=128, W=128 - too slow
        ]

        # Large sizes (5D) - commented out to reduce CI timeout
        large_5d_shapes = [
            # (2, 32, 32, 32, 32),  # N=2, C=32, D=32, H=32, W=32 - too slow
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
            inp = utils.generate_tensor_input(shape, cur_dtype, self.device)

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

            # Reduced mode combinations to avoid CI benchmark timeout.
            # Original had 11-14 yields per shape; now reduced to 4-5 core cases.

            # 1. Nearest neighbor - zeros padding
            yield inp, grid, {
                "mode": "nearest",
                "padding_mode": "zeros",
                "align_corners": False,
            }

            # 2. Bilinear - zeros padding (most common)
            yield inp, grid, {
                "mode": "bilinear",
                "padding_mode": "zeros",
                "align_corners": False,
            }

            # 3. Bilinear - border padding
            yield inp, grid, {
                "mode": "bilinear",
                "padding_mode": "border",
                "align_corners": True,
            }

            # 4. Bicubic (4D only)
            if not is_5d:
                yield inp, grid, {
                    "mode": "bicubic",
                    "padding_mode": "zeros",
                    "align_corners": False,
                }


@pytest.mark.grid_sample
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_grid_sample(dtype):
    """Benchmark grid_sample forward operation."""
    bench = GridSampleBenchmark(
        op_name="grid_sample",
        torch_op=torch.nn.functional.grid_sample,
        gems_op=flag_gems.grid_sample,
        dtypes=[dtype],
    )
    bench.run()
