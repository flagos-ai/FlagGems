import torch

import flag_gems

from .performance_utils import GenericBenchmark


class ConvTranspose2dBenchmark(GenericBenchmark):
    """
    Benchmark for conv_transpose2d operation.
    """

    def set_more_shapes(self):
        """Define test shapes for conv_transpose2d."""
        # Format: (batch, in_channels, height, width, out_channels_per_group, kernel_h, kernel_w, groups)
        self.shapes = [
            # Small sizes
            (1, 2, 8, 8, 1, 3, 3, 1),
            (2, 4, 8, 8, 2, 3, 3, 1),
            # Regular sizes - typical in neural networks
            (4, 8, 16, 16, 16, 3, 3, 1),
            (8, 16, 32, 32, 32, 3, 3, 1),
            (4, 32, 64, 64, 64, 3, 3, 1),
            (2, 64, 128, 128, 128, 3, 3, 1),
            # Large sizes
            (1, 128, 256, 256, 64, 3, 3, 1),
            (1, 256, 512, 512, 128, 3, 3, 1),
            # Different kernel sizes
            (4, 8, 16, 16, 16, 5, 5, 1),
            (4, 16, 32, 32, 32, 7, 7, 1),
            # With groups
            (4, 8, 16, 16, 4, 3, 3, 2),
            (4, 16, 32, 32, 8, 3, 3, 2),
            (4, 32, 64, 64, 4, 3, 3, 4),
        ]

    def get_input_iter(self, cur_dtype):
        """Return iterator over input tensors."""
        for shape in self.shapes:
            (
                batch,
                in_channels,
                height,
                width,
                out_channels_per_group,
                kernel_h,
                kernel_w,
                groups,
            ) = shape

            inp = torch.randn(
                (batch, in_channels, height, width),
                dtype=cur_dtype,
                device=self.device,
            )
            weight = torch.randn(
                (in_channels, out_channels_per_group, kernel_h, kernel_w),
                dtype=cur_dtype,
                device=self.device,
            )
            bias = torch.randn(
                out_channels_per_group * groups,
                dtype=cur_dtype,
                device=self.device,
            )

            # Yield: (input, weight, bias), then kwargs dict
            yield inp, weight, bias, {
                "stride": 2,
                "padding": 1,
                "output_padding": 0,
                "groups": groups,
                "dilation": 1,
            }

    def set_more_metrics(self):
        """Set additional metrics for benchmarking."""
        self.metrics = ["latency", "speedup", "accuracy"]
        return []  # Return empty list since we're setting self.metrics directly


def test_perf_conv_transpose2d():
    """Run performance benchmark for conv_transpose2d."""
    bench = ConvTranspose2dBenchmark(
        input_fn=lambda shape, dtype, device: None,  # Not used, we override get_input_iter
        op_name="conv_transpose2d",
        torch_op=torch.nn.functional.conv_transpose2d,
        dtypes=[torch.float16, torch.float32],
    )
    bench.set_gems(flag_gems.conv_transpose2d)
    bench.run()
