import numpy as np
import pytest
import torch

from . import base, consts

# Jagged to padded dense forward benchmark
JAGGED_TO_PADDED_SHAPES = [
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 64),
    (256, 128),
    (512, 256),
]


class JaggedToPaddedDenseForwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = JAGGED_TO_PADDED_SHAPES

    def get_input_iter(self, cur_dtype):
        for batch_size, max_length in self.shapes:
            # Generate random sequence lengths
            np.random.seed(42)
            seq_lengths = np.random.randint(1, max_length + 1, size=batch_size).tolist()

            # Create offsets tensor (cumulative)
            offsets = [0] + list(np.cumsum(seq_lengths).astype(int).tolist())
            offsets = torch.tensor(offsets, device=self.device, dtype=torch.int64)

            # Create values tensor (concatenated sequences)
            total_length = sum(seq_lengths)
            values = torch.randn(total_length, dtype=cur_dtype, device=self.device)

            yield values, [offsets], [max_length], 0.0


@pytest.mark.jagged_to_padded_dense_forward
def test_jagged_to_padded_dense_forward():
    bench = JaggedToPaddedDenseForwardBenchmark(
        op_name="jagged_to_padded_dense_forward",
        torch_op=torch.ops.aten._jagged_to_padded_dense_forward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
