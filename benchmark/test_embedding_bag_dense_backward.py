import pytest
import torch

from . import base, consts

# _embedding_bag_dense_backward benchmark
EMBEDDING_BAG_BACKWARD_SHAPES = [
    # (num_bags, embedding_dim, num_weights, num_samples_per_bag_avg)
    (8, 16, 50, 4),
    (16, 32, 100, 4),
    (32, 64, 100, 4),
    (64, 128, 200, 4),
    (128, 256, 500, 4),
    (256, 128, 500, 8),
    (512, 256, 1000, 8),
]


class EmbeddingBagDenseBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EMBEDDING_BAG_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for num_bags, embedding_dim, num_weights, samples_per_bag in self.shapes:
            num_samples = num_bags * samples_per_bag
            # Create weight (embedding table)
            weight = torch.randn(
                num_weights, embedding_dim, dtype=cur_dtype, device=self.device
            )
            # Create indices
            indices = torch.randint(
                0, num_weights, (num_samples,), dtype=torch.long, device=self.device
            )
            # Create offsets
            offsets = torch.arange(
                0,
                num_samples + 1,
                samples_per_bag,
                dtype=torch.long,
                device=self.device,
            )[:num_bags]
            # Forward pass to get required tensors
            (
                output,
                offset2bag,
                bag_size,
                maximum_indices,
            ) = torch.ops.aten._embedding_bag(
                weight, indices, offsets, False, 0, False, None, False, -1
            )
            # Generate random gradient
            grad = torch.randn_like(output)
            yield (
                grad,
                indices,
                offset2bag,
                bag_size,
                maximum_indices,
                num_weights,
                False,  # scale_grad_by_freq
                0,  # mode
                None,  # per_sample_weights
                -1,  # padding_idx
            )


@pytest.mark.embedding_bag_dense_backward
def test_embedding_bag_dense_backward():
    bench = EmbeddingBagDenseBackwardBenchmark(
        op_name="embedding_bag_dense_backward",
        torch_op=torch.ops.aten._embedding_bag_dense_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
