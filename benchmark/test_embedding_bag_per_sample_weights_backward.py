import pytest
import torch

from . import base

# Custom shapes for embedding bag backward: (Batch, num_bags, bag_size, embedding_dim)
EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD_SHAPES = [
    (2, 2, 3, 4),  # Batch, num_bags, bag_size, embedding_dim
    (4, 4, 8, 16),
    (1, 1, 10, 32),
    (3, 3, 5, 8),
]


class EmbeddingBagPerSampleWeightsBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EMBEDDING_BAG_PER_SAMPLE_WEIGHTS_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for Batch, num_bags, bag_size, embedding_dim in self.shapes:
            num_samples = Batch * bag_size
            num_embeddings = 64

            grad = torch.randn(
                num_bags, embedding_dim, dtype=cur_dtype, device=self.device
            )
            weight = torch.randn(
                num_embeddings, embedding_dim, dtype=cur_dtype, device=self.device
            )
            indices = torch.randint(
                0, num_embeddings, (num_samples,), device=self.device, dtype=torch.long
            )
            offsets = torch.tensor(
                [i * bag_size for i in range(num_bags + 1)],
                device=self.device,
                dtype=torch.long,
            )
            offset2bag = torch.tensor(
                [i // bag_size for i in range(num_samples)],
                device=self.device,
                dtype=torch.long,
            )
            yield grad, weight, indices, offsets, offset2bag, 0, -1


@pytest.mark.embedding_bag_per_sample_weights_backward
def test_embedding_bag_per_sample_weights_backward():
    bench = EmbeddingBagPerSampleWeightsBackwardBenchmark(
        op_name="embedding_bag_per_sample_weights_backward",
        torch_op=torch.ops.aten._embedding_bag_per_sample_weights_backward,
        # _embedding_bag_per_sample_weights_backward only supports float32 in PyTorch
        dtypes=[torch.float32],
    )
    bench.run()
