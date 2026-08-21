import pytest
import torch

from . import base, consts

# embedding_bag benchmark
# (num_bags, embedding_dim, num_weights, num_samples_per_bag_avg)
EMBEDDING_BAG_SHAPES = [
    (8, 16, 50, 4),
    (16, 32, 100, 4),
    (32, 64, 100, 4),
    (64, 128, 200, 4),
    (128, 256, 500, 4),
    (256, 128, 500, 8),
    (512, 256, 1000, 8),
]


class EmbeddingBagBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = EMBEDDING_BAG_SHAPES

    def get_input_iter(self, cur_dtype):
        for num_bags, embedding_dim, num_weights, samples_per_bag in self.shapes:
            num_samples = num_bags * samples_per_bag
            weight = torch.randn(
                num_weights, embedding_dim, dtype=cur_dtype, device=self.device
            )
            indices = torch.randint(
                0, num_weights, (num_samples,), dtype=torch.long, device=self.device
            )
            offsets = torch.arange(
                0,
                num_samples,
                samples_per_bag,
                dtype=torch.long,
                device=self.device,
            )[:num_bags]
            yield (
                weight,
                indices,
                offsets,
                False,  # scale_grad_by_freq
                0,  # mode (sum)
                False,  # sparse
                None,  # per_sample_weights
                False,  # include_last_offset
            )


@pytest.mark.embedding_bag
def test_embedding_bag():
    bench = EmbeddingBagBenchmark(
        op_name="embedding_bag",
        torch_op=torch.ops.aten.embedding_bag,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
