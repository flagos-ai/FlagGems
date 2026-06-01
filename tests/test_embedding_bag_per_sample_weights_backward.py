import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.embedding_bag_per_sample_weights_backward
@pytest.mark.parametrize(
    "Batch, num_bags, bag_size, embedding_dim",
    [
        (2, 2, 3, 4),
        (4, 4, 8, 16),
        (1, 1, 10, 32),
        (3, 3, 5, 8),
    ],
)
# _embedding_bag_per_sample_weights_backward only supports float32 in PyTorch
@pytest.mark.parametrize("dtype", [torch.float32])
def test_embedding_bag_per_sample_weights_backward(
    Batch, num_bags, bag_size, embedding_dim, dtype
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    num_samples = Batch * bag_size
    num_embeddings = 64

    # Create inputs
    grad = torch.randn(num_bags, embedding_dim, dtype=dtype, device=flag_gems.device)
    weight = torch.randn(
        num_embeddings, embedding_dim, dtype=dtype, device=flag_gems.device
    )
    indices = torch.randint(
        0, num_embeddings, (num_samples,), device=flag_gems.device, dtype=torch.long
    )
    offsets = torch.tensor(
        [i * bag_size for i in range(num_bags + 1)],
        device=flag_gems.device,
        dtype=torch.long,
    )
    offset2bag = torch.tensor(
        [i // bag_size for i in range(num_samples)],
        device=flag_gems.device,
        dtype=torch.long,
    )

    ref_grad = utils.to_reference(grad)
    ref_weight = utils.to_reference(weight)
    ref_indices = utils.to_reference(indices)
    ref_offsets = utils.to_reference(offsets)
    ref_offset2bag = utils.to_reference(offset2bag)

    ref_out = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
        ref_grad, ref_weight, ref_indices, ref_offsets, ref_offset2bag, 0, -1
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten._embedding_bag_per_sample_weights_backward.default(
            grad, weight, indices, offsets, offset2bag, 0, -1
        )

    utils.gems_assert_close(res_out, ref_out, dtype)
