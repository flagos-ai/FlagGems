import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.embedding_bag
# num_bags / embedding_dim chosen to cover small and medium bag counts and
# both power-of-two and non-power-of-two embedding widths.
@pytest.mark.parametrize("num_bags", [3, 8, 16])
@pytest.mark.parametrize("embedding_dim", [16, 32])
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_embedding_bag(num_bags, embedding_dim, mode, dtype):
    """Test embedding_bag forward accuracy across sum/mean/max modes."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    num_weights = 50
    num_samples = num_bags * 3  # Average 3 samples per bag

    weight = torch.randn(
        num_weights, embedding_dim, dtype=dtype, device=flag_gems.device
    )
    indices = torch.randint(
        0, num_weights, (num_samples,), dtype=torch.long, device=flag_gems.device
    )
    # Regular offsets: split samples evenly into bags, first offset is 0.
    samples_per_bag = max(1, num_samples // num_bags)
    offsets = torch.arange(
        0, num_samples, samples_per_bag, dtype=torch.long, device=flag_gems.device
    )[:num_bags]
    offsets[0] = 0

    # per_sample_weights is only supported for sum mode (mode=0) in aten.
    per_sample_weights = None
    if mode == 0:
        per_sample_weights = torch.rand(
            num_samples, dtype=dtype, device=flag_gems.device
        )

    ref_weight = utils.to_reference(weight, True)
    # indices/offsets are integer index tensors: move to the reference device
    # (when TO_CPU) but do not upcast them to float.
    ref_indices = utils.to_reference(indices)
    ref_offsets = utils.to_reference(offsets)
    ref_psw = utils.to_reference(per_sample_weights, True)
    ref_out = utils.to_reference(
        torch.ops.aten.embedding_bag(
            ref_weight, ref_indices, ref_offsets, False, mode, False, ref_psw, False
        )[0]
    )
    with flag_gems.use_gems():
        res_out = torch.ops.aten.embedding_bag(
            weight, indices, offsets, False, mode, False, per_sample_weights, False
        )[0]

    utils.gems_assert_close(res_out, ref_out, dtype)
