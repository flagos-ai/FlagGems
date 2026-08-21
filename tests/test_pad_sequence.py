import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.pad_sequence
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float64, torch.bfloat16, torch.float16],
)
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(
    "seq_shapes",
    [
        # 2D features with varying batch sizes
        [(3, 4), (5, 4), (2, 4)],
        [(64, 8), (32, 8)],
        [(128, 16), (256, 16), (100, 16), (200, 16)],
        [(512, 32), (512, 32)],
        [(80, 4), (80, 4), (80, 4), (80, 4)],
        [(64, 8), (32, 8), (48, 8), (16, 8), (40, 8), (56, 8), (24, 8), (8, 8)],
        # 1D sequences
        [(10,), (20,), (30,)],
        # 3D features
        [(4, 3, 7), (6, 3, 7), (1, 3, 7)],
    ],
)
def test_pad_sequence_correctness(seq_shapes, batch_first, dtype):
    sequences = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) for shape in seq_shapes
    ]
    ref_sequences = [utils.to_reference(seq) for seq in sequences]

    ref_out = torch.nn.utils.rnn.pad_sequence(
        ref_sequences,
        batch_first=batch_first,
        padding_value=0.0,
    )

    with flag_gems.use_gems():
        result = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=batch_first,
            padding_value=0.0,
        )

    utils.gems_assert_close(result, ref_out, dtype)


@pytest.mark.pad_sequence
def test_pad_sequence_padding_value():
    sequences = [
        torch.ones((4, 3), device=flag_gems.device),
        torch.ones((2, 3), device=flag_gems.device),
    ]

    with flag_gems.use_gems():
        result = torch.nn.utils.rnn.pad_sequence(
            sequences,
            batch_first=False,
            padding_value=5.0,
        )

    expected = torch.tensor(5.0, device=flag_gems.device)
    assert torch.all(result[2:, 1] == expected)


@pytest.mark.pad_sequence
def test_pad_sequence_empty_error():
    with pytest.raises(RuntimeError):
        with flag_gems.use_gems():
            torch.nn.utils.rnn.pad_sequence([])
