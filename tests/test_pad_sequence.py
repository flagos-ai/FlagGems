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

    result = flag_gems.pad_sequence(
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

    result = flag_gems.pad_sequence(
        sequences,
        batch_first=False,
        padding_value=5.0,
    )

    expected = torch.tensor(5.0, device=flag_gems.device)
    assert torch.all(result[2:, 1] == expected)


@pytest.mark.pad_sequence
def test_pad_sequence_empty_error():
    with pytest.raises(RuntimeError):
        flag_gems.pad_sequence([])


def _check_pad_sequence_boundary(sequences, batch_first, padding_value):
    ref_sequences = [utils.to_reference(seq) for seq in sequences]
    ref_out = torch.nn.utils.rnn.pad_sequence(
        ref_sequences, batch_first=batch_first, padding_value=padding_value
    )
    result = flag_gems.pad_sequence(
        sequences, batch_first=batch_first, padding_value=padding_value
    )
    assert result.shape == ref_out.shape
    assert result.dtype == sequences[0].dtype
    assert result.device == sequences[0].device
    assert result.is_contiguous()
    # Empty outputs have no values to compare; metadata is checked above.
    if result.numel() != 0:
        utils.gems_assert_close(result, ref_out, sequences[0].dtype)


@pytest.mark.pad_sequence
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(
    "max_length, tail",
    [
        pytest.param(127, (4, 4), id="3d-below-threshold"),
        pytest.param(128, (4, 4), id="3d-at-threshold"),
        pytest.param(129, (4, 4), id="3d-above-threshold"),
        pytest.param(129, (2, 2, 4), id="4d-above-threshold"),
        pytest.param(2049, (), id="1d-above-threshold"),
    ],
)
def test_pad_sequence_high_rank_boundary(max_length, tail, batch_first, dtype):
    # B=4, feature=16: totals 8128 / 8192 / 8256; 1D: total=8196.
    # The >8192 cases with batch_first=False must exercise generic batch_copy.
    lengths = [max_length, max_length - 13, max_length // 2, 1]
    sequences = [
        torch.randn((length, *tail), dtype=dtype, device=flag_gems.device)
        for length in lengths
    ]
    _check_pad_sequence_boundary(sequences, batch_first, padding_value=-2.5)


@pytest.mark.pad_sequence
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.float64, torch.bfloat16, torch.float16]
)
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(
    "seq_shapes",
    [
        pytest.param(
            [(length, 7) for length in (0, 1, 17, 3, 9, 0, 5, 11, 2)],
            id="batch9-2d-with-empty",
        ),
        pytest.param([(i % 5,) for i in range(16)], id="batch16-1d"),
        pytest.param([((i * 7) % 23, 3, 5) for i in range(32)], id="batch32-3d"),
        pytest.param([((i * 7) % 23, 7) for i in range(33)], id="batch33-2d"),
        pytest.param([(19, 17)] * 16, id="batch16-equal-length"),
    ],
)
def test_pad_sequence_large_batch_correctness(seq_shapes, batch_first, dtype):
    # B > 8 exercises generic flat; B=33 also crosses Hygon's group boundary.
    # A zero-length first sequence exercises repeated source offsets.
    sequences = [
        torch.randn(shape, dtype=dtype, device=flag_gems.device) for shape in seq_shapes
    ]
    _check_pad_sequence_boundary(sequences, batch_first, padding_value=-2.5)


@pytest.mark.pad_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("batch", [2, 3, 8, 9])
@pytest.mark.parametrize(
    "first_shape, other_shape",
    [
        pytest.param((65, 64), (31, 96), id="feature-mismatch"),
        pytest.param((65, 2, 32), (31, 64), id="ndim-mismatch"),
        pytest.param((65, 2, 32), (31, 4, 16), id="same-numel-different-tail"),
    ],
)
def test_pad_sequence_invalid_shape(first_shape, other_shape, batch, batch_first):
    # Keep the invalid sequence's per-step storage >= that of the first.
    # Missing generic validation can then be detected without requiring an OOB read.
    shapes = [first_shape] * batch
    shapes[1] = other_shape
    ref_sequences = [torch.randn(shape, dtype=torch.float32) for shape in shapes]
    with pytest.raises(RuntimeError):
        torch.nn.utils.rnn.pad_sequence(ref_sequences, batch_first=batch_first)
    sequences = [seq.to(device=flag_gems.device) for seq in ref_sequences]
    with pytest.raises(RuntimeError):
        flag_gems.pad_sequence(sequences, batch_first=batch_first)


@pytest.mark.pad_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("scalar_index", [0, 1])
def test_pad_sequence_scalar_error(scalar_index, batch_first):
    shapes = [(3, 4), (5, 4)]
    shapes[scalar_index] = ()
    ref_sequences = [torch.randn(shape, dtype=torch.float32) for shape in shapes]
    # Native may report either error depending on scalar position and version.
    with pytest.raises((RuntimeError, IndexError)):
        torch.nn.utils.rnn.pad_sequence(ref_sequences, batch_first=batch_first)
    sequences = [seq.to(device=flag_gems.device) for seq in ref_sequences]
    with pytest.raises(RuntimeError):
        flag_gems.pad_sequence(sequences, batch_first=batch_first)


@pytest.mark.pad_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("batch", [1, 2, 4, 9])
def test_pad_sequence_noncontiguous_boundary(batch, batch_first):
    sequences = [
        torch.randn(
            (129 - i * 7, 4, 16),
            dtype=torch.float32,
            device=flag_gems.device,
        )[..., ::2]
        for i in range(batch)
    ]
    assert all(not seq.is_contiguous() for seq in sequences)
    _check_pad_sequence_boundary(sequences, batch_first, padding_value=-2.5)


@pytest.mark.pad_sequence
@pytest.mark.parametrize("batch_first", [False, True])
def test_pad_sequence_mixed_device_error(batch_first):
    if torch.device(flag_gems.device).type == "cpu":
        pytest.skip("Requires an accelerator and CPU")
    # Follow the explicit mixed-device rejection in the Ascend/Hygon backends.
    # B=3 and a small output select generic's torch copy_ path if validation
    # is missing, avoiding passing a CPU pointer to an accelerator kernel.
    sequences = [
        torch.ones((5, 4), dtype=torch.float32, device=flag_gems.device),
        torch.ones((3, 4), dtype=torch.float32, device="cpu"),
        torch.ones((1, 4), dtype=torch.float32, device=flag_gems.device),
    ]
    with pytest.raises(RuntimeError):
        flag_gems.pad_sequence(sequences, batch_first=batch_first)


@pytest.mark.pad_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("batch", [2, 4, 9])
@pytest.mark.parametrize(
    "first_dtype, other_dtype",
    [
        (torch.float32, torch.float16),
        (torch.float16, torch.float32),
    ],
)
def test_pad_sequence_mixed_dtype(batch, batch_first, first_dtype, other_dtype):
    # Keep native reference dtypes unchanged: promotion by a reference helper
    # could hide the required conversion to the first tensor's dtype.
    ref_sequences = [
        torch.randn(
            (17 - i % 7, 7),
            dtype=other_dtype if i == batch - 1 else first_dtype,
        )
        for i in range(batch)
    ]
    ref_out = torch.nn.utils.rnn.pad_sequence(
        ref_sequences, batch_first=batch_first, padding_value=-2.5
    )
    sequences = [seq.to(device=flag_gems.device) for seq in ref_sequences]
    result = flag_gems.pad_sequence(
        sequences, batch_first=batch_first, padding_value=-2.5
    )
    assert result.dtype == first_dtype
    assert result.device == sequences[0].device
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), ref_out, rtol=0, atol=0)


@pytest.mark.pad_sequence
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize(
    "seq_shapes",
    [
        pytest.param([(0, 4)], id="single-empty"),
        pytest.param([(0, 4)] * 2, id="small-all-empty"),
        pytest.param([(0, 4)] * 3, id="direct-all-empty"),
        pytest.param([(0, 4)] * 9, id="flat-all-empty"),
        pytest.param([(3, 0), (5, 0)], id="small-zero-feature"),
        pytest.param([(3, 2, 0), (5, 2, 0), (1, 2, 0)], id="direct-zero-feature"),
        pytest.param([(i + 1, 0) for i in range(9)], id="flat-zero-feature"),
    ],
)
def test_pad_sequence_zero_output(seq_shapes, batch_first):
    sequences = [
        torch.empty(shape, dtype=torch.float32, device=flag_gems.device)
        for shape in seq_shapes
    ]
    _check_pad_sequence_boundary(sequences, batch_first, padding_value=-2.5)
