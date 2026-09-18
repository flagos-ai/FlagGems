import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from .accuracy_utils import gems_assert_close

# ``_sobol_engine_scramble_`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._sobol_engine_scramble_`` and
# ``-m _sobol_engine_scramble_`` both work (same pattern as ``_reshape_alias``).
setattr(
    pytest.mark,
    "_sobol_engine_scramble_",
    MarkDecorator(
        Mark("_sobol_engine_scramble_", (), {}, _ispytest=True), _ispytest=True
    ),
)


@pytest.mark._sobol_engine_scramble_
@pytest.mark.parametrize("dimension", [1, 2, 3, 5, 10, 20])
def test_sobol_engine_scramble_(dimension):
    """Test _sobol_engine_scramble_ in-place operator."""
    MAXBIT = 30

    # Generate random binary inputs
    sobolstate = torch.randint(
        0, 2, (dimension, MAXBIT), dtype=torch.long, device=flag_gems.device
    )
    ltm = torch.randint(
        0, 2, (dimension, MAXBIT, MAXBIT), dtype=torch.long, device=flag_gems.device
    ).tril()

    # The reference (aten) implementation has no CUDA kernel, so run it on CPU.
    # to_reference() would upcast the integer state to float64, which aten's
    # scramble rejects, so move the int64 tensors to CPU directly.
    ref_sobolstate = sobolstate.clone().cpu()
    ref_ltm = ltm.cpu()

    ref_out = torch._sobol_engine_scramble_(ref_sobolstate, ref_ltm, dimension)

    # FlagGems computation: call the implementation directly (KernelGen tests
    # must not use the global dispatch override).
    res_out = flag_gems._sobol_engine_scramble_(sobolstate, ltm, dimension)

    # Verify return value is the same object (in-place)
    assert res_out is sobolstate

    gems_assert_close(sobolstate.cpu(), ref_sobolstate, dtype=torch.long)
    gems_assert_close(res_out.cpu(), ref_out.cpu(), dtype=torch.long)
