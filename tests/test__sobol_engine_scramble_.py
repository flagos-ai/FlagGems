import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close


# The registry id is ``underscore_sobol_engine_scramble_`` (the aten name starts
# with an underscore, naming Rule 2). CI derives the pytest marker from the
# implementation file name via the #6359 convention: ``_sobol_engine_scramble_``
# -> ``sobol_engine_scramble_`` (the stripped name is not itself a registered
# id). The rule-check marker gate instead compares against the registry id, so
# declare both names; ``pytest -m <either>`` then selects these tests.
@pytest.mark.sobol_engine_scramble_
@pytest.mark.underscore_sobol_engine_scramble_
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
