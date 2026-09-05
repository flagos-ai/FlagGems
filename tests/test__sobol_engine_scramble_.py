import pytest
import torch

import flag_gems

from .accuracy_utils import gems_assert_close, to_reference


@pytest.mark.sobol_engine_scramble
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

    # Clone for reference
    ref_sobolstate = to_reference(sobolstate.clone(), True)
    ref_ltm = to_reference(ltm, True)

    # Reference computation
    ref_out = torch._sobol_engine_scramble_(ref_sobolstate, ref_ltm, dimension)

    # FlagGems computation (no use_gems() for KernelGen operators)
    res_out = torch._sobol_engine_scramble_(sobolstate, ltm, dimension)

    # Verify the mutated input matches reference
    gems_assert_close(sobolstate, ref_sobolstate, dtype=torch.long)

    # Verify return value is the same object (in-place)
    assert res_out is sobolstate

    # Verify return value matches reference
    gems_assert_close(res_out, ref_out, dtype=torch.long)
