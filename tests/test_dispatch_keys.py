import pytest

import flag_gems


@pytest.mark.parametrize(
    "key",
    [
        # Regression guards for dispatch keys that were registered under a
        # non-existent ATen overload and therefore never dispatched (the op
        # silently fell back to eager instead of using the FlagGems kernel):
        #   greater.out       -> greater.Tensor_out
        #   geometric.float   -> geometric
        #   geometric_.float  -> geometric_
        "greater.Tensor_out",
        "geometric",
        "geometric_",
    ],
)
def test_fixed_dispatch_keys_registered(key):
    flag_gems.enable()
    assert (
        key in flag_gems.all_registered_keys()
    ), f"{key} is not registered; the FlagGems kernel would not be dispatched"
