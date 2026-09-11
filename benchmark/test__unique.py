import pytest
import torch
from triton.testing import do_bench

import flag_gems

from . import consts


@pytest.mark.unique
def test__unique():
    """Benchmark _unique with shapes suitable for unique operations"""

    # Use smaller shapes for unique operations since they involve sorting
    shapes = [
        (1024,),
        (4096,),
        (16384,),
        (65536,),
        (262144,),
        (1048576,),
    ]

    print("\nOperator: _unique  Performance Test")

    for dtype in consts.FLOAT_DTYPES:
        print(f"\n--- dtype={dtype} ---")
        for shape in shapes:
            try:
                # Create input with some duplicates
                inp = torch.randn(shape, dtype=dtype, device="cuda")

                # Benchmark torch version
                torch_latency = do_bench(lambda: torch._unique(inp))

                # Benchmark gems version
                gems_latency = do_bench(lambda: flag_gems._unique(inp))

                speedup = torch_latency / gems_latency
                gbps = (inp.numel() * inp.element_size()) / gems_latency / 1e6

                result = (
                    f"SUCCESS    {torch_latency:.6f}    "
                    f"{gems_latency:.6f}    {speedup:.3f}    "
                    f"{gbps:.2f}    {list(shape)}"
                )
                print(result)
            except Exception as e:
                print(f"FAILED     {shape}    {str(e)}")
