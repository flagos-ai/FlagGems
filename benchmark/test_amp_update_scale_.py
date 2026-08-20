from typing import Generator

import pytest
import torch

from . import base

# ``growth_interval`` is kept within the signed int32 range. For the no-inf
# path we use a large-but-safe interval so the scale never grows during the
# repeated latency measurement (keeping the benchmarked state stable). The
# backoff path is also stable (it underflows to 0 harmlessly), so these
# cases are safe to repeat in-place.
DEFAULT_SHAPES = [
    (1,),  # no-inf path, scale stays put
    (1,),  # inf-found path, scale backs off
    (1,),  # tracker one-below-interval path
]


DEFAULT_SHAPE_DESC = "scalar"


class AmpUpdateScaleBenchmark(base.Benchmark):
    """
    Benchmark for ``_amp_update_scale_``.

    The operator is a scalar (0-d) update: it mutates the current loss scale
    (a float32 scalar) and the growth tracker (an int32 scalar) according to a
    ``found_inf`` flag. Because the inputs are always scalars, the "shape" axis
    is not meaningful; we instead exercise representative input states.
    """

    # ``growth_interval`` is kept within the signed int32 range. For the no-inf
    # path we use a large-but-safe interval so the scale never grows during the
    # repeated latency measurement (keeping the benchmarked state stable). The
    # backoff path is also stable (it underflows to 0 harmlessly), so these
    # cases are safe to repeat in-place.
    DEFAULT_SHAPES = [
        (1,),  # no-inf path, scale stays put
        (1,),  # inf-found path, scale backs off
        (1,),  # tracker one-below-interval path
    ]
    DEFAULT_SHAPE_DESC = "scalar"

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype) -> Generator:
        # float32 is the only supported scale/found_inf dtype; ``dtype`` is
        # ignored here because the operator's inputs are fixed to float32 /
        # int32 scalars by the AMP contract.
        scale = torch.tensor(65536.0, device=self.device, dtype=torch.float32)
        tracker = torch.tensor(0, device=self.device, dtype=torch.int32)
        found_inf = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        # No-inf path: large-but-safe interval so the scale never grows.
        yield (
            scale.clone(),
            tracker.clone(),
            found_inf.clone(),
            2.0,
            0.5,
            2000000000,
        )

        # Inf-found path: scale backs off each call (underflows harmlessly).
        yield (
            scale.clone(),
            tracker.clone(),
            torch.tensor(1.0, device=self.device, dtype=torch.float32),
            2.0,
            0.5,
            1000,
        )

        # Tracker just below a small interval: exercises the growth branch.
        yield (
            torch.tensor(1024.0, device=self.device, dtype=torch.float32),
            torch.tensor(4, device=self.device, dtype=torch.int32),
            found_inf.clone(),
            2.0,
            0.5,
            5,
        )


@pytest.mark.amp_update_scale_
def test_amp_update_scale_():
    bench = AmpUpdateScaleBenchmark(
        op_name="amp_update_scale_",
        torch_op=torch._amp_update_scale_,
        # The operator only accepts float32 scale/found_inf and int32 tracker.
        dtypes=[torch.float32],
    )
    bench.run()
