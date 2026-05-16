from typing import Callable

import torch

from flag_gems.runtime.backend._ascend.ops.gather_ascend import gather
from flag_gems.runtime.backend._ascend.ops.gather_collapsed_uintdiv import (
    apply_prefix_narrows,
    can_collapse_axes,
    gather_collapsed,
)


def gather_auto(
    inp: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    out: torch.Tensor,
    grid_fn,
    magic_map=None,
    use_collapsed=False,
    with_negative_index=False,
) -> Callable[[], None]:
    ok, narrows = can_collapse_axes(inp, index, dim)
    if ok and use_collapsed:
        inp = apply_prefix_narrows(inp, narrows)
        run_kernel = gather_collapsed(
            inp,
            dim,
            index,
            out,
            grid_fn=grid_fn,
            return_run_kernel=True,
            with_negative_index=with_negative_index,
        )

    else:

        def run_kernel():
            gather(
                inp,
                dim,
                index,
                out,
                grid_fn=grid_fn,
                magic_map=magic_map,
                with_negative_index=with_negative_index,
            )

    return run_kernel
