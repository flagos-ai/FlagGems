#!/usr/bin/env python3
"""Emit one NVTX-ranged reference and route-mask EP combine launch."""

from __future__ import annotations

import torch

from compare_ep_sum_route_mask import (
    launch_mask_sum,
    make_routing,
    route_masks_from_mapping,
)
from flag_gems.fused.moe_sum import moe_sum_ep


def main():
    torch.manual_seed(20260824)
    m, topk, hidden, global_e, local_e = 96, 8, 4096, 288, 18
    inp = torch.randn((m, topk, hidden), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((m, hidden), device="cuda", dtype=torch.bfloat16)
    ids = make_routing("uniform", m, topk, global_e, local_e, inp.device)
    expert_map = torch.full((global_e,), -1, device="cuda", dtype=torch.int32)
    expert_map[:local_e] = torch.arange(local_e, device="cuda", dtype=torch.int32)
    masks = route_masks_from_mapping(ids, expert_map, local_e)

    def reference():
        moe_sum_ep(
            inp,
            out,
            ids,
            expert_map,
            local_e,
            fixed_block_size=512,
            fixed_num_warps=2,
        )

    def route_mask():
        launch_mask_sum(inp, out, masks, 256, 1, branch=True)

    for _ in range(3):
        reference()
        route_mask()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("ep_sum_reference")
    reference()
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("ep_sum_route_mask")
    route_mask()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
