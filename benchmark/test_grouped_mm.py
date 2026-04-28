import random
from typing import Generator

import pytest
import torch

import flag_gems

from . import base, utils


class GroupmmBenchmark(base.BlasBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for groups, n, k in self.shapes:
            yield from self.input_fn(groups, n, k, dtype, self.device)

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        groups, N, K = args[1].shape
        size_per_group = torch.diff(
            args[2], prepend=torch.zeros(1, device="cuda", dtype=torch.int32)
        )
        total_flops = 0
        for i in range(groups):
            total_flops += size_per_group[i].item() * N * K * 2
        return total_flops


def _input_fn(groups, N, K, cur_dtype, device):
    assert cur_dtype == torch.bfloat16

    group_A_list = []
    group_B_list = []
    A_offs = 0
    B_offs = 0
    M_list = []
    for i in range(groups):
        M_g = random.randint(1, 16384)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device="cuda", dtype=cur_dtype)
        B_g = torch.rand([K_g, N_g], device="cuda", dtype=cur_dtype)
        group_A_list.append(A_g)
        group_B_list.append(B_g)
        M_list.append(M_g)
        A_offs += M_g * K_g
        B_offs += K_g * N_g

    mat_a = torch.cat([x for x in group_A_list], dim=0)
    mat_b = torch.stack([x for x in group_B_list], dim=0)
    offs = torch.tensor(
        [sum(M_list[: i + 1]) for i in range(groups)], dtype=torch.int32, device="cuda"
    )

    yield mat_a, mat_b, offs


@pytest.mark.grouped_mm
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm(monkeypatch):
    if flag_gems.vendor_name == "mthreads":
        monkeypatch.setenv("MUSA_ENABLE_SQMMA", "1")

    bench = GroupmmBenchmark(
        op_name="grouped_mm",
        input_fn=_input_fn,
        torch_op=torch._grouped_mm,
        gems_op=flag_gems.group_mm,
        dtypes=[torch.bfloat16],
    )

    bench.run()
