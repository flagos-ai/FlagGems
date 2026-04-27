import pytest
import torch

import flag_gems

from . import attri_util as attrs
from . import performance_utils as base
from . import utils

vendor_name = flag_gems.vendor_name


class RepetitionPenaltyBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name, torch_op, dtypes)
        self.gems_op = None

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1024),
            (1, 4096),
            (1, 8192),
            (8, 4096),
            (16, 4096),
            (32, 1024),
            (8, 8192),
            (64, 32000),
        ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            num_seqs, _ = shape
            yield (
                torch.randn(shape, dtype=dtype, device=self.device),
                torch.randint(0, 2, shape, dtype=torch.bool, device=self.device),
                torch.randint(0, 2, shape, dtype=torch.bool, device=self.device),
                torch.empty(num_seqs, dtype=dtype, device=self.device).uniform_(
                    1.0, 2.0
                ),
            )

    def set_gems(self, gems_op):
        self.gems_op = gems_op


UNSUPPORTED_VENDORS = {
    "metax",
    "kunlunxin",
    "iluvatar",
    "mthreads",
    "hygon",
    "cambricon",
}


@pytest.mark.skipif(utils.SkipVersion("vllm", "<0.4"), reason="vLLM <0.4 not supported")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(vendor_name in UNSUPPORTED_VENDORS, reason="Vendor not supported")
@pytest.mark.apply_repetition_penalties
def test_apply_repetition_penalties():
    vllm_ops = pytest.importorskip("vllm._custom_ops")

    bench = RepetitionPenaltyBenchmark(
        op_name="apply_repetition_penalties",
        torch_op=vllm_ops.apply_repetition_penalties,
        dtypes=attrs.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.apply_repetition_penalties)
    bench.run()
