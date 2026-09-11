import pytest
import torch

from flag_gems.fused.DSA.bin_topk import bucket_sort_topk

from . import base


class BucketSortTopKBenchmark(base.Benchmark):
    def __init__(self, op_name, torch_op, dtypes):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)

    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (1, 1024, 32),
            (4, 4096, 128),
            (8, 8192, 256),
            (16, 16384, 512),
            (32, 32768, 1024),
            (64, 32768, 2048),
            (96, 32768, 2048),
        ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            batch_size, seq_len, topk = shape
            inputs = torch.randn(batch_size, seq_len, dtype=dtype, device=self.device)
            starts = torch.zeros(batch_size, dtype=torch.int32, device=self.device)
            ends = (
                torch.ones(batch_size, dtype=torch.int32, device=self.device) * seq_len
            )
            yield (inputs, starts, ends, topk)


def _torch_topk_ref(inputs, starts, ends, topk):
    return torch.topk(inputs, topk, dim=-1)[1].to(torch.int32)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
@pytest.mark.bucket_sort_topk
def test_bucket_sort_topk_perf():
    bench = BucketSortTopKBenchmark(
        op_name="bucket_sort_topk",
        torch_op=_torch_topk_ref,
        dtypes=[torch.float32],
    )
    bench.set_gems(bucket_sort_topk)
    bench.run()
