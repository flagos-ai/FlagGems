import pytest
import torch

import flag_gems

from . import base, utils


class PadSequenceBenchmark(base.Benchmark):
    def get_input_iter(self, dtype):
        cases = [
            # small case
            [(64, 8), (32, 8)],
            [(128, 16), (256, 16), (100, 16), (200, 16)],
            # batch=8 variable length
            [(64, 8), (32, 8), (48, 8), (16, 8), (40, 8), (56, 8), (24, 8), (8, 8)],
            # medium transformer
            [(512, 768), (256, 768), (400, 768), (128, 768)],
            # transformer hidden 1024
            [(1024, 1024), (768, 1024), (512, 1024), (256, 1024)],
            # LLM hidden size
            [(2048, 4096), (1024, 4096), (1536, 4096), (512, 4096)],
            # larger batch (batch=16)
            [
                (512, 512),
                (384, 512),
                (256, 512),
                (128, 512),
                (448, 512),
                (320, 512),
                (192, 512),
                (64, 512),
                (400, 512),
                (300, 512),
                (200, 512),
                (100, 512),
                (256, 512),
                (128, 512),
                (96, 512),
                (32, 512),
            ],
            # batch=32
            [(512, 1024)] * 32,
        ]

        for shapes in cases:
            sequences = [
                utils.generate_tensor_input(shape, dtype, self.device)
                for shape in shapes
            ]
            yield (sequences, {"batch_first": False, "padding_value": 0.0})
            yield (sequences, {"batch_first": True, "padding_value": 0.0})


@pytest.mark.pad_sequence
def test_pad_sequence():
    bench = PadSequenceBenchmark(
        op_name="pad_sequence",
        torch_op=torch.nn.utils.rnn.pad_sequence,
        gems_op=flag_gems.pad_sequence,
        dtypes=[torch.float32, torch.float16, torch.bfloat16],
    )
    bench.run()
