import gc

import pytest
import torch

import flag_gems

from . import base
from .conftest import Config, emit_record_logger, update_result
from .consts import BenchmarkMetrics, BenchmarkResult, OperationAttribute

# Representative (n, k/2) uint8 shapes for int4-packed weights. The native
# aten::_convert_weight_to_int4pack contract requires n % 8 == 0 and
# k = k_half * 2 divisible by innerKTiles * 16; every shape below satisfies
# this for innerKTiles in {2, 4, 8}.
CONVERT_WEIGHT_SHAPES = [
    (16, 64),
    (16, 128),
    (32, 128),
    (64, 256),
]

INNER_K_TILES = (2, 4, 8)


def _convert_weight_input_fn(shape, device):
    """Yield paired inputs for native and gems from one int4 weight source.

    Native aten::_convert_weight_to_int4pack takes uint8 [n, k/2] (two int4
    values per byte) and produces a Marlin-style tiled int32 tensor; the
    FlagGems implementation takes int32 [n, k] (one int4 value per element)
    and produces byte-pair-packed uint8 [n, k/2]. Both inputs are derived
    from the same logical int4 weight [n, k] so the comparison is meaningful,
    and the conversion happens here, outside the timed region.
    """
    n, k_half = shape
    k = k_half * 2
    int4 = torch.randint(0, 16, (n, k), dtype=torch.int32, device=device)
    base_input = ((int4[:, 1::2] << 4) | int4[:, 0::2]).to(torch.uint8)
    gems_input = int4
    for innerKTiles in INNER_K_TILES:
        yield (base_input, innerKTiles), (gems_input, innerKTiles)


class ConvertWeightBenchmark(base.Benchmark):
    """Benchmark for convert_weight_to_int4pack with mismatched native/gems
    input formats.

    Because the two ops cannot share one input, get_input_iter yields paired
    inputs and _run_metric dispatches each to its own op. Inputs are built
    outside the timed region so neither latency is distorted by conversion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gems_op = flag_gems._convert_weight_to_int4pack

    def set_shapes(self, shape_file_path=None):
        self.shapes = CONVERT_WEIGHT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from _convert_weight_input_fn(shape, self.device)

    def _run_metric(self, input_item):
        metric = BenchmarkMetrics()
        base_input, gems_input = input_item
        base_args = list(base_input)
        gems_args = list(gems_input)
        metric.shape_detail = self.record_shapes(*gems_args)
        try:
            if "latency_base" in self.to_bench_metrics:
                metric.latency_base = self.get_latency(self.torch_op, *base_args)
            if "latency" in self.to_bench_metrics:
                metric.latency = self.get_latency(self.gems_op, *gems_args)
            if "speedup" in self.to_bench_metrics:
                metric.speedup = metric.latency_base / metric.latency
        except (RuntimeError, Exception) as e:
            metric.error_msg = str(e)
            pytest.fail(str(e))
        return metric

    def run(self):
        if Config.query:
            self.init_default_config()
            attri = OperationAttribute(
                op_name=self.op_name,
                recommended_core_shapes=self.shapes,
                shape_desc=self.shape_desc,
            )
            print(attri)
            emit_record_logger(attri.to_dict())
            return

        self.init_user_config()
        for dtype in self.to_bench_dtypes:
            metrics = []
            input_iter = self.get_input_iter(dtype)
            done = False
            while not done:
                try:
                    input_item = next(input_iter)
                except StopIteration:
                    done = True
                    continue
                except (RuntimeError, Exception) as e:
                    print(
                        f"\033[31mFAILED\033[0m: Operator={self.op_name} "
                        f"dtype={dtype} err=<<<{e}>>>"
                    )
                    pytest.fail(str(e))
                metric = self._run_metric(input_item)
                metrics.append(metric)
                gc.collect()

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)
            update_result(self.op_name, result.to_json())
            emit_record_logger(result.to_json())


@pytest.mark.convert_weight_to_int4pack
def test_convert_weight_to_int4pack():
    # torch_op is the native aten op (the baseline Caeruleann asked for);
    # gems_op is set in __init__. dtype=torch.uint8 matches the native input
    # contract; the gems input is derived from the same int4 source.
    bench = ConvertWeightBenchmark(
        op_name="convert_weight_to_int4pack",
        torch_op=torch._convert_weight_to_int4pack,
        dtypes=[torch.uint8],
    )
    bench.run()
