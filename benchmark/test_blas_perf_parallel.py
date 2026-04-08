import concurrent.futures
import fcntl
import gc
import os
import pickle
import subprocess
import sys
import tempfile

import pytest
import torch
import yaml

import benchmark.test_blas_perf as blas_perf
import flag_gems
from benchmark.attri_util import BenchmarkMetrics, BenchmarkResult, OperationAttribute
from benchmark.conftest import Config, emit_record_logger

PARALLEL_WORKER_ENV = "FLAGGEMS_BENCH_PARALLEL_WORKER"
PARALLEL_RESULT_FILE_ENV = "FLAGGEMS_BENCH_RESULT_FILE"
torch_device_object = flag_gems.runtime.backend.gen_torch_device_object()


def _parallel_device_is_available():
    if hasattr(torch_device_object, "is_available"):
        return torch_device_object.is_available()
    return False


def _parallel_device_count():
    if hasattr(torch_device_object, "device_count"):
        return torch_device_object.device_count()
    return 0


def _parallel_visible_devices_env():
    env_name_map = {
        "cuda": "CUDA_VISIBLE_DEVICES",
        "musa": "MUSA_VISIBLE_DEVICES",
    }
    return env_name_map.get(flag_gems.device)


def _parallel_device_label():
    return flag_gems.device.upper()


class ParallelBenchmarkMixin:
    SHAPE_CONFIG_KEYS = ()

    def set_more_shapes(self):
        if os.environ.get(PARALLEL_WORKER_ENV):
            return []
        return super().set_more_shapes()

    def get_parallel_metric_group_size(self, shape):
        return 1

    def should_forward_parallel_dtype(self, dtype_name):
        return True

    def _iter_expected_shapes(self):
        for shape in self.shapes:
            group_size = max(1, int(self.get_parallel_metric_group_size(shape)))
            for _ in range(group_size):
                yield tuple(shape) if isinstance(shape, (list, tuple)) else shape

    def _get_error_shape_output_path(self):
        return os.path.abspath("FlagTune/error_shape.yaml")

    def _record_failed_shape(self, shape):
        if shape is None:
            return

        normalized_shape = list(shape) if isinstance(shape, (list, tuple)) else [shape]
        error_shape_path = self._get_error_shape_output_path()
        output_dir = os.path.dirname(error_shape_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        lock_path = os.path.join(output_dir, ".error_output.lock")

        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                error_config = {
                    self.op_name: {
                        "shapes": [normalized_shape],
                        "shape_desc": self.shape_desc,
                    }
                }

                tmp_error_file = tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".yaml",
                    dir=output_dir,
                    delete=False,
                    encoding="utf-8",
                )
                try:
                    yaml.safe_dump(
                        error_config,
                        tmp_error_file,
                        sort_keys=False,
                    )
                    tmp_error_file.flush()
                    os.replace(tmp_error_file.name, error_shape_path)
                finally:
                    tmp_error_file.close()
                    if os.path.exists(tmp_error_file.name):
                        os.remove(tmp_error_file.name)
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

        sys.stderr.write(
            "[error_shape] "
            f"op={self.op_name} failed_shape={normalized_shape} "
            f"saved_to={error_shape_path}\n"
        )
        sys.stderr.flush()

    def _build_metric_from_input(self, input_item):
        metric = BenchmarkMetrics()
        args, kwargs = self.unpack_to_args_kwargs(input_item)
        metric.shape_detail = self.record_shapes(*args, **kwargs)
        if "latency_base" in self.to_bench_metrics:
            metric.latency_base = self.get_latency(self.torch_op, *args, **kwargs)
        if "latency" in self.to_bench_metrics:
            if self.gems_op:
                metric.latency = self.get_latency(self.gems_op, *args, **kwargs)
            else:
                if self.op_name == "zero_":
                    with flag_gems.use_gems():
                        metric.latency = self.get_latency(
                            self.torch_op, *args, **kwargs
                        )
                else:
                    with flag_gems.use_gems(exclude=["zero_"]):
                        metric.latency = self.get_latency(
                            self.torch_op, *args, **kwargs
                        )
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        if "gbps" in self.to_bench_metrics:
            metric.gbps_base = self.get_gbps(args, latency=metric.latency_base)
            metric.gbps = self.get_gbps(args, latency=metric.latency)
        if "tflops" in self.to_bench_metrics:
            metric.tflops = (
                self.get_tflops(self.torch_op, *args, **kwargs)
                / metric.latency
                / 1e12
                * 1e3
            )
        return metric

    def _run_inputs(self, input_items):
        metrics = []
        expected_shapes = self._iter_expected_shapes()
        for input_item in input_items:
            current_shape = next(expected_shapes, None)
            metric = BenchmarkMetrics()
            try:
                metric = self._build_metric_from_input(input_item)
            except Exception as e:
                metric.error_msg = str(e)
                self._record_failed_shape(current_shape)
                pytest.fail(str(e))
            finally:
                metrics.append(metric)
                gc.collect()
        return metrics

    def _get_shape_config_keys(self):
        keys = list(self.SHAPE_CONFIG_KEYS) + [self.op_name]
        keys.extend(cls.__name__ for cls in type(self).__mro__)
        return list(dict.fromkeys(key for key in keys if key))

    def _resolve_shape_config_key(self, yaml_config):
        for key in self._get_shape_config_keys():
            if key in yaml_config:
                return key

        preferred_key = self._get_shape_config_keys()[0]
        yaml_config[preferred_key] = {
            "shapes": [list(shape) for shape in self.shapes],
            "shape_desc": self.shape_desc,
        }
        return preferred_key

    def _split_shapes_evenly(self, num_buckets):
        indexed_shapes = list(enumerate(self.shapes))
        if not indexed_shapes:
            return []

        def estimate_shape_cost(shape):
            if self.op_name in {
                "mm",
                "addmm",
                "bmm",
                "baddbmm",
                "w8a8_block_fp8_matmul",
            }:
                normalized_shape = shape
                if len(shape) == 3:
                    normalized_shape = (1, *shape)
                if len(normalized_shape) == 4:
                    _, m, n, k = normalized_shape
                else:
                    normalized_shape = None

                if normalized_shape is None:
                    return 1

                if self.op_name in {"mm", "bmm", "w8a8_block_fp8_matmul"}:
                    return m * n * k * 2
                return m * n * (2 * k + 1)

            cost = 1
            for dim in shape:
                if isinstance(dim, bool):
                    continue
                if isinstance(dim, (int, float)):
                    cost *= max(1, int(dim))
            return cost

        sorted_items = sorted(
            indexed_shapes,
            key=lambda idx_shape: estimate_shape_cost(idx_shape[1]),
            reverse=True,
        )

        chunks = [[] for _ in range(num_buckets)]
        bucket_costs = [0] * num_buckets
        next_start_bucket = 0

        def select_bucket_by_min_cost():
            nonlocal next_start_bucket
            ordered = list(range(next_start_bucket, num_buckets)) + list(
                range(0, next_start_bucket)
            )
            target = min(ordered, key=lambda i: bucket_costs[i])
            next_start_bucket = (target + 1) % num_buckets
            return target

        heavy_prefix_count = min(len(sorted_items), max(num_buckets, num_buckets * 2))

        for idx_shape in sorted_items[:heavy_prefix_count]:
            target = select_bucket_by_min_cost()
            chunks[target].append(idx_shape)
            bucket_costs[target] += estimate_shape_cost(idx_shape[1])

        for idx_shape in sorted_items[heavy_prefix_count:]:
            target = select_bucket_by_min_cost()
            chunks[target].append(idx_shape)
            bucket_costs[target] += estimate_shape_cost(idx_shape[1])

        for bucket in chunks:
            bucket.sort(key=lambda x: x[0])
        return [bucket for bucket in chunks if bucket]

    def _run_parallel_worker_subprocess(self, node_id, shape_chunk, gpu_id, dtype_name):
        shape_chunk_only = [shape for _, shape in shape_chunk]

        with open(Config.shape_file, "r") as shape_file:
            yaml_config = yaml.safe_load(shape_file) or {}
        shape_key = self._resolve_shape_config_key(yaml_config)
        shape_entry = yaml_config.get(shape_key, {})
        shape_entry["shapes"] = [list(shape) for shape in shape_chunk_only]
        shape_entry["shape_desc"] = shape_entry.get("shape_desc", self.shape_desc)
        yaml_config[shape_key] = shape_entry

        tmp_shape_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, encoding="utf-8"
        )
        try:
            yaml.safe_dump(yaml_config, tmp_shape_file)
            tmp_shape_file.flush()
            tmp_shape_path = tmp_shape_file.name
        finally:
            tmp_shape_file.close()

        tmp_result_file = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".pkl", delete=False
        )
        try:
            tmp_result_file.flush()
            tmp_result_path = tmp_result_file.name
        finally:
            tmp_result_file.close()

        mode_arg = "--fg_mode" if flag_gems.vendor_name == "kunlunxin" else "--mode"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            node_id,
            mode_arg,
            Config.mode.value,
            "--level",
            Config.bench_level.value,
            "--warmup",
            str(Config.warm_up),
            "--iter",
            str(Config.repetition),
            "--shape_file",
            tmp_shape_path,
        ]
        if self.should_forward_parallel_dtype(dtype_name):
            cmd.extend(["--dtypes", dtype_name])
        if Config.user_desired_metrics:
            for metric in Config.user_desired_metrics:
                cmd.extend(["--metrics", metric])

        env = os.environ.copy()
        visible_devices_env = _parallel_visible_devices_env()
        if visible_devices_env is None:
            pytest.fail(
                f"--parallel is not supported on device type '{flag_gems.device}'."
            )
        env[visible_devices_env] = str(gpu_id)
        env[PARALLEL_WORKER_ENV] = "1"
        env[PARALLEL_RESULT_FILE_ENV] = tmp_result_path

        completed = subprocess.run(cmd, capture_output=True, text=True, env=env)

        try:
            result_payload = None
            if completed.returncode == 0 and os.path.getsize(tmp_result_path) > 0:
                with open(tmp_result_path, "rb") as result_file:
                    result_payload = pickle.load(result_file)
        finally:
            if os.path.exists(tmp_shape_path):
                os.remove(tmp_shape_path)
            if os.path.exists(tmp_result_path):
                os.remove(tmp_result_path)

        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "result_payload": result_payload,
        }

    def _run_parallel_dtype(self, dtype):
        required_gpus = int(Config.parallel)
        if required_gpus <= 0:
            return self._run_inputs(self.get_input_iter(dtype))
        if not _parallel_device_is_available():
            pytest.skip(f"--parallel N requires {_parallel_device_label()}.")
        available_gpus = _parallel_device_count()
        if available_gpus < required_gpus:
            pytest.skip(
                "--parallel requires at least "
                f"{required_gpus} {_parallel_device_label()} devices, "
                f"found {available_gpus}."
            )

        node_info = os.environ.get("PYTEST_CURRENT_TEST")
        if not node_info:
            pytest.fail("--parallel requires PYTEST_CURRENT_TEST context.")
        node_id = node_info.split(" (")[0]

        shape_chunks = self._split_shapes_evenly(required_gpus)
        if not shape_chunks:
            return []

        dtype_name = str(dtype).split(".")[-1]
        merged_metrics = []
        future_to_chunk = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(shape_chunks)) as ex:
            for gpu_id, shape_chunk in enumerate(shape_chunks):
                future = ex.submit(
                    self._run_parallel_worker_subprocess,
                    node_id=node_id,
                    shape_chunk=shape_chunk,
                    gpu_id=gpu_id,
                    dtype_name=dtype_name,
                )
                future_to_chunk[future] = shape_chunk

            for future in concurrent.futures.as_completed(future_to_chunk):
                shape_chunk = future_to_chunk[future]
                completed = future.result()
                if completed["returncode"] != 0:
                    if completed["stdout"]:
                        sys.stdout.write(completed["stdout"])
                    if completed["stderr"]:
                        sys.stderr.write(completed["stderr"])
                    pytest.fail(
                        "parallel benchmark worker failed with "
                        f"return code {completed['returncode']}"
                    )

                result_payload = completed["result_payload"]
                if result_payload is None:
                    pytest.fail("parallel benchmark worker did not produce result")

                chunk_metrics = result_payload.result
                cursor = 0
                for shape_index, shape in shape_chunk:
                    group_size = self.get_parallel_metric_group_size(shape)
                    next_cursor = cursor + group_size
                    if next_cursor > len(chunk_metrics):
                        pytest.fail(
                            "parallel benchmark worker returned incomplete metrics"
                        )
                    for metric_order, metric in enumerate(
                        chunk_metrics[cursor:next_cursor]
                    ):
                        merged_metrics.append((shape_index, metric_order, metric))
                    cursor = next_cursor

                if cursor != len(chunk_metrics):
                    pytest.fail("parallel benchmark worker returned unexpected metrics")

        merged_metrics.sort(key=lambda item: (item[0], item[1]))
        return [metric for _, _, metric in merged_metrics]

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
            if Config.parallel > 0 and not os.environ.get(PARALLEL_WORKER_ENV):
                metrics = self._run_parallel_dtype(dtype)
            else:
                metrics = self._run_inputs(self.get_input_iter(dtype))

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)
            emit_record_logger(result.to_json())
            if os.environ.get(PARALLEL_RESULT_FILE_ENV):
                with open(os.environ[PARALLEL_RESULT_FILE_ENV], "wb") as result_file:
                    pickle.dump(result, result_file)


class ParallelBlasBenchmark(ParallelBenchmarkMixin, blas_perf.BlasBenchmark):
    def get_parallel_metric_group_size(self, shape):
        if Config.bench_level == blas_perf.BenchLevel.COMPREHENSIVE:
            return 2
        return 1


class ParallelBaddbmmBenchmark(ParallelBenchmarkMixin, blas_perf.BaddbmmBenchmark):
    def get_parallel_metric_group_size(self, shape):
        if Config.bench_level == blas_perf.BenchLevel.COMPREHENSIVE:
            return 2
        return 1


class ParallelMvAndOuterBenchmark(
    ParallelBenchmarkMixin, blas_perf.MvAndOuterBenchmark
):
    pass


class ParallelAddmvBenchmark(ParallelBenchmarkMixin, blas_perf.AddmvBenchmark):
    pass


class ParallelVdotBenchmark(ParallelBenchmarkMixin, blas_perf.VdotBenchmark):
    pass


class ParallelAddrBenchmark(ParallelBenchmarkMixin, blas_perf.AddrBenchmark):
    pass


class ParallelW8A8BlockFP8MatmulBenchmark(
    ParallelBenchmarkMixin, blas_perf.W8A8BlockFP8MatmulBenchmark
):
    SHAPE_CONFIG_KEYS = ("mm", "BlasBenchmark")

    def set_more_shapes(self):
        if os.environ.get(PARALLEL_WORKER_ENV):
            return []
        return blas_perf.BlasBenchmark.set_more_shapes(self)

    def should_forward_parallel_dtype(self, dtype_name):
        if Config.user_desired_dtypes is None and dtype_name == "fp8":
            return False
        return True

    def set_shapes(self, shape_file_path=None):
        if not os.path.isfile(shape_file_path):
            raise FileNotFoundError(f"Shape file '{shape_file_path}' does not exist.")

        with open(shape_file_path, "r") as shape_file:
            yaml_config = yaml.safe_load(shape_file) or {}

        for shape_key in self._get_shape_config_keys():
            if shape_key in yaml_config:
                self.shapes = yaml_config[shape_key].get(
                    "shapes", blas_perf.Benchmark.DEFAULT_SHAPES
                )
                break
        else:
            self.shapes = blas_perf.Benchmark.DEFAULT_SHAPES

        self.shapes = [tuple(shape) for shape in self.shapes]
        if (
            Config.bench_level == blas_perf.BenchLevel.COMPREHENSIVE
            and not Config.query
            and not os.environ.get(PARALLEL_WORKER_ENV)
        ):
            additional_shapes = self.set_more_shapes()
            if additional_shapes:
                self.shapes = list(dict.fromkeys(self.shapes + additional_shapes))

        normalized_shapes = []
        for shape in self.shapes:
            if len(shape) == 4:
                _, m, n, k = shape
                normalized_shapes.append((m, n, k))
            elif len(shape) == 3:
                normalized_shapes.append(shape)
            else:
                raise ValueError(
                    "w8a8_block_fp8_matmul benchmark expects shapes in (M, N, K) "
                    "or (B, M, N, K) format."
                )

        self.shapes = normalized_shapes
        self.shape_desc = "M, N, K"


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, bench_cls",
    [
        pytest.param(
            "addmm",
            torch.addmm,
            blas_perf.addmm_input_fn,
            ParallelBlasBenchmark,
            marks=pytest.mark.addmm,
        ),
        pytest.param(
            "bmm",
            torch.bmm,
            blas_perf.bmm_input_fn,
            ParallelBlasBenchmark,
            marks=pytest.mark.bmm,
        ),
        pytest.param(
            "mm",
            torch.Tensor.mm,
            blas_perf.mm_input_fn,
            ParallelBlasBenchmark,
            marks=pytest.mark.mm,
        ),
        pytest.param(
            "baddbmm",
            torch.baddbmm,
            blas_perf.baddbmm_input_fn,
            ParallelBaddbmmBenchmark,
            marks=pytest.mark.baddbmm,
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn, bench_cls):
    if flag_gems.vendor_name == "mthreads" and op_name != "baddbmm":
        os.environ["MUSA_ENABLE_SQMMA"] = "1"

    bench = bench_cls(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=blas_perf.FLOAT_DTYPES,
    )
    bench.run()

    if flag_gems.vendor_name == "mthreads" and op_name != "baddbmm":
        del os.environ["MUSA_ENABLE_SQMMA"]


@pytest.mark.w8a8_block_fp8_matmul
def test_perf_w8a8_block_fp8_matmul():
    if not blas_perf.VLLM_W8A8_BLOCK_FP8_AVAILABLE:
        pytest.skip("w8a8_block_fp8_matmul benchmark requires vLLM baseline operator")
    if blas_perf.get_w8a8_block_fp8_dtype() is None:
        pytest.skip(
            "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
        )

    bench = ParallelW8A8BlockFP8MatmulBenchmark(
        op_name="w8a8_block_fp8_matmul",
        torch_op=blas_perf.vllm_w8a8_triton_block_scaled_mm,
        dtypes=["fp8"],
    )
    bench.set_gems(flag_gems.w8a8_block_fp8_matmul)
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "mv",
            torch.Tensor.mv,
            blas_perf.mv_input_fn,
            marks=pytest.mark.mv,
        ),
        pytest.param(
            "outer",
            torch.Tensor.outer,
            blas_perf.outer_input_fn,
            marks=pytest.mark.outer,
        ),
    ],
)
def test_mv_and_outer_benchmark(op_name, torch_op, input_fn):
    bench = ParallelMvAndOuterBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=blas_perf.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "addmv",
            torch.addmv,
            blas_perf.addmv_input_fn,
            marks=pytest.mark.addmv,
        ),
    ],
)
def test_addmv_benchmark(op_name, torch_op, input_fn):
    bench = ParallelAddmvBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=blas_perf.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.vdot
def test_vdot_benchmark():
    def vdot_input_fn(m, cur_dtype, device):
        inp1 = torch.randn([m], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        yield inp1, inp2

    bench = ParallelVdotBenchmark(
        input_fn=vdot_input_fn,
        op_name="vdot",
        torch_op=torch.Tensor.vdot,
        dtypes=blas_perf.COMPLEX_DTYPES + blas_perf.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.addr
def test_addr_benchmark():
    def addr_input_fn(m, n, cur_dtype, device):
        inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
        inp2 = torch.randn([m], dtype=cur_dtype, device=device)
        inp3 = torch.randn([n], dtype=cur_dtype, device=device)
        yield inp1, inp2, inp3, {"alpha": 0.5, "beta": 0.5}

    bench = ParallelAddrBenchmark(
        input_fn=addr_input_fn,
        op_name="addr",
        torch_op=torch.Tensor.addr,
        dtypes=blas_perf.FLOAT_DTYPES,
    )
    bench.run()
