import concurrent.futures
import fcntl
import gc
import math
import os
import pickle
import subprocess
import sys
import tempfile
from typing import Generator

import pytest
import torch
import yaml

import flag_gems
from benchmark.attri_util import (
    COMPLEX_DTYPES,
    DEFAULT_METRICS,
    FLOAT_DTYPES,
    BenchLevel,
    BenchmarkMetrics,
    BenchmarkResult,
    OperationAttribute,
    model_shapes,
)
from benchmark.conftest import Config, emit_record_logger
from benchmark.performance_utils import Benchmark, GenericBenchmark2DOnly

try:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        w8a8_triton_block_scaled_mm as vllm_w8a8_triton_block_scaled_mm,
    )

    VLLM_W8A8_BLOCK_FP8_AVAILABLE = True
except Exception:
    vllm_w8a8_triton_block_scaled_mm = None
    VLLM_W8A8_BLOCK_FP8_AVAILABLE = False

try:
    from flag_gems.runtime.backend._mthreads.sparse_attention import (
        sparse_attn_triton as sparse_attention_mthreads_baseline,
    )

    SPARSE_ATTENTION_MTHREADS_BASELINE_AVAILABLE = True
except Exception:
    sparse_attention_mthreads_baseline = None
    SPARSE_ATTENTION_MTHREADS_BASELINE_AVAILABLE = False

try:
    from vllm.utils.deep_gemm import (
        fp8_gemm_nt,
        is_deep_gemm_supported,
        transform_sf_into_required_layout,
    )

    DEEPGEMM_AVAILABLE = is_deep_gemm_supported()
except Exception:
    fp8_gemm_nt = None
    transform_sf_into_required_layout = None
    DEEPGEMM_AVAILABLE = False

PARALLEL_WORKER_ENV = "FLAGGEMS_BENCH_PARALLEL_WORKER"
PARALLEL_RESULT_FILE_ENV = "FLAGGEMS_BENCH_RESULT_FILE"
torch_device_object = flag_gems.runtime.backend.gen_torch_device_object()
DEEPGEMM_N_MULTIPLE = 64
DEEPGEMM_K_MULTIPLE = 128


# ============================================================================
# Blas benchmark classes (from test_blas_perf.py)
# ============================================================================


class BlasBenchmark(Benchmark):
    """
    benchmark for blas
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def get_input_iter(self, cur_dtype) -> Generator:
        for b, m, n, k in self.shapes:
            yield from self.input_fn(b, m, n, k, cur_dtype, self.device, False)

        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            for b, m, n, k in self.shapes:
                yield from self.input_fn(b, m, n, k, cur_dtype, self.device, True)

    def set_more_shapes(self):
        large_k_shapes = [
            (8, 1848, 1536, 151936),
            (8, 1848, 1536, 128256),
            (8, 1848, 1536, 152064),
            (8, 4096, 1, 152064),
        ]

        model_shaps = model_shapes()
        return large_k_shapes + model_shaps

    def get_tflops(self, op, *args, **kwargs):
        total_flops = 0
        # shape(m,k)(k,n)
        # total_flops mxnx2k
        if self.op_name == "mm":
            total_flops = args[0].shape[0] * args[0].shape[1] * args[1].shape[1] * 2
        # shape(m,n)(n,p)
        # total_flops mxpx(2n+1)
        elif self.op_name == "addmm":
            total_flops = (
                args[0].shape[0] * args[1].shape[1] * (args[1].shape[0] * 2 + 1)
            )
        # total_flops bxnxpx2m
        elif self.op_name == "bmm":
            total_flops = (
                args[0].shape[0]
                * args[0].shape[1]
                * args[1].shape[2]
                * 2
                * args[0].shape[2]
            )
        return total_flops


class BaddbmmBenchmark(BlasBenchmark):
    """
    benchmark for Baddbmm
    """

    def set_more_shapes(self):
        model_shapes_list = model_shapes()

        skip_shapes = [
            (4, 8192, 128256, 4096),
            (4, 8192, 152064, 3584),
        ]

        filtered = []
        for shape in model_shapes_list:
            if shape not in skip_shapes:
                filtered.append(shape)

        return filtered

    def get_tflops(self, op, *args, **kwargs):
        # shape(b,m,k)(b,k,n)
        # total_flops = b * m * n * (2 * k + 1)
        total_flops = (
            args[1].shape[0]
            * args[1].shape[1]
            * args[2].shape[2]
            * (args[1].shape[2] * 2 + 1)
        )
        return total_flops


class GroupmmBenchmark(BlasBenchmark):
    """
    benchmark for Groupmm
    """

    def get_input_iter(self, cur_dtype) -> Generator:
        for groups, n, k in self.shapes:
            yield from self.input_fn(groups, n, k, cur_dtype, self.device)

    def set_more_shapes(self):
        return None

    def get_tflops(self, op, *args, **kwargs):
        groups, N, K = args[1].shape
        size_per_group = torch.diff(
            args[2], prepend=torch.zeros(1, device="cuda", dtype=torch.int32)
        )
        total_flops = 0
        for i in range(groups):
            total_flops += size_per_group[i].item() * N * K * 2
        return total_flops


class MvAndOuterBenchmark(GenericBenchmark2DOnly):
    """
    Benchmark for MV and Outer operations
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


class AddmvBenchmark(GenericBenchmark2DOnly):
    """
    Benchmark for addmv
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for m, n in self.shapes:
            yield from self.input_fn(m, n, cur_dtype, self.device)


class VdotBenchmark(BlasBenchmark):
    """
    benchmark for vdot
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            m = shape[0]
            yield from self.input_fn(m, cur_dtype, self.device)


class AddrBenchmark(BlasBenchmark):
    """
    benchmark for addr
    """

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            m, n = shape[0], shape[1]
            yield from self.input_fn(m, n, cur_dtype, self.device)


# ============================================================================
# Input functions (from test_blas_perf.py)
# ============================================================================


def addmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    bias = torch.randn([m, n], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield bias, inp1, inp2,


def bmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.transpose(1, 2)
    else:
        inp2 = torch.randn([b, k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


def baddbmm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=cur_dtype, device=device, requires_grad=True)

    if b_column_major:
        inp2 = torch.randn(
            [b, n, k], dtype=cur_dtype, device=device, requires_grad=True
        )
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn(
            [b, k, n], dtype=cur_dtype, device=device, requires_grad=True
        )

    bias = torch.randn([b, m, n], dtype=cur_dtype, device=device, requires_grad=True)

    yield bias, inp1, inp2


def mm_input_fn(b, m, n, k, cur_dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=cur_dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=cur_dtype, device=device)
        yield inp1, inp2.t()
    else:
        inp2 = torch.randn([k, n], dtype=cur_dtype, device=device)
        yield inp1, inp2


def group_mm_input_fn(groups, N, K, cur_dtype, device):
    assert cur_dtype == torch.bfloat16
    import random

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


def mv_input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m, n], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def outer_input_fn(m, n, cur_dtype, device):
    inp1 = torch.randn([m], dtype=cur_dtype, device=device)
    inp2 = torch.randn([n], dtype=cur_dtype, device=device)
    yield inp1, inp2


def addmv_input_fn(m, n, cur_dtype, device):
    mat = torch.randn([m, n], dtype=cur_dtype, device=device)
    vec = torch.randn([n], dtype=cur_dtype, device=device)
    bias = torch.randn([m], dtype=cur_dtype, device=device)
    # torch.addmv(bias, mat, vec)
    yield bias, mat, vec


# ============================================================================
# FP8 utilities (from test_blas_perf.py)
# ============================================================================

W8A8_BLOCK_FP8_MNK_SHAPES = [
    (64, 128, 128),
    (128, 256, 512),
    (1, 4096, 7168),
    (16, 4096, 7168),
    (64, 4096, 7168),
    (83, 7748, 3884),
    (84, 7168, 3884),
]
W8A8_BLOCK_FP8_BLOCK_SIZE = [128, 128]


def get_w8a8_block_fp8_dtype():
    if flag_gems.device != "cuda" or not torch.cuda.is_available():
        return None

    major, _ = torch.cuda.get_device_capability()
    if major > 8 and hasattr(torch, "float8_e4m3fn"):
        return torch.float8_e4m3fn
    if major == 8 and hasattr(torch, "float8_e5m2"):
        return torch.float8_e5m2
    return None


def rand_fp8_tensor(shape, device, dtype):
    finfo = torch.finfo(dtype)
    return (
        torch.randn(shape, device=device, dtype=torch.float32)
        .clamp(min=finfo.min, max=finfo.max)
        .to(dtype)
    )


class W8A8BlockFP8MatmulBenchmark(Benchmark):
    """
    Benchmark for w8a8_block_fp8_matmul.
    """

    DEFAULT_METRICS = DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, *args, block_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_size = (
            W8A8_BLOCK_FP8_BLOCK_SIZE[:] if block_size is None else list(block_size)
        )
        self.shape_desc = "M, N, K"

    def set_shapes(self, shape_file_path=None):
        self.shapes = W8A8_BLOCK_FP8_MNK_SHAPES[:]
        self.shape_desc = "M, N, K"

    def get_input_iter(self, cur_dtype) -> Generator:
        fp8_dtype = get_w8a8_block_fp8_dtype()
        if fp8_dtype is None:
            raise RuntimeError(
                "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
            )

        block_n, block_k = self.block_size
        for m, n, k in self.shapes:
            num_k_groups = (k + block_k - 1) // block_k
            num_n_groups = (n + block_n - 1) // block_n

            A = rand_fp8_tensor((m, k), self.device, fp8_dtype).contiguous()
            B = rand_fp8_tensor((n, k), self.device, fp8_dtype).contiguous()
            As = (
                0.01
                * torch.rand((m, num_k_groups), dtype=torch.float32, device=self.device)
                + 0.005
            ).contiguous()
            Bs = (
                0.01
                * torch.rand(
                    (num_n_groups, num_k_groups),
                    dtype=torch.float32,
                    device=self.device,
                )
                + 0.005
            ).contiguous()

            yield A, B, As, Bs, self.block_size[:], torch.float16

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        m, k = A.shape
        n = B.shape[0]
        return 2 * m * n * k


# ============================================================================
# Parallel benchmark infrastructure
# ============================================================================


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
            if self.op_name == "sparse_attention":
                if len(shape) != 6:
                    return 1
                batch, seq_len, _, topk, heads, dim = shape
                block = 64
                topk_aligned = ((max(1, int(topk)) + block - 1) // block) * block
                heads_padded = max(16, 1 << (max(1, int(heads)) - 1).bit_length())
                # The sparse attention kernel processes top-k indices in BLOCK=64
                # chunks and pads H to at least 16 / next power of two.
                return (
                    max(1, int(batch))
                    * max(1, int(seq_len))
                    * topk_aligned
                    * heads_padded
                    * max(1, int(dim))
                    * 4
                )

            if self.op_name in {
                "mm",
                "addmm",
                "bmm",
                "baddbmm",
                "w8a8_block_fp8_matmul",
                "w8a8_block_fp8_matmul_deepgemm",
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

                if self.op_name in {
                    "mm",
                    "bmm",
                    "w8a8_block_fp8_matmul",
                }:
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


class ParallelBlasBenchmark(ParallelBenchmarkMixin, BlasBenchmark):
    def get_parallel_metric_group_size(self, shape):
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            return 2
        return 1


class ParallelBaddbmmBenchmark(ParallelBenchmarkMixin, BaddbmmBenchmark):
    def get_parallel_metric_group_size(self, shape):
        if Config.bench_level == BenchLevel.COMPREHENSIVE:
            return 2
        return 1


class ParallelMvAndOuterBenchmark(ParallelBenchmarkMixin, MvAndOuterBenchmark):
    pass


class ParallelAddmvBenchmark(ParallelBenchmarkMixin, AddmvBenchmark):
    pass


class ParallelVdotBenchmark(ParallelBenchmarkMixin, VdotBenchmark):
    pass


class ParallelAddrBenchmark(ParallelBenchmarkMixin, AddrBenchmark):
    pass


class ParallelW8A8BlockFP8MatmulBenchmark(
    ParallelBenchmarkMixin, W8A8BlockFP8MatmulBenchmark
):
    SHAPE_CONFIG_KEYS = ("w8a8_block_fp8_matmul", "BlasBenchmark")

    def set_more_shapes(self):
        if os.environ.get(PARALLEL_WORKER_ENV):
            return []
        return BlasBenchmark.set_more_shapes(self)

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
                    "shapes", Benchmark.DEFAULT_SHAPES
                )
                break
        else:
            self.shapes = Benchmark.DEFAULT_SHAPES

        self.shapes = [tuple(shape) for shape in self.shapes]
        if (
            Config.bench_level == BenchLevel.COMPREHENSIVE
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


def _deepgemm_block_scaled_mm(A, B, As_dg, Bs_dg, output):
    fp8_gemm_nt((A, As_dg), (B, Bs_dg), output)
    return output


class ParallelW8A8BlockFP8DeepGemmBenchmark(ParallelW8A8BlockFP8MatmulBenchmark):
    def __init__(self, *args, output_dtype=torch.bfloat16, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dtype = output_dtype

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        self.shapes = [
            (m, n, k)
            for m, n, k in self.shapes
            if n % DEEPGEMM_N_MULTIPLE == 0 and k % DEEPGEMM_K_MULTIPLE == 0
        ]

    def get_input_iter(self, cur_dtype):
        fp8_dtype = get_w8a8_block_fp8_dtype()
        if fp8_dtype is None:
            raise RuntimeError(
                "DeepGEMM benchmark requires CUDA device with FP8 support"
            )

        block_n, block_k = self.block_size
        recipe = (1, 128, 128)

        for m, n, k in self.shapes:
            num_k_groups = (k + block_k - 1) // block_k
            num_n_groups = (n + block_n - 1) // block_n

            A = rand_fp8_tensor((m, k), self.device, fp8_dtype).contiguous()
            B = rand_fp8_tensor((n, k), self.device, fp8_dtype).contiguous()
            As = (
                0.01
                * torch.rand((m, num_k_groups), dtype=torch.float32, device=self.device)
                + 0.005
            ).contiguous()
            Bs = (
                0.01
                * torch.rand(
                    (num_n_groups, num_k_groups),
                    dtype=torch.float32,
                    device=self.device,
                )
                + 0.005
            ).contiguous()

            As_dg = transform_sf_into_required_layout(
                sf=As.unsqueeze(0),
                mn=m,
                k=k,
                recipe=recipe,
                num_groups=1,
                is_sfa=True,
            ).squeeze(0)
            Bs_dg = transform_sf_into_required_layout(
                sf=Bs.unsqueeze(0),
                mn=n,
                k=k,
                recipe=recipe,
                num_groups=1,
                is_sfa=False,
            ).squeeze(0)
            output = torch.empty((m, n), dtype=self.output_dtype, device=self.device)

            yield (
                A,
                B,
                As_dg,
                Bs_dg,
                output,
            ), (
                A,
                B,
                As,
                Bs,
                self.block_size[:],
                self.output_dtype,
            )

    def _build_metric_from_input(self, input_item):
        dg_input, gems_input = input_item
        metric = BenchmarkMetrics()

        dg_args, dg_kwargs = self.unpack_to_args_kwargs(dg_input)
        gems_args, gems_kwargs = self.unpack_to_args_kwargs(gems_input)
        metric.shape_detail = self.record_shapes(*gems_args, **gems_kwargs)

        if "latency_base" in self.to_bench_metrics:
            metric.latency_base = self.get_latency(self.torch_op, *dg_args, **dg_kwargs)
        if "latency" in self.to_bench_metrics:
            metric.latency = self.get_latency(self.gems_op, *gems_args, **gems_kwargs)
        if "speedup" in self.to_bench_metrics:
            metric.speedup = metric.latency_base / metric.latency
        if "tflops" in self.to_bench_metrics:
            metric.tflops = (
                self.get_tflops(self.torch_op, *dg_args, **dg_kwargs)
                / metric.latency
                / 1e12
                * 1e3
            )
        return metric

    def get_tflops(self, op, *args, **kwargs):
        A, B = args[0], args[1]
        m, k = A.shape
        n = B.shape[0]
        return 2 * m * n * k


#
# sparse_attention shape layout:
# (batch, seq_len, kv_len, topk, heads, dim)
#
SPARSE_ATTENTION_SHAPES = [
    (16, 1, 136, 136, 8, 512),
    (16, 1, 392, 385, 8, 512),
    (16, 1, 392, 386, 8, 512),
    (16, 1, 392, 387, 8, 512),
    (32, 1, 392, 388, 8, 512),
    (32, 1, 392, 389, 8, 512),
    (32, 1, 392, 390, 8, 512),
    (32, 1, 392, 391, 8, 512),
    (64, 1, 136, 136, 8, 512),
    (64, 1, 392, 385, 8, 512),
    (64, 1, 392, 388, 8, 512),
    (64, 1, 392, 389, 8, 512),
]


def torch_sparse_attention(q, kv, attn_sink, topk_idxs, softmax_scale):
    batch, seq_len, heads, dim = q.shape
    topk = topk_idxs.shape[-1]

    kv_expanded = kv[:, None, :, :].expand(batch, seq_len, -1, dim)
    idx_expanded = topk_idxs[:, :, :, None].expand(batch, seq_len, topk, dim).long()
    gathered_kv = torch.gather(kv_expanded, 2, idx_expanded)

    scores = (
        torch.einsum("bmhd,bmtd->bmht", q.float(), gathered_kv.float()) * softmax_scale
    )
    sink = attn_sink[None, None, :, None].expand(batch, seq_len, heads, 1)
    attn = torch.softmax(torch.cat([scores, sink], dim=-1), dim=-1)

    out = torch.einsum("bmht,bmtd->bmhd", attn[:, :, :, :-1], gathered_kv.float())
    return out.to(q.dtype)


class ParallelSparseAttentionBenchmark(ParallelBenchmarkMixin, Benchmark):
    SHAPE_CONFIG_KEYS = ("sparse_attention",)
    DEFAULT_METRICS = BlasBenchmark.DEFAULT_METRICS[:]
    DEFAULT_DTYPES = [torch.bfloat16]
    DEFAULT_SHAPES = SPARSE_ATTENTION_SHAPES[:]
    DEFAULT_SHAPE_DESC = "B, M, KV_LEN, TOPK, H, D"
    DEFAULT_SHAPE_FILES = os.path.join(os.path.dirname(__file__), "core_shapes.yaml")

    def set_more_shapes(self):
        return []

    def set_shapes(self, shape_file_path=None):
        shape_file_path = shape_file_path or self.DEFAULT_SHAPE_FILES
        self.shapes = self.DEFAULT_SHAPES[:]
        self.shape_desc = self.DEFAULT_SHAPE_DESC

        if not os.path.isfile(shape_file_path):
            raise FileNotFoundError(f"Shape file '{shape_file_path}' does not exist.")

        with open(shape_file_path, "r") as shape_file:
            yaml_config = yaml.safe_load(shape_file) or {}

        for shape_key in self.SHAPE_CONFIG_KEYS + (self.op_name,):
            if shape_key in yaml_config:
                self.shapes = yaml_config[shape_key].get("shapes", self.DEFAULT_SHAPES)
                self.shape_desc = yaml_config[shape_key].get(
                    "shape_desc", self.DEFAULT_SHAPE_DESC
                )
                break

        self.shapes = [tuple(shape) for shape in self.shapes]
        for shape in self.shapes:
            if len(shape) != 6:
                raise ValueError(
                    "sparse_attention benchmark expects shapes in "
                    "(batch, seq_len, kv_len, topk, heads, dim) format."
                )

    def get_input_iter(self, cur_dtype):
        for seed, (batch, seq_len, kv_len, topk, heads, dim) in enumerate(self.shapes):
            torch.manual_seed(2026 + seed)
            q = torch.randn(
                (batch, seq_len, heads, dim),
                dtype=cur_dtype,
                device=self.device,
            )
            kv = torch.randn(
                (batch, kv_len, dim),
                dtype=cur_dtype,
                device=self.device,
            )
            attn_sink = torch.zeros((heads,), dtype=torch.float32, device=self.device)
            topk_idxs = torch.randint(
                0,
                kv_len,
                (batch, seq_len, topk),
                dtype=torch.int32,
                device=self.device,
            )
            yield q, kv, attn_sink, topk_idxs, 1.0 / math.sqrt(dim)

    def get_tflops(self, op, *args, **kwargs):
        q, _, _, topk_idxs = args[:4]
        batch, seq_len, heads, dim = q.shape
        topk = topk_idxs.shape[-1]
        return batch * seq_len * topk * 4 * heads * dim


# ============================================================================
# Test functions
# ============================================================================


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn, bench_cls",
    [
        pytest.param(
            "addmm",
            torch.addmm,
            addmm_input_fn,
            ParallelBlasBenchmark,
            marks=pytest.mark.addmm,
        ),
        pytest.param(
            "bmm",
            torch.bmm,
            bmm_input_fn,
            ParallelBlasBenchmark,
            marks=pytest.mark.bmm,
        ),
        pytest.param(
            "mm",
            torch.Tensor.mm,
            mm_input_fn,
            ParallelBlasBenchmark,
            marks=pytest.mark.mm,
        ),
        pytest.param(
            "baddbmm",
            torch.baddbmm,
            baddbmm_input_fn,
            ParallelBaddbmmBenchmark,
            marks=pytest.mark.baddbmm,
        ),
    ],
)
def test_blas_benchmark(op_name, torch_op, input_fn, bench_cls):
    bench = bench_cls(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.w8a8_block_fp8_matmul
def test_perf_w8a8_block_fp8_matmul():
    if not VLLM_W8A8_BLOCK_FP8_AVAILABLE:
        pytest.skip("w8a8_block_fp8_matmul benchmark requires vLLM baseline operator")
    if get_w8a8_block_fp8_dtype() is None:
        pytest.skip(
            "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
        )

    bench = ParallelW8A8BlockFP8MatmulBenchmark(
        op_name="w8a8_block_fp8_matmul",
        torch_op=vllm_w8a8_triton_block_scaled_mm,
        dtypes=["fp8"],
    )
    bench.set_gems(flag_gems.w8a8_block_fp8_matmul)
    bench.run()


@pytest.mark.w8a8_block_fp8_matmul_deepgemm
def test_perf_w8a8_block_fp8_matmul_deepgemm():
    if not DEEPGEMM_AVAILABLE:
        pytest.skip("DeepGEMM is not available on this platform")
    if get_w8a8_block_fp8_dtype() is None:
        pytest.skip(
            "w8a8_block_fp8_matmul benchmark requires CUDA device with FP8 support"
        )

    bench = ParallelW8A8BlockFP8DeepGemmBenchmark(
        op_name="w8a8_block_fp8_matmul_deepgemm",
        torch_op=_deepgemm_block_scaled_mm,
        dtypes=["fp8"],
        output_dtype=torch.bfloat16,
    )
    bench.set_gems(flag_gems.w8a8_block_fp8_matmul)
    bench.run()


@pytest.mark.sparse_attention
def test_perf_sparse_attention():
    if not SPARSE_ATTENTION_MTHREADS_BASELINE_AVAILABLE:
        pytest.skip(
            "sparse_attention benchmark requires mthreads sparse_attention baseline"
        )

    bench = ParallelSparseAttentionBenchmark(
        op_name="sparse_attention",
        torch_op=sparse_attention_mthreads_baseline,
    )
    bench.set_gems(flag_gems.sparse_attn_triton)
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "mv",
            torch.Tensor.mv,
            mv_input_fn,
            marks=pytest.mark.mv,
        ),
        pytest.param(
            "outer",
            torch.Tensor.outer,
            outer_input_fn,
            marks=pytest.mark.outer,
        ),
    ],
)
def test_mv_and_outer_benchmark(op_name, torch_op, input_fn):
    bench = ParallelMvAndOuterBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.parametrize(
    "op_name, torch_op, input_fn",
    [
        pytest.param(
            "addmv",
            torch.addmv,
            addmv_input_fn,
            marks=pytest.mark.addmv,
        ),
    ],
)
def test_addmv_benchmark(op_name, torch_op, input_fn):
    bench = ParallelAddmvBenchmark(
        input_fn=input_fn,
        op_name=op_name,
        torch_op=torch_op,
        dtypes=FLOAT_DTYPES,
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
        dtypes=COMPLEX_DTYPES + FLOAT_DTYPES,
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
        dtypes=FLOAT_DTYPES,
    )
    bench.run()
