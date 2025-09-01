import os
import threading
from contextlib import contextmanager

import pytest
import torch
import triton
from triton import language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry, libtuner


# not_raises is copied from https://gist.github.com/oisinmulvihill/45c14271fad7794a4a52516ecb784e69
@contextmanager
def not_raises(ExpectedException):
    try:
        yield

    except ExpectedException as error:
        raise AssertionError(f"Raised exception {error} when it should not!")

    except Exception as error:
        raise AssertionError(f"An unexpected exception {error} raised.")


def softmax_inner_decorator_cascade(x, dim, dtype=None):
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]  # pre_dim
    inp = x.contiguous()
    if dtype is None:
        dtype = x.dtype

    out = torch.empty_like(inp, dtype=dtype)

    with torch_device_fn.device(out.device):
        grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
        softmax_kernel_inner[grid](
            out,
            inp,
            M,
            N,
            DUMMY=60,
        )
    return out


def softmax_inner_pass_kernel_arg_via_kw(x, dim, dtype=None):
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]  # pre_dim
    inp = x.contiguous()
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
    softmax_kernel_inner[grid](
        out,
        inp,
        M,
        N=N,
        DUMMY=60,
    )
    return out


def softmax_inner_kernel_arg_apply_default(x, dim, dtype=None):
    assert dim >= -x.ndim and dim < x.ndim, "Invalid dim"
    dim = dim % x.ndim
    M = 1
    N = x.shape[dim]
    for i in range(dim):
        M *= x.shape[i]  # pre_dim
    inp = x.contiguous()
    if dtype is None:
        dtype = x.dtype
    out = torch.empty_like(inp, dtype=dtype)

    grid = lambda meta: (triton.cdiv(M, meta["TILE_M"]), 1, 1)
    softmax_kernel_inner[grid](
        out,
        inp,
        M,
        N,
    )
    return out


@libentry()
@triton.autotune(
    configs=[
        triton.Config({"TILE_N": 32}),
        triton.Config({"TILE_N": 64}),
        triton.Config({"TILE_N": 128}),
        triton.Config({"TILE_N": 256}),
        triton.Config({"TILE_N": 512}),
        triton.Config({"TILE_N": 1024}),
    ],
    key=["N"],
)
@triton.heuristics(
    values={
        "TILE_M": lambda args: 1024 // args["TILE_N"],
        "ONE_TILE_PER_CTA": lambda args: args["TILE_N"] >= args["N"],
    },
)
@triton.jit
def softmax_kernel_inner(
    output_ptr,
    input_ptr,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
    ONE_TILE_PER_CTA: tl.constexpr,
    DUMMY=42,
):
    _ = DUMMY
    pid_m = tl.program_id(0)
    m_offsets = pid_m * TILE_M + tl.arange(0, TILE_M)
    if ONE_TILE_PER_CTA:
        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        input_ptrs = input_ptr + offset
        mask = (m_offsets[:, None] < M) & (n_offsets < N)
        inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
        m = tl.max(inp, 1)
        e = tl.exp(inp - m[:, None])
        z = tl.sum(e, 1)
        out = e / z[:, None]
        output_ptrs = output_ptr + offset
        tl.store(output_ptrs, out, mask=mask)
    else:
        m = tl.full([TILE_M], value=float("-inf"), dtype=tl.float32)
        z = tl.full([TILE_M], value=0.0, dtype=tl.float32)

        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            m_new = tl.maximum(m, tl.max(inp, 1))
            alpha = m - m_new
            z = z * tl.exp(alpha) + tl.sum(tl.exp(inp - m_new[:, None]), axis=1)
            m = m_new
            n_offsets += TILE_N
            offset += TILE_N

        n_offsets = tl.arange(0, TILE_N)
        offset = m_offsets[:, None] * N + n_offsets
        for _ in range(0, N, TILE_N):
            mask = (m_offsets[:, None] < M) & (n_offsets < N)
            input_ptrs = input_ptr + offset
            inp = tl.load(input_ptrs, mask=mask, other=-float("inf"))
            o = tl.exp(inp - m[:, None]) / z[:, None]
            output_ptrs = output_ptr + offset
            tl.store(output_ptrs, o, mask=mask)
            n_offsets += TILE_N
            offset += TILE_N


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Test Files for Operators Not Pending Testing",
)
def test_decorator_cascade():
    # to test inner decorator can use arguments supplied by outer decorator
    # and grid function can use arguments supplied by all the decorator
    x = torch.randn((128, 128, 128), device=flag_gems.device)
    with not_raises(KeyError):
        _ = softmax_inner_decorator_cascade(x, dim=2)


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Test Files for Operators Not Pending Testing",
)
def test_pass_kernel_arg_via_kw():
    x = torch.randn((128, 128, 128), device=flag_gems.device)
    with not_raises(KeyError):
        _ = softmax_inner_pass_kernel_arg_via_kw(x, dim=2)


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Test Files for Operators Not Pending Testing",
)
def test_kernel_arg_apply_default():
    x = torch.randn((128, 128, 128), device=flag_gems.device)
    with not_raises(KeyError):
        _ = softmax_inner_kernel_arg_apply_default(x, dim=2)


class TaskThread(threading.Thread):
    def __init__(self, func, args):
        threading.Thread.__init__(self)
        self.func = func
        self.args = args

    def run(self):
        return self.func(*self.args)


def run_two_threads():
    devices = [0, 0]
    fs = []

    def task_fn(dev):
        x = torch.randn((128, 128, 128), device=dev)
        return softmax_inner_decorator_cascade(x, 1)

    for dev in devices:
        work = TaskThread(task_fn, (dev,))
        work.start()
        fs.append(work)

    for i in range(len(fs)):
        fs[i].join()


@pytest.mark.skipif(
    flag_gems.vendor_name == "kunlunxin",
    reason="Test Files for Operators Not Pending Testing",
)
def test_threadsafety():
    for i in range(100):
        with not_raises(Exception):
            run_two_threads()


def test_hash_generation():
    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
            triton.Config({"TILE_N": 128}),
            triton.Config({"TILE_N": 256}),
            triton.Config({"TILE_N": 512}),
            triton.Config({"TILE_N": 1024}),
        ],
        key=["x"],
    )
    @triton.jit
    def kernel_a(x, y):
        return x + y + 1

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
            triton.Config({"TILE_N": 128}),
            triton.Config({"TILE_N": 256}),
            triton.Config({"TILE_N": 512}),
            triton.Config({"TILE_N": 1024}),
        ],
        key=["x"],
    )
    @triton.jit
    def kernel_b(x, y):
        return x + y

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
            triton.Config({"TILE_N": 128}),
            triton.Config({"TILE_N": 256}),
            triton.Config({"TILE_N": 512}),
            triton.Config({"TILE_N": 1024}),
        ],
        key=["x"],
    )
    @triton.jit
    def kernel_a_copy(x, y):
        return x + y + 1

    assert kernel_a.kernel_hash != kernel_a_copy.kernel_hash
    assert kernel_a.kernel_hash != kernel_b.kernel_hash


def test_hash_changes_when_dependency_modified():
    @triton.jit
    def sub_func(x, y):
        return x + y

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
        ],
        key=["x"],
    )
    @triton.jit
    def main_kernel(x, y):
        return sub_func(x, y) * 2

    original_hash = main_kernel.kernel_hash

    @triton.jit
    def sub_func(x, y):  # noqa:F811
        return x + y + 1

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
        ],
        key=["x"],
    )
    @triton.jit
    def main_kernel(x, y):
        return sub_func(x, y) * 2

    modified_hash = main_kernel.kernel_hash

    assert original_hash != modified_hash, (
        f"Expected different hashes when sub-function changes, "
        f"but got same hash: {original_hash}"
    )
    original_hash = modified_hash

    @triton.jit
    def sub_func(x, y, z=0):  # noqa:F811
        return x + y + z

    @libtuner(
        configs=[
            triton.Config({"TILE_N": 32}),
            triton.Config({"TILE_N": 64}),
        ],
        key=["x"],
    )
    @triton.jit
    def main_kernel(x, y):
        return sub_func(x, y) * 2

    modified_hash = main_kernel.kernel_hash
    assert original_hash != modified_hash, (
        f"Expected different hashes when sub-function changes, "
        f"but got same hash: {original_hash}"
    )


@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason=" Cannot re-initialize MUSA in forked subprocess",
)
def test_libcache_vllm_signal_scenario():
    import multiprocessing
    import signal
    import sqlite3
    import time

    def child_process():
        import time

        import triton

        from flag_gems.utils.libentry import libcache

        cache = libcache["test_vllm_operator"]
        cache[(128, 256, "torch.float32")] = triton.Config(
            {"TILE_SIZE": 64}, num_warps=4
        )
        cache[(256, 512, "torch.float32")] = triton.Config(
            {"TILE_SIZE": 128}, num_warps=8
        )
        while True:
            time.sleep(0.1)

    from flag_gems.runtime.backend import vendor_module
    from flag_gems.utils.code_cache import config_cache_dir
    from flag_gems.utils.libentry import major_version, minor_version

    cache_file_name = (
        f"TunedConfig_{torch.cuda.get_device_name().replace(' ', '_')}_triton_{major_version}_{minor_version}.db"
        if vendor_module.vendor_info.vendor_name == "nvidia"
        else f"TunedConfig_{vendor_module.vendor_info.vendor_name}_triton_{major_version}_{minor_version}.db"
    )
    cache_path = config_cache_dir() / cache_file_name
    # Start child process
    process = multiprocessing.Process(target=child_process)
    process.start()
    time.sleep(1)
    os.kill(process.pid, signal.SIGINT)
    process.join(timeout=5)

    cache_saved = False
    if cache_path.exists():
        conn = sqlite3.connect(cache_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM test_vllm_operator")
            count = cursor.fetchone()[0]
            if count > 0:
                cache_saved = True
            cursor.execute("DELETE FROM test_vllm_operator")
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            pass

    assert cache_saved, f"Test documented current behavior: cache_saved={cache_saved}"

    if process.is_alive():
        os.kill(process.pid, signal.SIGKILL)
        process.join()


@pytest.mark.skipif(
    flag_gems.vendor_name == "mthreads",
    reason=" Cannot re-initialize MUSA in forked subprocess",
)
def test_libcache_concurrent_write_on_signal():
    """
    Tests that LibCache can handle concurrent writes from multiple processes
    when they are all terminated by a signal. This simulates a scenario where
    multiple vLLM workers are terminated at once.
    """
    import multiprocessing
    import signal
    import sqlite3
    import time

    NUM_PROCESSES = 10
    TABLE_NAME = "test_concurrent_signal_operator"

    def child_process_main(process_id):
        import time

        import triton

        from flag_gems.utils.libentry import libcache

        cache = libcache[TABLE_NAME]
        cache[(f"key_from_proc_{process_id}",)] = triton.Config(
            {}, num_warps=process_id + 1
        )
        while True:
            time.sleep(0.1)

    from flag_gems.runtime.backend import vendor_module
    from flag_gems.utils.code_cache import config_cache_dir
    from flag_gems.utils.libentry import major_version, minor_version

    cache_file_name = (
        f"TunedConfig_{torch.cuda.get_device_name().replace(' ', '_')}_triton_{major_version}_{minor_version}.db"
        if vendor_module.vendor_info.vendor_name == "nvidia"
        else f"TunedConfig_{vendor_module.vendor_info.vendor_name}_triton_{major_version}_{minor_version}.db"
    )
    cache_path = config_cache_dir() / cache_file_name
    if cache_path.exists():
        try:
            with sqlite3.connect(cache_path, timeout=10.0) as conn:
                conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
        except sqlite3.Error:
            pass

    ctx = multiprocessing.get_context("fork")
    processes = [
        ctx.Process(target=child_process_main, args=(i,)) for i in range(NUM_PROCESSES)
    ]
    for p in processes:
        p.start()

    try:
        time.sleep(2)
        for p in processes:
            os.kill(p.pid, signal.SIGTERM)

        for p in processes:
            p.join(timeout=10)

        total_entries = 0
        if cache_path.exists():
            with sqlite3.connect(cache_path) as conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
                    total_entries = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    pass  # Table might not exist if saving failed

        assert total_entries == NUM_PROCESSES, (
            f"Expected {NUM_PROCESSES} entries from concurrent processes, "
            f"but found {total_entries}."
        )

    finally:
        for p in processes:
            if p.is_alive():
                p.kill()
        if cache_path.exists():
            try:
                cache_path.unlink()
            except sqlite3.Error:
                pass
