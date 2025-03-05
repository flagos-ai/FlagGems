import threading
from contextlib import contextmanager

import pytest
import torch
import triton
from triton import language as tl

import flag_gems
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


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
