"""Autograd for the paddle entries that launch Triton kernels directly.

An entry registered over `paddle._C_ops.<op>` runs the Triton kernel from Python,
so Paddle's autograd never records it: inside `use_gems()` the output came back
with stop_gradient=True and the backward pass stopped there. `with_autograd`
wraps such an entry in a paddle.autograd.PyLayer whose backward is given by a
gradient rule below.

Calls where no input requires grad (or grad is disabled) skip the PyLayer and go
straight to the forward implementation, so inference and the benchmarks keep the
original dispatch cost.

A gradient rule is `rule(args, outs, grads, need) -> {param_name: grad}`:
  args   the forward arguments by parameter name, defaults applied
  outs   the forward outputs, as a tuple
  grads  the gradients of the outputs, as a tuple
  need   names of the tensor arguments that require grad
Gradients are built from paddle APIs, so inside `use_gems()` they run on the
FlagGems kernels as well.
"""

import functools
import inspect
import math

import paddle
from paddle.autograd import PyLayer

_Tensor = paddle.Tensor


def _requires_grad(args, kwargs):
    for a in args:
        if isinstance(a, _Tensor) and not a.stop_gradient:
            return True
    for a in kwargs.values():
        if isinstance(a, _Tensor) and not a.stop_gradient:
            return True
    return False


def with_autograd(fwd, rule):
    sig = inspect.signature(fwd)
    names = list(sig.parameters)

    def forward(ctx, *values):
        args = dict(zip(names, values))
        out = fwd(**args)
        outs = out if isinstance(out, tuple) else (out,)
        ctx.tensor_names = [n for n in names if isinstance(args[n], _Tensor)]
        ctx.need = {n for n in ctx.tensor_names if not args[n].stop_gradient}
        ctx.static = {n: v for n, v in args.items() if not isinstance(v, _Tensor)}
        ctx.save_for_backward(*(args[n] for n in ctx.tensor_names), *outs)
        nondiff = [o for o in outs if not paddle.is_floating_point(o)]
        if nondiff:
            ctx.mark_non_differentiable(*nondiff)
        return out

    def backward(ctx, *grads):
        saved = ctx.saved_tensor()
        k = len(ctx.tensor_names)
        args = dict(ctx.static)
        args.update(zip(ctx.tensor_names, saved[:k]))
        result = rule(args, tuple(saved[k:]), grads, ctx.need)
        res = []
        for n in ctx.tensor_names:
            g = result.get(n) if n in ctx.need else None
            if g is not None:
                x = args[n]
                if g.dtype != x.dtype:
                    g = paddle.cast(g, x.dtype)
                assert list(g.shape) == list(x.shape), (n, g.shape, x.shape)
            res.append(g)
        return tuple(res) if len(res) > 1 else res[0]

    layer = type(
        f"{fwd.__name__}_autograd",
        (PyLayer,),
        {
            "forward": staticmethod(forward),
            "backward": staticmethod(backward),
            "__module__": fwd.__module__,
        },
    )

    @functools.wraps(fwd)
    def entry(*args, **kwargs):
        if (
            not _requires_grad(args, kwargs)
            or not paddle.is_grad_enabled()
            or kwargs.get("out") is not None
        ):
            return fwd(*args, **kwargs)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return layer.apply(*(bound.arguments[n] for n in names))

    entry.pylayer = layer
    return entry


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _axes(axis, ndim):
    """paddle's reduction axis (None / [] = all, int, sequence) as sorted non-negative ints."""
    if axis is None:
        return list(range(ndim))
    if isinstance(axis, _Tensor):
        axis = axis.tolist()
    if isinstance(axis, int):
        axis = [axis]
    axis = [int(a) for a in axis]
    if not axis:
        return list(range(ndim))
    return sorted(a % ndim for a in axis) if ndim else []


def _expand_reduced(g, shape, axes, keepdim):
    """Broadcast a reduction's output (or its grad) back to the input shape."""
    if not keepdim:
        for a in axes:
            g = g.unsqueeze(a)
    return g.expand(shape)


def _sum_to(g, shape):
    """Reduce a broadcast result back to `shape` (the inverse of broadcasting)."""
    shape = list(shape)
    lead = g.ndim - len(shape)
    if lead > 0:
        g = g.sum(axis=list(range(lead)))
    dims = [i for i, s in enumerate(shape) if s == 1 and g.shape[i] != 1]
    if dims:
        g = g.sum(axis=dims, keepdim=True)
    return g.reshape(shape)


def _t(m):
    """Swap the last two dims, materialised so every kernel sees a dense layout."""
    perm = list(range(m.ndim))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    return m.transpose(perm).contiguous()


# ---------------------------------------------------------------------------
# gradient rules
# ---------------------------------------------------------------------------


def sum_grad(args, outs, grads, need):
    x = args["x"]
    axes = _axes(args["axis"], x.ndim)
    return {"x": _expand_reduced(grads[0], x.shape, axes, args["keepdim"])}


def mean_grad(args, outs, grads, need):
    x = args["x"]
    axes = _axes(args["axis"], x.ndim)
    count = math.prod(x.shape[a] for a in axes)
    g = _expand_reduced(grads[0], x.shape, axes, args["keepdim"])
    return {"x": g / count}


def _extremum_grad(args, outs, grads, axis):
    # Paddle's max / min / amax / amin grads split the gradient evenly among ties.
    x = args["x"]
    axes = _axes(axis, x.ndim)
    keepdim = args["keepdim"]
    mask = paddle.cast(x == _expand_reduced(outs[0], x.shape, axes, keepdim), grads[0].dtype)
    count = mask.sum(axis=axes, keepdim=True)
    g = _expand_reduced(grads[0], x.shape, axes, keepdim)
    return {"x": g * mask / count}


def amax_grad(args, outs, grads, need):
    axis = args["axis"] if args["axis"] is not None else args.get("dim")
    return _extremum_grad(args, outs, grads, axis)


def max_min_grad(args, outs, grads, need):
    return _extremum_grad(args, outs, grads, args["axis"])


def triu_grad(args, outs, grads, need):
    return {"x": paddle.triu(grads[0], args["diagonal"])}


def index_select_grad(args, outs, grads, need):
    inp = args["inp"]
    g = paddle.index_add(
        paddle.zeros(inp.shape, grads[0].dtype), args["index"], args["dim"], grads[0]
    )
    return {"inp": g}


def topk_grad(args, outs, grads, need):
    x = args["x"]
    _, indices = outs
    g = paddle.put_along_axis(
        paddle.zeros(x.shape, grads[0].dtype), indices, grads[0], args["axis"]
    )
    return {"x": g}


def bmm_grad(args, outs, grads, need):
    x, y, g = args["x"], args["y"], grads[0]
    res = {}
    if "x" in need:
        res["x"] = paddle.bmm(g, _t(y))
    if "y" in need:
        res["y"] = paddle.bmm(_t(x), g)
    return res


def mv_grad(args, outs, grads, need):
    inp, vec, g = args["inp"], args["vec"], grads[0]
    res = {}
    if "inp" in need:
        res["inp"] = g.unsqueeze(1) * vec.unsqueeze(0)
    if "vec" in need:
        # matmul rather than mv: paddle's native mv has no fp16/bf16 GPU kernel
        res["vec"] = paddle.matmul(g, inp)
    return res


def addmm_grad(args, outs, grads, need):
    bias, mat1, mat2, g = args["bias"], args["mat1"], args["mat2"], grads[0]
    beta, alpha = args["beta"], args["alpha"]
    res = {}
    if "bias" in need:
        res["bias"] = _sum_to(g * beta, bias.shape)
    if "mat1" in need:
        res["mat1"] = paddle.matmul(g, mat2, transpose_y=True) * alpha
    if "mat2" in need:
        res["mat2"] = paddle.matmul(mat1, g, transpose_x=True) * alpha
    return res


def dot_grad(args, outs, grads, need):
    x, y, g = args["x"], args["y"], grads[0]
    if x.ndim == 2:
        g = g.unsqueeze(-1)
    return {"x": g * y, "y": g * x}


def diag_grad(args, outs, grads, need):
    x, k, g = args["x"], args["offset"], grads[0]
    if x.ndim == 1:
        return {"x": paddle.diagonal(g, offset=k)}
    m, n = x.shape
    i = paddle.arange(g.shape[0], dtype="int64")
    flat = (i + max(-k, 0)) * n + (i + max(k, 0))
    return {"x": paddle.scatter(paddle.zeros([m * n], g.dtype), flat, g).reshape([m, n])}
