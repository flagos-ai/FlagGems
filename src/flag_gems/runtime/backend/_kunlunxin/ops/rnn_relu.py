import logging

import torch

from flag_gems.ops.linear import linear as _linear

logger = logging.getLogger(__name__)


def rnn_relu(
    input,
    hx=None,
    params=None,
    has_biases=True,
    num_layers=1,
    dropout=0.0,
    train=False,
    bidirectional=False,
    batch_first=False,
):
    """Single-layer unidirectional Elman RNN with ReLU activation (kunlunxin).

    XPU can not compile a fused Triton RNN kernel (a 2D weight-tile + reduction
    inside the sequential loop overflows uni_sram), so the recurrence is folded
    into a minimal sequence of primitive ops.

    Fast path: the input projection ``W_ih @ x + b_ih`` is pre-computed once as
    a single batched ``linear`` (M = seq*batch) and each step fuses the hidden
    recurrence with ``addmm`` (``pre[t] + h @ W_hh^T`` in one op). The fused
    accumulate (2 launches/step vs the CompositeImplicitAutograd decomposition's
    linear+add+relu, 3 launches) is the sole source of the folded speed-up.

    Safe path: on very small shapes the ``triton.autotune`` / ``libtuner``
    ``do_bench`` estimate for those big-M / addmm kernels rounds to 0 and raises
    ``ZeroDivisionError`` while cold-tuning (a benign infra edge, not a numerical
    issue; it is deterministic per (M,N,K)). Small per-step ``linear`` matmuls
    (M = batch) never hit that edge, so we transparently recompute with a
    per-step ``linear`` decomposition — same math, crash-free.
    """
    logger.debug("GEMS_KUNLUNXIN RNN_RELU")

    if params is None:
        raise ValueError("params must be provided")
    if hx is None:
        raise ValueError("hx must be provided to match torch.rnn_relu schema")
    if not (num_layers == 1 and not bidirectional and dropout == 0):
        raise NotImplementedError(
            "GEMS RNN_RELU only supports single-layer unidirectional without dropout"
        )

    w_ih = params[0]
    w_hh = params[1]
    if has_biases:
        b_ih = params[2]
        b_hh = params[3]
    else:
        b_ih = None
        b_hh = None

    x = input.transpose(0, 1).contiguous() if batch_first else input
    seq_len, batch_size, input_size = x.shape
    hidden_size = w_hh.shape[0]
    hx2d = hx.reshape(batch_size, hidden_size)

    try:
        # ---- fast path: batched input projection + fused addmm recurrence ----
        x2d = x.reshape(seq_len * batch_size, input_size)
        pre = _linear(x2d, w_ih, b_ih).reshape(seq_len, batch_size, hidden_size)
        if b_hh is not None:
            pre = pre + b_hh
        w_hh_t = w_hh.t().contiguous()
        h = hx2d
        outputs = []
        for t in range(seq_len):
            h = torch.relu(torch.addmm(pre[t], h, w_hh_t))
            outputs.append(h)
        output = torch.stack(outputs, 0)
    except ZeroDivisionError:
        # ---- safe path: per-step small linear (M=batch), crash-free ----
        h = hx2d
        outputs = []
        for t in range(seq_len):
            ih_t = _linear(x[t], w_ih, b_ih)
            hh_t = _linear(h, w_hh, b_hh)
            # accumulate in fp32 to avoid double-rounding on low-precision dtypes
            h = torch.relu(ih_t.to(torch.float32) + hh_t.to(torch.float32)).to(x.dtype)
            outputs.append(h)
        output = torch.stack(outputs, 0)

    if batch_first:
        output = output.transpose(0, 1).contiguous()

    return output, h.unsqueeze(0)
