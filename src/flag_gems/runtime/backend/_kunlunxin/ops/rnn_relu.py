import logging

import torch

logger = logging.getLogger(__name__)
_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)


def rnn_relu(
    input,
    hx,
    params,
    has_biases,
    num_layers,
    dropout,
    train,
    bidirectional,
    batch_first,
):
    logger.debug("GEMS_KUNLUNXIN RNN_RELU")
    if num_layers != 1 or dropout != 0 or bidirectional:
        return torch.ops.aten.rnn_relu.input.redispatch(
            _CUDA_KEYSET,
            input,
            hx,
            params,
            has_biases,
            num_layers,
            dropout,
            train,
            bidirectional,
            batch_first,
        )

    x = input.transpose(0, 1) if batch_first else input
    w_ih, w_hh = params[:2]
    b_ih = params[2] if has_biases else None
    b_hh = params[3] if has_biases else None
    h = hx[0]
    outputs = []
    for step in x:
        input_part = torch.nn.functional.linear(step, w_ih, b_ih)
        hidden_part = torch.nn.functional.linear(h, w_hh, b_hh)
        h = torch.relu(input_part + hidden_part)
        outputs.append(h)

    output = torch.stack(outputs)
    if batch_first:
        output = output.transpose(0, 1)
    return output, h.unsqueeze(0)
