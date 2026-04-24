import logging

import triton
import triton.language as tl

from flag_gems.utils import pointwise_dynamic


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_forward(x):
    # Use standard tl.exp rather than potentially undefined exp2
    # sigmoid(x) = 1 / (1 + e^(-x))
    return 1 / (1 + tl.exp(-x.to(tl.float32)))


@pointwise_dynamic(promotion_methods=[(0, "INT_TO_FLOAT")])
@triton.jit
def sigmoid_backward_kernel(dy, y):
    y_f32 = y.to(tl.float32)
    dy_f32 = dy.to(tl.float32)
    return dy_f32 * (1.0 - y_f32) * y_f32


def sigmoid(self):
    logging.debug("GEMS SIGMOID FORWARD")
    output = sigmoid_forward(self)
    return output


def sigmoid_backward(grad_output, output):
    logging.debug("GEMS SIGMOID BACKWARD")
    grad_input = sigmoid_backward_kernel(grad_output, output)
    return grad_input


def sigmoid_(A):
    logging.debug("GEMS SIGMOID_ FORWARD")
    out = sigmoid_forward(A, out0=A)
    return out
