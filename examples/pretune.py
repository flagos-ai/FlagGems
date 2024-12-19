import argparse

import torch

import flag_gems

device = flag_gems.device

DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
]

LLAMA_SHAPES = {
    "mm": [
        [1024, 4096],
        [128256, 4096],
        [14336, 4096],
        [4096, 14336],
        [4096, 4096],
        [6144, 4096],
        [28672, 4096],
    ],
}

QWEN_SHAPES = {
    "mm": [
        [3584, 3584],
        [18944, 3584],
        [3584, 18944],
        [152064, 3584],
        [37888, 3584],
    ],
    "addmm": [
        [3584, 3584],
        [512, 3584],
        [4608, 3584],
    ],
}


MODEL_SHAPES = {
    "llama": LLAMA_SHAPES,
    "qwen": QWEN_SHAPES,
}


def pretune_mm(max_tokens, shapes):
    for dtype in DTYPES:
        for M in range(1, max_tokens + 1):
            for N, K in shapes:
                tensor_a = torch.randn([M, K], dtype=dtype, device=device)
                tensor_b = torch.randn([K, N], dtype=dtype, device=device)
                flag_gems.mm(tensor_a, tensor_b)


def pretune_addmm(max_tokens, shapes):
    for dtype in DTYPES:
        for M in range(1, max_tokens + 1):
            for N, K in shapes:
                tensor_a = torch.randn([M, K], dtype=dtype, device=device)
                tensor_b = torch.randn([K, N], dtype=dtype, device=device)
                bias = torch.randn([M, N], dtype=dtype, device=device)
                flag_gems.addmm(bias, tensor_a, tensor_b)


OPERATORS = {
    "mm": pretune_mm,
    "addmm": pretune_addmm,
}


def args_parser():
    parser = argparse.ArgumentParser(
        description="pretune for gemm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="llama",
        help="model name",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=False,
        default=100,
        help="max tokens",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    model = MODEL_SHAPES.get(args.model)
    max_tokens = args.max_tokens
    if not model:
        exit(0)
    for op, func in OPERATORS.items():
        shapes = model.get(op)
        if not shapes:
            continue
        func(max_tokens, shapes)
