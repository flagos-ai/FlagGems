import argparse

import torch

import flag_gems

device = flag_gems.device

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

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

QWEN3_06B_SHAPES = {
    "mm": [
        [4096, 1024],
        [1024, 2048],
        [6144, 1024],
        [1024, 3072],
    ],
    "mm_logits": [
        [151936, 1024],
    ],
    "index": [
        1024,
    ],
}


QWEN3_8B_SHAPES = {
    "mm": [
        [6144, 4096],
        [4096, 4096],
        [24576, 4096],
        [4096, 12288],
    ],
    "mm_logits": [
        [151936, 4096],
    ],
    "index": [
        4096,
    ],
}


QWEN3_30B_A3B_SHAPES = {
    "mm": [
        [5120, 2048],
        [2048, 4096],
        [128, 2048],
    ],
    "mm_logits": [
        [151936, 2048],
    ],
    "index": [
        2048,
    ],
}


QWEN25_7B_INSTRUCT_SHAPES = {
    "mm": [
        [3584, 3584],
        [37888, 3584],
        [3584, 18944],
    ],
    "mm_logits": [
        [152064, 3584],
    ],
    "addmm": [
        [4608, 3584],
    ],
    "index": [
        3584,
    ],
}


MODEL_SHAPES = {
    "llama": LLAMA_SHAPES,
    "qwen": QWEN_SHAPES,
    "qwen3_0.6b": QWEN3_06B_SHAPES,
    "qwen3_8b": QWEN3_8B_SHAPES,
    "qwen3_30b_a3b": QWEN3_30B_A3B_SHAPES,
    "qwen2.5_7b_instruct": QWEN25_7B_INSTRUCT_SHAPES,
}


def pretune_mm(max_tokens, max_reqs, shapes, dtype):
    for M in range(1, max_tokens + 1, 32):
        for N, K in shapes:
            tensor_a = torch.randn([M, K], dtype=dtype, device=device)
            tensor_b = torch.randn([K, N], dtype=dtype, device=device)
            flag_gems.mm(tensor_a, tensor_b)


def pretune_mm_logits(max_tokens, max_reqs, shapes, dtype):
    for M in range(1, max_reqs + 1, 32):
        for N, K in shapes:
            tensor_a = torch.randn([M, K], dtype=dtype, device=device)
            tensor_b = torch.randn([K, N], dtype=dtype, device=device)
            flag_gems.mm(tensor_a, tensor_b)


def pretune_addmm(max_tokens, max_reqs, shapes, dtype):
    for M in range(1, max_tokens + 1, 32):
        for N, K in shapes:
            tensor_a = torch.randn([M, K], dtype=dtype, device=device)
            tensor_b = torch.randn([K, N], dtype=dtype, device=device)
            bias = torch.randn([M, N], dtype=dtype, device=device)
            flag_gems.addmm(bias, tensor_a, tensor_b)


def pretune_index(max_tokens, max_reqs, shapes, dtype):
    import numpy as np

    for M in range(1, max_tokens + 1, 32):
        for N in shapes:
            inp = torch.randn([M, N], dtype=dtype, device=device)
            index = np.random.choice(np.arange(N), size=M, replace=True)
            indices = [
                torch.tensor(index, device=device),
            ]
            flag_gems.index(inp, indices)
            indices[0] = indices[0].to(torch.int32)
            flag_gems.index(inp, indices)


OPERATORS = {
    "mm": pretune_mm,
    "mm_logits": pretune_mm_logits,
    "addmm": pretune_addmm,
    "index": pretune_index,
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
        "--dtype",
        type=str,
        required=False,
        default="bfloat16",
        help="model data type",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=False,
        default=16384,
        help="max tokens",
    )
    parser.add_argument(
        "--max_reqs",
        type=int,
        required=False,
        default=1024,
        help="max requests",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    model = MODEL_SHAPES.get(args.model)
    dtype = DTYPES.get(args.dtype)
    max_tokens = args.max_tokens
    max_reqs = args.max_reqs
    if not model:
        exit(0)
    for op, func in OPERATORS.items():
        shapes = model.get(op)
        if not shapes:
            continue
        func(max_tokens, max_reqs, shapes, dtype)
