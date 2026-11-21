# Dense FlashAttention dropout: equations and test contract

For each batch/head, let `Q`, `K`, `V` be the query, key and value matrices,
`s = 1 / sqrt(head_dim)`, and `p` the dropout probability. Causal masking is
applied to the score matrix before softmax. For the tested square case, entries
above the main diagonal have score negative infinity.

```text
S      = s Q K^T + causal_mask
LSE_i  = log(sum_j exp(S_ij))
P      = softmax(S)
R_ij   = M_ij / (1 - p),  M_ij in {0, 1}
P_drop = P * R
O      = P_drop V
```

`*` denotes elementwise multiplication. The dropout mask is sampled once in
forward and held constant for differentiation. Dropout changes the value path;
it does not change softmax normalization or LSE. At `p = 0`, use `R = 1`.

Given the upstream gradient `G = dL/dO`, the chain rule gives:

```text
dV     = P_drop^T G
dP     = G V^T
dP_in  = dP * R
D_i    = sum_j P_ij dP_in_ij = sum_d G_id O_id
dS     = P * (dP_in - D)
dQ     = s dS K
dK     = s dS^T Q
```

The outer factor in `dS` must be the original `P`, including where a dropout
mask bit is zero. Softmax couples all keys within a row: a dropped probability
can still receive a nonzero score gradient through the normalization term.
Using `P_drop` as this outer factor incorrectly drops and rescales that term.
The `dV` path does require `P_drop`. These distinct roles explain why a backward
kernel must retain the undropped probability instead of overwriting it.

The dense backward kernels in `src/flag_gems/ops/flash_attention_backward.py`
follow these roles. The Hygon forward implementation is in
`src/flag_gems/runtime/backend/_hygon/ops/_flash_attention_forward.py`.
FP16 uses low-precision dot products with FP32 accumulation; FP32 dense backward
uses IEEE dot products. The Hygon low-precision dV path retains the probability
conversion residual using an additional dot product.

## Correctness evidence

`tests/test_hygon_flash_attention_dropout_autograd.py` differentiates the CPU
FP64 forward expression with `torch.autograd.grad` and compares O, LSE and all
three gradients to the Hygon dense forward/backward pair. It replays the returned
Philox seed/offset mask, including a nonzero starting reservation. It covers
FP16/FP32, causal/non-causal, dropout 0.2/0.5 and sequence/head dimensions 64/128.
It reuses the existing mask replay helper; it does not claim independent RNG
algorithm validation or cross-implementation random sequence compatibility.

Four CPU controls compare the existing hand-derived reference to autograd for
zero/nonzero dropout. The two nonzero-dropout controls also show that replacing
the outer P with P_drop produces incorrect dQ and dK at the hardware-test
FP16 tolerances. Existing FP16/FP32 suites retain zero-dropout, mask replay,
reservation progression and shape/stride coverage.

This contract is dense fixed-length contiguous square BSHD MHA through
`flag_gems._flash_attention_forward` and `flag_gems.flash_attention_backward`.
For nonzero dropout the FlagGems forward returns size-one int64 seed and offset
tensors so the Hygon backward can distinguish its `tl.rand` mask from the native
Hygon scalar seed/offset ABI. Pass both tensors through to backward unchanged.
Variable-length kernels are handled separately by PR #9116. Direct dense tests
do not validate public SDPA dropout, mixed boxing-forward/FlagGems-backward,
rectangular attention or arbitrary strides, and do not close issues #8748 or
Torch-FL #795.
