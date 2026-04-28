# `upsample_nearest2d` Implementation Plan

## Stage 0: Baseline Semantics

Before changing kernels, build a small local harness that compares against
PyTorch:

- 4D contiguous input `(N, C, H, W)`;
- integer upsampling, non-integer upsampling, downsampling, and identity;
- `output_size` provided with `scales_h/scales_w=None`;
- explicit `scales_h/scales_w`;
- `float16`, `bfloat16`, and `float32`;
- small shapes and image-like shapes.

Use PyTorch as the semantic source:

```python
torch._C._nn.upsample_nearest2d(input, output_size, scales_h, scales_w)
torch.ops.aten.upsample_nearest2d_backward(
    grad_output, output_size, input_size, scales_h, scales_w
)
```

## Stage 1: Forward Audit

Audit `src/flag_gems/ops/upsample_nearest2d.py` for:

- `output_size` validation;
- scale calculation parity with PyTorch;
- contiguous and non-contiguous behavior;
- channels-last behavior, if supported;
- index overflow handling for large tensors.

Do not widen support silently. If a layout is unsupported, test and document
that boundary or fall back cleanly.

## Stage 2: Backward

Backward is likely the useful competition addition.

Required behavior:

- output shape is `input_size`;
- `grad_output` contributions accumulate when several output pixels map to one
  input pixel;
- non-integer and downsampling scale cases match PyTorch;
- dtype and device behavior match the forward path.

Implementation options:

- atomic-add scatter from output pixels to input pixels;
- tile-based reduction for common integer upsampling factors;
- fast paths for `SAME_H` or `SAME_W`.

Start with a correct atomic implementation, then add fast paths only after
benchmarks expose the bottleneck.

## Stage 3: Integration

Expected files:

- `src/flag_gems/ops/upsample_nearest2d.py`
- `src/flag_gems/ops/upsample_nearest2d_backward.py`
- `src/flag_gems/ops/__init__.py`
- `src/flag_gems/__init__.py`
- `tests/test_upsample_nearest2d.py`
- `tests/test_upsample_nearest2d_backward.py`
- `benchmark/test_upsample_nearest2d.py`
- `benchmark/test_upsample_nearest2d_backward.py`

## Stage 4: Competition Proof

Before opening a PR, collect:

- quick and full accuracy logs;
- benchmark logs for forward and backward;
- a small table of speedup by dtype and shape group;
- notes on supported layouts and unsupported cases;
- comparison with current public PRs, especially `#2262`.

