# `upsample_nearest2d` Test Matrix

Use this as the coverage checklist for pytest.

## Forward

- Shapes:
  - `(1, 1, 1, 1)`
  - `(1, 3, 8, 8)`
  - `(2, 3, 32, 32)`
  - `(4, 16, 64, 64)`
  - `(1, 64, 128, 128)`
- Scale/output cases:
  - identity: `(1.0, 1.0)`
  - integer upsample: `(2.0, 2.0)`, `(4.0, 4.0)`
  - non-integer upsample: `(2.1, 3.7)`, `(1.3, 5.1)`
  - downsample: `(0.5, 0.5)`, `(0.3, 0.5)`
  - asymmetric: `(1.0, 2.0)`, `(2.0, 1.0)`
- Argument paths:
  - `output_size` only;
  - explicit `scales_h/scales_w`;
  - `output_size` and scales together, matching PyTorch behavior.
- Dtypes:
  - `torch.float16`
  - `torch.bfloat16`
  - `torch.float32`
- Layouts:
  - contiguous NCHW;
  - transposed/non-contiguous input, if supported;
  - channels-last input, if supported.

## Backward

- Same shape and scale groups as forward.
- Include integer upsampling where many outputs map to one input.
- Include downsampling where some input positions receive no gradient.
- Compare against:

```python
torch.ops.aten.upsample_nearest2d_backward(
    grad_output,
    output_size,
    input_size,
    scales_h,
    scales_w,
)
```

## Tolerances

Nearest upsampling is copy/add heavy, so tolerances should normally be strict:

- `float32`: `rtol=1e-5`, `atol=1e-6`
- `float16`: `rtol=1e-3`, `atol=1e-3`
- `bfloat16`: `rtol=1e-2`, `atol=1e-2`

Adjust only when PyTorch reference behavior or backend numerics justify it.

