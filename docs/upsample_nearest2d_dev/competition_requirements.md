# Competition Requirements for `upsample_nearest2d`

Source pages:

- ModelScope Track 1 statement:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- FlagGems pull requests:
  https://github.com/flagos-ai/FlagGems/pulls
- PyTorch nearest upsampling documentation:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.UpsamplingNearest2d.html
- PyTorch ATen reference:
  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/UpSampleNearest2d.cpp

## Operator

Competition row:

```text
upsample_nearest2d
Difficulty: medium
Category: upsampling
Schema:
upsample_nearest2d(
    Tensor self,
    SymInt[2] output_size,
    float? scales_h=None,
    float? scales_w=None,
) -> Tensor
upsample_nearest2d_backward(
    Tensor grad_output,
    SymInt[2] output_size,
    SymInt[4] input_size,
    float? scales_h=None,
    float? scales_w=None,
) -> Tensor
```

## Hard Requirements

- Match PyTorch semantics for supported forward inputs.
- If backward is submitted, match PyTorch gradient semantics for repeated
  nearest-neighbor mappings.
- Provide pytest coverage for `output_size` and `scales_h/scales_w` paths.
- Provide benchmark coverage using the FlagGems benchmark framework.
- Show speedup versus PyTorch/native baseline. Competition text says the
  device-side speedup must be at least `0.9x`.
- Follow FlagGems repository structure and style.
- PR title must contain:

```text
[FlagGems Operator Development Competition]
```

## Practical Interpretation

The forward path already exists upstream, and a public PR currently targets
backward. To be competitive, the submission needs clear value beyond a minimal
duplicate:

- complete and well-scoped forward/backward integration;
- stronger edge-case tests than the existing public PRs;
- reproducible benchmark logs;
- careful support boundaries for layout, dtype, and scale behavior.

