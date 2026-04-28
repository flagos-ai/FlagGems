# `upsample_nearest2d` Benchmark Plan

## Goals

- Show speedup versus PyTorch/native baseline.
- Measure forward and backward separately.
- Include enough shapes to expose memory-bandwidth behavior.

## Forward Shapes

Use image-like shapes:

```text
(1, 3, 224, 224) -> x2
(8, 3, 224, 224) -> x2
(1, 64, 56, 56) -> x2
(16, 64, 56, 56) -> x2
(8, 128, 28, 28) -> x2
(4, 256, 14, 14) -> x2
(1, 64, 128, 256) -> x2
```

Also include non-integer and downsampling scales:

```text
(2.1, 3.7)
(1.3, 5.1)
(0.5, 0.5)
(0.3, 0.5)
```

## Backward Shapes

Mirror the forward shapes. For backward, report both:

- `grad_output -> grad_input` time;
- speedup by scale group, because integer upsampling and downsampling have
  different accumulation patterns.

## Commands

```bash
pytest benchmark/test_upsample_nearest2d.py -s --record log
pytest benchmark/test_upsample_nearest2d_backward.py -s --record log
```

If the backward benchmark file does not exist yet, create it as part of the
implementation.

## Report Template

```text
Device:
PyTorch:
Triton:
FlagGems commit:

Forward:
shape dtype scale torch_ms gems_ms speedup pass

Backward:
shape dtype scale torch_ms gems_ms speedup pass

Known weak cases:
```

