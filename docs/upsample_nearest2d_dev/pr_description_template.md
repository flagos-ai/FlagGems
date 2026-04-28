# [FlagGems Operator Development Competition] Add upsample_nearest2d forward/backward

## Summary

This PR adds or improves `upsample_nearest2d` support in FlagGems.

Target API:

```python
torch._C._nn.upsample_nearest2d(input, output_size, scales_h=None, scales_w=None)
```

If backward is included:

```python
torch.ops.aten.upsample_nearest2d_backward(
    grad_output, output_size, input_size, scales_h=None, scales_w=None
)
```

## Correctness

Commands:

```bash
pytest tests/test_upsample_nearest2d.py -s
pytest tests/test_upsample_nearest2d_backward.py -s
```

Coverage:

- integer upsample
- non-integer upsample
- downsample
- identity scale
- `output_size`
- `scales_h/scales_w`
- dtype coverage

## Benchmark

Commands:

```bash
pytest benchmark/test_upsample_nearest2d.py -s --record log
pytest benchmark/test_upsample_nearest2d_backward.py -s --record log
```

Results:

```text
Device:
Forward:
Backward:
```

## Support Boundaries

- Supported layouts:
- Unsupported layouts:
- Known weak cases:

