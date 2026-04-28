# PR Checklist for `upsample_nearest2d`

## Title

```text
[FlagGems Operator Development Competition] Add upsample_nearest2d forward/backward
```

Adjust the final wording to match the actual scope.

## Files

- [ ] `src/flag_gems/ops/upsample_nearest2d.py` audited or updated.
- [ ] `src/flag_gems/ops/upsample_nearest2d_backward.py` added if backward is included.
- [ ] `src/flag_gems/ops/__init__.py` exports the operator(s).
- [ ] `src/flag_gems/__init__.py` registers the ATen override(s).
- [ ] `tests/test_upsample_nearest2d.py` covers forward.
- [ ] `tests/test_upsample_nearest2d_backward.py` covers backward if included.
- [ ] `benchmark/test_upsample_nearest2d.py` covers forward.
- [ ] `benchmark/test_upsample_nearest2d_backward.py` covers backward if included.

## Correctness

- [ ] `output_size` path matches PyTorch.
- [ ] `scales_h/scales_w` path matches PyTorch.
- [ ] Integer upsample, non-integer upsample, identity, and downsample pass.
- [ ] Dtypes include `float16`, `bfloat16`, and `float32` where supported.
- [ ] Layout support is tested or explicitly bounded.
- [ ] Backward accumulation matches PyTorch.

## Performance

- [ ] Forward benchmark logs are attached.
- [ ] Backward benchmark logs are attached if backward is included.
- [ ] Weak cases are disclosed instead of hidden.
- [ ] Speedup is at least `0.9x` on the reported device-side benchmark.

## Hygiene

- [ ] No unrelated files or generated caches.
- [ ] No broad formatting churn.
- [ ] Public competing PRs have been checked for overlap.
- [ ] PR description includes commands, logs, and support boundaries.

