# Upsample Nearest2d Competition Workspace

This repository is a FlagGems development workspace prepared for the
`upsample_nearest2d` operator task in the FlagOS / ModelScope Track 1
competition.

Target API:

```python
torch._C._nn.upsample_nearest2d(
    input,
    output_size,
    scales_h=None,
    scales_w=None,
)
```

The practical competition target is stronger than simply having a forward
kernel: the PR should cover PyTorch-compatible forward semantics, backward
coverage if submitted, benchmark evidence, and clean integration into the
FlagGems operator registry.

## Start Here

- `docs/upsample_nearest2d_dev/competition_requirements.md`
- `docs/upsample_nearest2d_dev/implementation_plan.md`
- `docs/upsample_nearest2d_dev/test_matrix.md`
- `docs/upsample_nearest2d_dev/benchmark_plan.md`
- `docs/upsample_nearest2d_dev/pr_checklist.md`
- `docs/upsample_nearest2d_dev/reference_links.md`
- `docs/upsample_nearest2d_dev/cloud_migration.md`

## Cloud Setup

On the cloud machine, from the repository root:

```bash
bash scripts/upsample_nearest2d/setup_cloud_env.sh
```

After implementing or modifying the operator:

```bash
bash scripts/upsample_nearest2d/run_accuracy_quick.sh
bash scripts/upsample_nearest2d/run_accuracy_full.sh
bash scripts/upsample_nearest2d/run_benchmark.sh
bash scripts/upsample_nearest2d/check_pr_ready.sh
```

## Expected Work Areas

- `src/flag_gems/ops/upsample_nearest2d.py`
- `src/flag_gems/ops/upsample_nearest2d_backward.py` if you implement backward
- `src/flag_gems/ops/__init__.py`
- `src/flag_gems/__init__.py`
- `tests/test_upsample_nearest2d.py`
- `tests/test_upsample_nearest2d_backward.py` if backward is included
- `benchmark/test_upsample_nearest2d.py`
- `benchmark/test_upsample_nearest2d_backward.py` if backward is included

## Current Competitive Risk

The main public risk is PR `#2262`, which targets
`upsample_nearest2d_backward` and has clean CI checks. A strong submission
should therefore avoid being a narrow duplicate. Prefer one of these angles:

- forward and backward in a clean, maintainable shape;
- broader PyTorch semantic coverage, especially `scales_h/scales_w`;
- non-contiguous or channels-last coverage if supported;
- benchmark logs showing speedup on meaningful image-like shapes.

