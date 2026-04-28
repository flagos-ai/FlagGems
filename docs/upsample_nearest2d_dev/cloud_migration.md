# Cloud Migration Notes

This workspace is intended to be copied to a GPU cloud machine.

## Before Uploading

From the repository root:

```bash
git status --short
git branch --show-current
git remote -v
```

Expected branch:

```text
competition/upsample-nearest2d-dev
```

## After Uploading

Install editable dependencies:

```bash
bash scripts/upsample_nearest2d/setup_cloud_env.sh
```

Check GPU visibility:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
PY
```

## Validation Loop

```bash
bash scripts/upsample_nearest2d/run_accuracy_quick.sh
bash scripts/upsample_nearest2d/run_accuracy_full.sh
bash scripts/upsample_nearest2d/run_benchmark.sh
```

## PR Readiness

```bash
bash scripts/upsample_nearest2d/check_pr_ready.sh
```

Attach the resulting accuracy and benchmark logs to the PR description.

