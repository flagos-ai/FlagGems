# Operator repair memory

## Repair branch
- Branch: `klx/repair-main`
- Worktree: `/workspace/fgfix_main`
- Base: `master` at `a7620cc19` (2026-08-28)
- The original `/workspace/FlagGems` worktree has user-owned uncommitted files; they are intentionally preserved and are not part of this branch.

## Integrated repairs
The repair branch contains these validated operator fixes; merge commits preserve source branch history:
- `acos/arccos`: `47c552a38`
- `adaptive_avg_pool2d` forward/backward: `0cf438215`, `9a6b1f769`
- `adaptive_max_pool3d` forward/backward: `9ef02dc00`
- `amin/amax` multidimensional plus `aminmax`/`vector_norm`: `9624c5f75`, `3ce8f6b2e`
- `arctan2`, `atan2_`, `asin/arcsin`, `atanh_`: `0babd93dc`, `b8b5f4dd0`, `52ed65dbd`, `05b655c74`
- `avg_pool2d` backward and `avg_pool3d`: `6a3d2527c`, `603a585ae`
- `batch_norm` family and `cat`: `453acfa46`, `9b67f38fb`
- complex/floor divide, `gcd`, `hypot_`, `logaddexp`: `b10fbd782`, `2d70ea1d3`, `33e7e8a9c`, `745b91f15`, `8eb3453f4`
- `tril` stride and max-pool indices families: `0f4f3dd59`, `bc68137e9`, `d7b6c3077`
- `nextafter`, `nll_loss`, `nonzero/unique`: `0193b7a2f`, `5896600f1`, `321b10994`
- scaled softmax, select backward, stable sort: `cb1616f8c`, `f706b5f98`, `2df12081e`
- `square`: `7b780d450`; upsample family: `0cb37c889`; var/std/mean/sum/norm family: `8d14d46a5`

## Validation
- `unique_dim`: 352 passed on the source repair branch.
- `square`: 54 passed on the source repair branch.
- Targeted nonzero/unique/unique_consecutive cases passed.
- `python -m compileall -q src` passes on `klx/repair-main`; no merge conflict markers remain.
- Black and `git diff --check` passed on source repair branches. The remote image lacks `ruff`, `flake8`, and `isort`; pre-commit bootstrap was not completed.

## Excluded branches
- `klx/square-fix` duplicates `klx/square-family-fix`.
- `klx/adaptive-avg-pool2d-fix` duplicates the clean forward fix.
- `klx/atanh-inplace-fix`, `clamp-min-fix`, `scatter-fix`, and `softmax-out-native-fix` contain old/shared or unrelated history; use only a reviewed final diff if revisited.
- `klx/pr2695-*` and `klx/test-triton-add-p800-fix` are infrastructure/test branches, not operator implementation fixes.
- `klx/glu-family-skip` changes test policy only and is not an implementation repair.

## Mandatory check before a new repair
1. Search this document for the operator and aliases (especially `nonzero`, `unique`, and `square`).
2. Run `git log --all -- <operator file>` and inspect the relevant source branch and tests/reports.
3. If a repair is listed, validate or extend it instead of reimplementing it; record new evidence here.
4. Only after the check finds no existing fix, create a new branch from `klx/repair-main` and add the operator repair.
