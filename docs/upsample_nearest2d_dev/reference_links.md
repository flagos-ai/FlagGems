# Reference Links for `upsample_nearest2d`

## Official

- Competition statement:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- PyTorch `UpsamplingNearest2d` docs:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.UpsamplingNearest2d.html
- PyTorch ATen implementation:
  https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/UpSampleNearest2d.cpp

## Local Files

- `src/flag_gems/ops/upsample_nearest2d.py`
- `src/flag_gems/ops/upsample_nearest1d.py`
- `src/flag_gems/ops/upsample_nearest3d.py`
- `tests/test_upsample_nearest2d.py`
- `benchmark/test_upsample_nearest2d.py`
- `conf/operators.yaml`

## Public PRs to Watch

- Main backward competitor:
  https://github.com/flagos-ai/FlagGems/pull/2262
- Older forward/backward work:
  https://github.com/flagos-ai/FlagGems/pull/1635
- Older forward work:
  https://github.com/flagos-ai/FlagGems/pull/1426
- Multi-operator PR, useful only as a rough comparison:
  https://github.com/flagos-ai/FlagGems/pull/2525

