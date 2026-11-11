# dyn_quant_pack_4bit_weight 修复方案

## 基线
- `pytest -m dyn_quant_pack_4bit_weight --ref cpu`：16 failed / 12 passed。

## 根因
backend override `_kunlunxin/ops/_dyn_quant_pack_4bit_weight.py` 的**参数契约校验不完整**（相对 generic 实现缺失）。导致：
- `block_size=0` 时 `in_features % block_size` 抛 `ZeroDivisionError`（应 RuntimeError）。
- 缺失 in_features 奇偶/正负、out_features、weights rank/shape、scales_zeros/bias 的 dtype/rank/shape/length 校验 → 无效输入未抛 RuntimeError（`DID NOT RAISE`）。

## 改动
将 generic `_dyn_quant_pack_4bit_weight` 的完整契约校验（in_features 正偶数 / out_features ≥0 / block_size>0 / weights 2D 形状 / scales_zeros 形状与 dtype / bias 形状与 dtype / device 一致）镜像进 backend override；`block_size <= 0` 校验置于任何取模之前。

## 验证
- `pytest -m dyn_quant_pack_4bit_weight --ref cpu`：**28 passed**（原 16 failed / 12 passed）。

## 说明
纯校验补全，不改 kernel 计算路径，无 fallback。
