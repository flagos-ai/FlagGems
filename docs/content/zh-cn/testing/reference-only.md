---
title: Benchmark reference-only
weight: 50
---

# Benchmark reference-only：检查原始 core baseline

`--reference-only` 仅用于 benchmark，默认 level 为 `core`，无需显式传入；可通过 `--level` 覆盖。普通 benchmark 仍默认 `comprehensive`。该模式执行原始 baseline，不需要 candidate，不使用 override，不做 warmup、计时、正确性比较或 Profile，只检查输入生成和性能 reference 能否运行，不能据此声称正确性 pytest 的 reference 或完整测试链路已经通过。

```bash
pytest -q benchmark/test_negative.py --reference-only --output benchmark-reference.json
```

可沿用单独一次 `--list-cases --level core` 得到的 `--case-id` 精确重放；列举用例时需与实际执行 level 匹配。KGS review 会使用整个 core 集合，不缩减 dtype 或 workload。输出每次覆盖写入，不累计旧结果。先核对 `flag_gems.__file__` 来自预期 checkout，避免旧 editable install 导致使用另一版本。

## 复用与边界

标准 case-based Benchmark 复用 `build_inputs()`、`unpack_to_args_kwargs()` 和原 `torch_op`。普通计时与 reference-only 共用 forward/backward callable 构造；backward 执行原 forward 和 `torch.autograd.grad`，但不调用计时器。每个 case 执行一次并同步，结束后释放输入和计算图，不进入 candidate 的 `use_gems`、`gems_op` 或 override 分支。

没有 case builder，或自定义 `run/get_latency/_measure_input` 的 benchmark 明确报告 `UNSUPPORTED`，不能绕过特殊 baseline 逻辑后宣称已验证。原 `skip_native` 和 pytest skip 条件保留。该模式不是任意 Python 测试代码的沙箱。

本接口不修改或执行正确性 pytest，不提供 correctness reference 的 marker、包装器或截断逻辑。正确性测试继续做源码 review，生成候选后运行原完整正确性测试。性能和正确性 reference 的 dtype、精度、设备、shape 可能不同，不能相互替代。

`--reference-only` 与 `--override`、`--override-config`、`--preflight-only`、`--profile-only`、`--list-cases`、`--query`、benchmark `--parallel` 及 xdist 并发互斥。不能与 `tests/` 下的正确性用例混跑。

## 报告

```json
{
  "schema_version": "flaggems.reference/v1",
  "phase": "timing",
  "status": "PASSED",
  "records": [{
    "nodeid": "benchmark/test_negative.py::test_negative",
    "operator": "negative",
    "case_id": "benchmark/test_negative.py::test_negative::core::float32::0",
    "count": 1,
    "status": "PASSED"
  }]
}
```

报告区分 `PASSED`、`FAILED`、`UNSUPPORTED`、`ALL_SKIP`、`NO_CASES`，不含 latency/speedup。pytest 阶段级失败或跳过另带 `pytest_phase`。原 pytest 中途 skip 时，整个 node 按源语义跳过，之前的调用只保留为执行证据，不据此声称该 node 完整通过。全部跳过可能仍为 pytest exit code 0，因此调用方必须检查结构化状态，不能仅凭进程退出码判断就绪。

KGS 配套通过设备 slot 和隔离 worker 调用此命令，核对冻结 benchmark fingerprint 与 core case 覆盖，保存原报告；该执行事实不是模型审核结论，也不写入候选优化 ledger。KG 的 `skip_review` 仅跳过模型审核，不跳过这项目标验证。

### 逐 case 失败信息

每个已选 case 保留 `case_id`、`ordinal`、`dtype`、`shape`、`params` 和执行状态。失败记录额外包含 `stage`（`build_inputs`、`prepare_reference`、`invoke` 或 `synchronize`）与 `failure`（`category`、异常 `type`、原始 `message`、`traceback`），用于区分输入构造、原 reference 调用和设备同步失败。`dtype` 来自 case 声明；backward 或原始 reference 内部可能转换 dtype，具体报错仍以原始异常为准。

`category` 是保守的诊断提示，不是设备能力表：`DTYPE_UNSUPPORTED` 仅识别明确的 PyTorch `not implemented for '<dtype>'` 信息；`API_MISSING` 仅用于实际访问缺失的 `torch` 模块属性；`NOT_IMPLEMENTED` 保留无法确定是整个 API、后端还是参数组合不支持的 `NotImplementedError`；其余为 `UNKNOWN`。不能从某一个 dtype 或某个输入失败推断整个算子不可用。

明确的上述能力错误在设备同步正常后继续下一个 case，所有失败仍逐条保留且总体为 `FAILED`。未知错误、同步失败或中断立即停止当前 node，剩余已选 case 标为 `NOT_RUN`，不伪造失败原因。错误后的同步若再次失败，另存 `recovery_failure`，保留首个异常。原 pytest skip 仍终止整个 node；未完成的 case 不计作通过。收集或 setup 阶段尚未生成 case 时只能提供 pytest 阶段级错误；进程被强杀时也不承诺完整报告。

## 验证范围

分支基于官方 `master@d17e23e48e26b6396bd3fc899c95dac74f305cbe`，已撤回早期 correctness reference-only 试验，对应正确性 pytest 与该 master 保持一致。Host 测试使用现有 `kernelgen-nvidia-cu128` 容器，验证原 baseline、backward、精确 case 选择、skip/异常、配置互斥与既有 Preflight/Profile 路径；设备同步使用模拟实现。未安装或升级依赖，未进行 GPU/跨芯片验收。
