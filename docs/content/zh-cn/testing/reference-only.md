---
title: Benchmark reference-only
weight: 50
---

# Benchmark reference-only：检查原始 core baseline

`--reference-only` 仅用于 benchmark，默认 level 为 `core`，无需显式传入；可通过 `--level` 覆盖。普通 benchmark 仍默认 `comprehensive`。该模式通过 `get_latency(self.torch_op, *args, **kwargs)` 复用 forward/backward 和 fresh-input 准备，在共享计时层使用 `measure_calls(fn, warmup_calls=0, repeat_calls=1)`：不预热，只执行一次。不需要 candidate，不使用 override，不做正确性比较、Profile 或加速比计算；不能据此声称正确性 pytest 的 reference 或完整测试链路已经通过。

```bash
pytest -q benchmark/test_negative.py --reference-only --output benchmark-reference.json
```

可沿用单独一次 `--list-cases --level core` 得到的 `--case-id` 精确重放；列举用例时需与实际执行 level 匹配。KGS review 会使用整个 core 集合，不缩减 dtype 或 workload。输出每次覆盖写入，不累计旧结果。先核对 `flag_gems.__file__` 来自预期 checkout，避免旧 editable install 导致使用另一版本。

## 复用与边界

标准 case-based Benchmark 复用 `build_inputs()`、`unpack_to_args_kwargs()` 和 `get_latency()`，明确传入原 `torch_op`。共享计时封装使用调用次数，不将 `--warmup/--iter` 的毫秒值当作次数；reference-only 不进入自动校准、重复采样或 CUDA Graph 捕获。它同步设备并记录单次 wall time，可能包含首次编译成本，只是就绪诊断，不是稳态 kernel latency。backward 沿用一次 forward 建图及一次 grad 调用；fresh-input 在测量前复制输入一次。结束后释放输入，不进入 candidate 的 `use_gems`、`gems_op` 或 override 分支。普通 benchmark 的计时模式、预热和重复测量完全不变。

没有 case builder 或自行覆盖 `run` 绕过共享执行入口的 benchmark，仍明确报告 `UNSUPPORTED`。原 `skip_native` 和 pytest skip 条件保留。该模式不是任意 Python 测试代码的沙箱。

无需装饰器或逐个 pytest 声明，也不比较方法是否与基类为同一个函数对象。`get_latency(op, ...)` 的统一契约就是测量传入的 `op`；自定义包装器应委托共享计时层，不得忽略参数而调用 Gems candidate 或自行绕过单次策略。这个封装不承诺控制任意第三方黑盒计时器。`OperatorBenchmark` 已复用此路径。reference-only 不调用同时测量两侧的 `_measure_input`；仅在该方法中定义的自定义 baseline 语义须迁入 `torch_op`/`get_latency` 的统一契约，不能假定本模式会执行它。

本接口不修改或执行正确性 pytest，不提供 correctness reference 的 marker、包装器或截断逻辑。正确性测试继续做源码 review，生成候选后运行原完整正确性测试。性能和正确性 reference 的 dtype、精度、设备、shape 可能不同，不能相互替代。

`--reference-only` 与 `--override`、`--override-config`、`--preflight-only`、`--profile-only`、`--list-cases`、`--query`、benchmark `--parallel` 及 xdist 并发互斥。不能与 `tests/` 下的正确性用例混跑。

## 报告

```json
{
  "schema_version": "flaggems.reference/v2",
  "phase": "timing",
  "status": "PASSED",
  "records": [{
    "nodeid": "benchmark/test_negative.py::test_negative",
    "operator": "negative",
    "case_id": "benchmark/test_negative.py::test_negative::core::float32::0",
    "latency_ms": 0.012,
    "status": "PASSED"
  }]
}
```

报告区分 `PASSED`、`FAILED`、`UNSUPPORTED`、`ALL_SKIP`、`NO_CASES`，成功测量保存诊断用 `latency_ms`，不含 speedup 或计时器内部调用 count。耗时须有限且非负；允许计时器返回分辨率内的零值，但零值不证明具有有效 headline performance。报告使用 `flaggems.reference/v2` 区别旧版不含测量值的报告，调用方需要同步更新校验。pytest 阶段级失败或跳过另带 `pytest_phase`。原 pytest 中途 skip 时，整个 node 按源语义跳过，之前的调用只保留为执行证据，不据此声称该 node 完整通过。全部跳过可能仍为 pytest exit code 0，因此调用方必须检查结构化状态，不能仅凭进程退出码判断就绪。

KGS 配套通过设备 slot 和隔离 worker 调用此命令，核对冻结 benchmark fingerprint 与 core case 覆盖，保存原报告；该执行事实不是模型审核结论，也不写入候选优化 ledger。KG 的 `skip_review` 仅跳过模型审核，不跳过这项目标验证。

### 逐 case 失败信息

每个已选 case 保留 `case_id`、`ordinal`、`dtype`、`shape`、`params` 和执行状态。失败记录额外包含 `stage`（`build_inputs`、`benchmark_reference` 或 `synchronize`）与 `failure`（`category`、异常 `type`、原始 `message`、`traceback`），用于区分输入构造、原 baseline 测量和外层设备同步失败。共享计时封装内部的调用或同步异常都属于 `benchmark_reference`，由原 traceback 定位。`dtype` 来自 case 声明；backward 或原始 reference 内部可能转换 dtype，具体报错仍以原始异常为准。

`category` 是保守的诊断提示，不是设备能力表：`DTYPE_UNSUPPORTED` 仅识别明确的 PyTorch `not implemented for '<dtype>'` 信息；`API_MISSING` 仅用于实际访问缺失的 `torch` 模块属性；`NOT_IMPLEMENTED` 保留无法确定是整个 API、后端还是参数组合不支持的 `NotImplementedError`；其余为 `UNKNOWN`。不能从某一个 dtype 或某个输入失败推断整个算子不可用。

明确的上述能力错误在设备同步正常后继续下一个 case，所有失败仍逐条保留且总体为 `FAILED`。未知错误、同步失败或中断立即停止当前 node，剩余已选 case 标为 `NOT_RUN`，不伪造失败原因。错误后的同步若再次失败，另存 `recovery_failure`，保留首个异常。原 pytest skip 仍终止整个 node；未完成的 case 不计作通过。收集或 setup 阶段尚未生成 case 时只能提供 pytest 阶段级错误；进程被强杀时也不承诺完整报告。

## 验证范围

当前封装基于官方 `kernelgen-dev@01522dc6`，不修改算子 correctness pytest。Host 测试使用现有 `kernelgen-nvidia-cu128` 容器，验证固定调用次数、普通 benchmark 不变、fresh-input、backward、精确 case 选择、skip/异常、配置互斥与既有 Preflight/Profile 路径；设备同步使用模拟实现。未安装或升级依赖，本轮尚未进行 GPU/跨芯片验收。此前 70 算子的 reference-only 结果属于旧实现，不能作为此次新计时封装的真机验收。
