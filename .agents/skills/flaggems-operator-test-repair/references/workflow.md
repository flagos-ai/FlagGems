# 本机命令与验收

在仓库根目录、目标后端真实运行时执行。以下变量由本任务按实际源码映射设置：`TEST_FILE`、`BENCH_FILE`、`MARKER`、`OP`、`REF_CANDIDATE`、`DUMMY_CANDIDATE`、`RUN_DIR`。候选文件中的函数示例名为 `run`；先核对当前版本的 CLI 与 schema，不从其他专用测试流程照搬参数。

## 原始环境与覆盖

记录 Git commit、实际 Python/Torch/厂商插件及加载模块路径、设备型号与可见性。区分原生执行和 SDK CPU fallback；不能以设备可见证明原生支持。调试 CPU fallback 在每个测试进程关闭并核验，避免掩盖后端缺失能力；不得修改其他任务的环境或设备状态。

保存原正确性 callspec 和 core case 元数据，包括 dtype、shape、stride、offset、标量、开关、None、外层参数及 alias/mutation。以完整的带类型联合组合比较覆盖，不能仅比较 case 数或各参数的边际集合。列表构造失败时先从原配置/builder 还原，不删除出错的输入组合。

## 完整正确性：ref-as-candidate

`REF_CANDIDATE` 是与原 API/overload 契约一致、已核验的 reference 包装；不能再次调用被 override 的 flag_gems 导出导致递归，也不能只是调用待测 candidate。`--ref cpu` 改 expected reference 的设备，不会自动替换 candidate。先排除整个 API 不可用的算子，本流程不新增 CPU oracle。

对于仅部分参数不支持的 API，分别核对 expected reference 与验证用 candidate 的能力：默认模式保留有依据的局部 skip；CPU 模式若 CPU 原生 API 支持该参数，ref-as-candidate 可以使用同 API 的 CPU 原生包装以保留覆盖。包装须正确搬回目标设备并保持 dtype、结构、out/mutation/alias 契约，不能用新数学 oracle 替代或换 API。两种模式可分别设置 REF_CANDIDATE；这不改变 benchmark 默认原生 reference 的精确 skip。无法保持契约时记录阻断，不因默认模式的失败直接跳过 CPU 模式。

```sh
pytest "$TEST_FILE" -m "$MARKER" --collect-only -q
pytest "$TEST_FILE" -m "$MARKER" --ref cpu --collect-only -q
pytest "$TEST_FILE" -m "$MARKER" --override "$OP:$REF_CANDIDATE:run"
pytest "$TEST_FILE" -m "$MARKER" --ref cpu --override "$OP:$REF_CANDIDATE:run"
```

跑完整参数集，不加 quick、-k 或 case 筛选。逐模式保留 pass/fail/skip 与 node ID。部分参数不支持按模式精确 skip；默认模式失败不自动授权 CPU 模式 skip。若 CPU 模式另有部分 dtype 不支持，单独保存失败依据。

## core list-cases 与实际 benchmark

```sh
pytest "$BENCH_FILE" -m "$MARKER" --level core --list-cases \
  --record json --output "$RUN_DIR/core-cases.json"
pytest "$BENCH_FILE" -m "$MARKER" --level core \
  --override "$OP:$REF_CANDIDATE:run" \
  --mode operator --warmup 0 --iter 1 \
  --record json --output "$RUN_DIR/core-timings.json"
```

低迭代计时用于确认 benchmark 函数能跑，不作为正式性能发布。接近 1 的 ratio 是同机自比较的辅助观察，不是硬阈值。

必查：catalog 非空、ID 唯一、operator 与 Definition 和 override 一致、原 core 联合参数完整保留、没有混入其他 overload。一个 pytest 节点中的每个内层 core case 都要列举。部分不支持项仍保留在 catalog，通过精确 skip 记录；支持项实际计时有限且 candidate_source 为 override。list/preflight 不能代替真实计时，不给未执行 case 填假 latency。

## dummy 注入

先发现当前 KernelGen/KGS 官方 dummy 检查工具和 capability，使用其当前接口。普通 FlagGems 的基本注入方式为：

```sh
pytest "$TEST_FILE" -m "$MARKER" --override "$OP:$DUMMY_CANDIDATE:run"
pytest "$TEST_FILE" -m "$MARKER" --ref cpu --override "$OP:$DUMMY_CANDIDATE:run"
```

dummy 接受真实参数并必抛带唯一身份/nonce 的专用异常。以上裸 pytest 命令可能遇首个失败后停止，或在 benchmark 首个内层 case 中断；正式检查工具必须逐 node/内层 case 遍历并记录真实调用，不能以一次异常证明全部命中。benchmark 探针可使用当前 `--preflight-only`，但仍必须带 override；不带 override 不能证明 ref-as-candidate 或注入。不要把 SGLang 专用参数直接套进来。

分别报告 REACHED、BYPASSED、PRE_CALL_ERROR、生命周期错误、NOT_RUN、SKIP。合法期望异常可能捕获 dummy，须靠直接调用记录确认；pytest 退出 1 可能是预期 dummy，也可能是真失败，必须读逐项结果。除精确 skip 外均应实际命中，零 case/全 skip 不通过。dummy 的结果不作为有效数值或性能数据。

## 真实 KGS inspect

使用当前环境已有的真实 KGS discovery/inspect 接口或生产 adapter，记录版本、完整命令/请求、原始响应及实际加载源码。不要杜撰固定 CLI。将 inspect 中的 operator、ID、dtype、shape、params/global ordinal 与真实 core catalog 逐 case 对齐。仅静态查看、模拟响应或普通 pytest 通过不能代替 inspect。

若当前机器缺少可用的 KGS 检查接口，记录为环境阻断；不引入远程机器配置，不据此宣称验收完成或提前提 PR。

## 证据与 review

按算子、当前后端、reference 模式保存原始/修改后 collection、数值结果、dummy 逐项命中、list-cases、inspect 与实际 core 计时。记录源码提交、完整命令、返回码、环境、case ID/数量、skip 的原失败依据；源码变化后补跑受影响检查，不能修改旧 raw 冒称新提交通过。

两个独立 review 轮次核对代码范围、Definition 契约、原联合覆盖、注入和全部实机证据。任何必过检查未运行或失败都保持未完成；整体 API 不可用单列“暂不处理”，不能算修复成功。只有文档/skill 自身改动时验证 skill 结构、引用和行为边界，不启动无关算子或 GPU 测试。
