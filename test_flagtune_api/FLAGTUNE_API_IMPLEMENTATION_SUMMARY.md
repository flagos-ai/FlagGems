# FlagTune API 实现总结

本文档总结本次在 `flagtune_api_gxm` 分支上完成的 FlagTune API 改造、实现思路、使用方法、验证方式和后续扩展方式。

## 1. 背景和目标

原来的 FlagTune 开关主要依赖环境变量：

```bash
USE_FLAGTUNE=1
```

但旧实现有两个明显问题：

1. `USE_FLAGTUNE` 是全局开关。只要环境变量打开，所有写了 `USE_FLAGTUNE` 分支的算子都会进入 expanded tuning space，比如 `addmm`、`baddbmm`、`bmm`、`mv`、`mm`、`w8a8` 等。
2. 旧代码在 kernel 装饰器创建 `LibTuner` 时读取 `USE_FLAGTUNE`。这发生在 `import flag_gems` 阶段，用户如果后面再调用 Python API，很多 tuner 配置已经固定，API 很难生效。

本次目标是实现一个更小、更明确的 API：

```python
flag_gems.flagtune(include="mm, bmm")
```

目标语义：

- 只有 `include` 里的算子开启 FlagTune。
- 内置 registry 默认注册 `mm` 和 `bmm`。
- 后续新增算子时，通过 `register_flagtune_op(...)` 注册，而不是继续在 `flagtune.py` 里手写多个固定集合。
- `USE_FLAGTUNE=1` 仍然作为总开关，但它不再让其他算子误进入 FlagTune。
- `bmm` 的 expanded YAML 从旧位置移动到 Hopper backend 目录。
- `configloader.py` 根据当前 backend/arch 自动查找对应 YAML。

## 2. 用户如何使用

推荐用法：

```python
import flag_gems

flag_gems.flagtune(include="mm, bmm")
flag_gems.enable()

# 后续 torch.mm / torch.bmm 会走 FlagGems，并且 mm、bmm 会启用 FlagTune
```

只打开 `bmm`：

```python
import flag_gems

flag_gems.flagtune(include="bmm")
flag_gems.enable()
```

只打开 `mm`：

```python
import flag_gems

flag_gems.flagtune(include="mm")
flag_gems.enable()
```

也可以传 iterable：

```python
flag_gems.flagtune(include=["mm", "bmm"])
```

默认参数是：

```python
flag_gems.flagtune()
```

等价于：

```python
flag_gems.flagtune(include="mm, bmm")
```

如果传入不支持的算子，会直接报错：

```python
flag_gems.flagtune(include="addmm")
```

会得到类似错误：

```text
ValueError: Unsupported flagtune op(s): addmm. Supported ops: bmm, mm
```

注意：`flag_gems.flagtune(...)` 只负责设置 FlagTune 范围，不负责注册 FlagGems 算子。实际替换 PyTorch op 仍然需要：

```python
flag_gems.enable()
```

或者：

```python
with flag_gems.use_gems():
    ...
```

## 3. 本次修改了哪些文件

### 3.1 新增 API 状态管理

新增文件：

```text
src/flag_gems/runtime/flagtune.py
```

主要内容：

- `flagtune(include=None)`
- `flagtune_enabled(op_name)`
- `get_flagtune_include()`
- `register_flagtune_op(op_name, default=False, description="", replace=False)`
- `get_flagtune_registry()`
- `get_supported_flagtune_ops()`
- `get_default_flagtune_include()`
- `USE_FLAGTUNE_ENV = "USE_FLAGTUNE"`
- `FLAGTUNE_INCLUDE_ENV = "FLAGTUNE_INCLUDE"`

`flagtune.py` 现在有一个小型 registry：

```python
register_flagtune_op("mm", default=True, description="matrix multiplication")
register_flagtune_op("bmm", default=True, description="batched matrix multiplication")
```

registry 是支持列表和默认 include 的唯一来源：

- `get_supported_flagtune_ops()` 从 registry 返回所有已注册 op。
- `get_default_flagtune_include()` 从 registry 返回 `default=True` 的 op。
- `flag_gems.flagtune()` 等价于 `flag_gems.flagtune(include=None)`，会启用 registry 默认 op。
- `flag_gems.flagtune(include="...")` 只接受已注册 op，未注册 op 会报错。

文件开头不再维护 `SUPPORTED_FLAGTUNE_OPS = {"mm", "bmm"}` 或 `DEFAULT_FLAGTUNE_INCLUDE = {"mm", "bmm"}` 这种固定集合。为了兼容旧代码读取这两个名字，模块提供了 `__getattr__`，访问时会动态返回 registry 当前状态；新增算子后结果也会跟着变化。

`flagtune(...)` 做两件事：

1. 设置 `os.environ["USE_FLAGTUNE"] = "1"`。
2. 记录 include 范围，并同步写入 `FLAGTUNE_INCLUDE`。

判断某个算子是否启用 FlagTune 时，不再只看 `USE_FLAGTUNE`，而是：

```python
USE_FLAGTUNE == "1" and op_name in include_ops
```

### 3.2 顶层导出 API

修改：

```text
src/flag_gems/runtime/__init__.py
src/flag_gems/__init__.py
```

现在用户可以直接调用：

```python
flag_gems.flagtune(...)
flag_gems.register_flagtune_op(...)
flag_gems.get_supported_flagtune_ops()
flag_gems.get_default_flagtune_include()
flag_gems.get_flagtune_registry()
```

### 3.3 LibTuner 支持运行时切换 configs

修改：

```text
src/flag_gems/utils/libentry.py
```

这是本次改造的核心。

旧逻辑是在 import 时决定：

```python
configs = runtime.ops_get_configs(...) if USE_FLAGTUNE else runtime.get_tuned_config(...)
```

新逻辑变成：

1. 默认创建 tuner 时仍然使用默认 tuned config。
2. 给 `libtuner(...)` 增加可选参数：

```python
flagtune_op_name="bmm"
flagtune_expand_op_name="bmm"
flagtune_yaml_path=None
flagtune_pre_hook=None
```

3. `LibEntry.run(...)` 每次真正运行 kernel 前，会调用 `_maybe_apply_flagtune()`。
4. 如果当前 op 被 `flag_gems.flagtune(include=...)` 命中，`LibTuner` 会把 configs 从默认 tuned config 切换为 expanded configs。
5. 如果 include 改了，比如先 `include="bmm"`，后面又 `include="mm"`，`bmm` 会切回默认 configs。
6. configs 切换后会清理 `LibEntry.kernel_cache`，避免复用旧 constexpr 编译结果。

这让 API 即使在 `import flag_gems` 之后调用，也能在下一次 kernel 运行前生效。

## 4. mm 算子的实现和 FlagTune 覆盖范围

在当前测试机器上，设备是：

```text
cuda / nvidia / hopper
```

实际顶层 `flag_gems.mm` 指向 Hopper 专用实现：

```text
src/flag_gems/runtime/backend/_nvidia/hopper/ops/mm.py
```

原因是 `flag_gems.__init__` 中会执行：

```python
runtime.replace_customized_ops(globals())
```

Hopper backend 会导出：

```python
from .mm import mm, mm_out, router_gemm
```

所以顶层注册表里的：

```python
("mm", mm)
```

最终使用的是 Hopper 版本的 `mm`。

### 4.1 Hopper mm 内部分支

Hopper `mm(a, b)` 内部大致按场景分流：

1. `N == 1`
   - 走 `gemv_mm`
   - 对应 kernel/tuner：`gemv_kernel`
   - expanded YAML key：`gemv`

2. `streamk_scenario(...)`
   - 走 `streamk_mm`
   - 当前没有接入本次 `flag_gems.flagtune` API

3. `cluster_remote_mm_scenario(...)`
   - 走 cluster remote 路径
   - 当前没有接入本次 `flag_gems.flagtune` API

4. `M < 2048 and N < 2048 and K >= 4096`
   - 走 `splitk_mm`
   - 对应 kernel/tuner：`mm_kernel_splitk`
   - expanded YAML key：`mm_splitk`

5. 其他普通矩阵乘
   - 走 `general_mm`
   - TMA 可用时走 `mm_kernel_general_host_tma`
   - expanded YAML key：`mm_general_tma`

### 4.2 include="mm" 具体覆盖哪些 tuner

本次为 Hopper `mm` 接入了三个内部 tuner：

```python
flagtune_op_name="mm"
flagtune_expand_op_name="mm_general_tma"
```

```python
flagtune_op_name="mm"
flagtune_expand_op_name="gemv"
```

```python
flagtune_op_name="mm"
flagtune_expand_op_name="mm_splitk"
```

也就是说，用户只需要写：

```python
flag_gems.flagtune(include="mm")
```

内部这三个 `mm` 相关路径都会按需切换到 expanded configs。

这些 configs 来自：

```text
src/flag_gems/runtime/backend/_nvidia/hopper/mm_hopper_expand.yaml
```

已验证该 YAML 里能找到：

```text
mm_general_tma
gemv
mm_splitk
```

## 5. bmm 算子的实现和 YAML 移动

`bmm` 当前使用通用实现：

```text
src/flag_gems/ops/bmm.py
```

本次为 `bmm_kernel` 增加了：

```python
flagtune_op_name="bmm"
flagtune_expand_op_name="bmm"
```

当用户调用：

```python
flag_gems.flagtune(include="bmm")
```

`bmm_kernel` 会从默认 configs 切换为 expanded configs。

原来的 YAML：

```text
src/flag_gems/utils/configs/general_ops_expand_configs.yaml
```

已经移动到：

```text
src/flag_gems/runtime/backend/_nvidia/hopper/general_ops_hopper_configs.yaml
```

在 Hopper 上，`bmm` 的 expand YAML 会自动解析到新路径。

## 6. USE_FLAGTUNE 的生效范围现在是什么

现在 `USE_FLAGTUNE=1` 是总开关，不再单独决定某个算子是否进入 expanded configs。

实际判断是：

```python
USE_FLAGTUNE == "1" and op_name in flagtune_include
```

例如：

```python
flag_gems.flagtune(include="bmm")
```

会设置：

```text
USE_FLAGTUNE=1
FLAGTUNE_INCLUDE=bmm
```

此时：

```python
runtime.flagtune_enabled("bmm")  # True
runtime.flagtune_enabled("mm")   # False
```

因此 `bmm` 会启用 FlagTune，`mm` 不会。

如果之后调用：

```python
flag_gems.flagtune(include="mm")
```

则：

```python
runtime.flagtune_enabled("bmm")  # False
runtime.flagtune_enabled("mm")   # True
```

下一次运行时，`bmm` tuner 会切回默认 configs，`mm` tuner 会切到 expanded configs。

## 7. 已关闭旧的误触发路径

旧代码中这些算子会直接响应 `USE_FLAGTUNE=1`：

- `addmm`
- `baddbmm`
- `bmm`
- `mv`
- Hopper `mm`
- Hopper `w8a8_block_fp8_matmul`
- MThreads `mm`
- MThreads `w8a8_block_fp8_matmul`
- MThreads `sparse_attention`

本次已经把旧的 import-time `USE_FLAGTUNE` 判断移除或改造：

- `bmm` 接入新的 `flag_gems.flagtune(include=...)` API。
- Hopper/MThreads `mm` 接入新的 `flag_gems.flagtune(include=...)` API。
- `addmm`、`baddbmm`、`mv` 不再被 `USE_FLAGTUNE=1` 打开。
- `w8a8` 和 `sparse_attention` 不再被这个 API 误打开。

因此当前内置且已经接入实际 tuner configs 切换的范围是：

```text
mm, bmm
```

API 层本身可以通过 registry 扩展支持新名字，但新增名字还需要继续完成 kernel 侧 `flagtune_op_name`、expanded YAML 和 config 生成逻辑，才会真正触发 tuner configs 切换。

## 8. configloader.py 如何按 backend/arch 找 YAML

修改文件：

```text
src/flag_gems/runtime/configloader.py
```

新增核心逻辑：

```python
_iter_expand_config_candidates(op_name)
_get_expand_config_path(op_name)
```

查找上下文顺序：

1. 当前 arch 目录
   - 例如 Hopper：

```text
src/flag_gems/runtime/backend/_nvidia/hopper
```

2. 当前 vendor 目录
   - 例如 NVIDIA：

```text
src/flag_gems/runtime/backend/_nvidia
```

每个目录下按以下文件名顺序查找：

```text
{op_name}_{arch_name}_expand.yaml
{op_name}_{vendor_name}_expand.yaml
{op_name}_expand.yaml
general_ops_{arch_name}_configs.yaml
general_ops_{vendor_name}_configs.yaml
general_ops_configs.yaml
```

举例，在 Hopper 上：

```python
_get_expand_config_path("bmm")
```

会找到：

```text
src/flag_gems/runtime/backend/_nvidia/hopper/general_ops_hopper_configs.yaml
```

```python
_get_expand_config_path("mm")
```

会找到：

```text
src/flag_gems/runtime/backend/_nvidia/hopper/mm_hopper_expand.yaml
```

另外，`get_expand_config(op_name, yaml_path=...)` 现在优先使用显式传入的 `yaml_path`。这对 `mm_general_tma`、`gemv`、`mm_splitk` 很重要，因为它们复用 `mm_hopper_expand.yaml`，由 kernel 侧明确传入路径。

## 9. 验证脚本

新增目录：

```text
test_flagtune_api/
```

包含：

```text
test_flagtune_api/check_flagtune_api.py
test_flagtune_api/run_checks.sh
test_flagtune_api/README.md
```

运行方式：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate vllm0.17
./test_flagtune_api/run_checks.sh
```

实际验证输出：

```text
PASS: flagtune API include scope and config switching look correct
bmm configs: 12 -> 384
mm configs: 162 -> 480
bmm expand yaml: /workspace/FlagGems-dev/src/flag_gems/runtime/backend/_nvidia/hopper/general_ops_hopper_configs.yaml
```

这个脚本验证：

- `flag_gems.flagtune` 顶层 API 存在。
- `flag_gems.register_flagtune_op` 顶层 API 存在。
- registry 默认包含 `mm`、`bmm`，并支持动态注册测试 op。
- `include="bmm"` 时只启用 `bmm`。
- `include="mm"` 时只启用 `mm`。
- `bmm` 可以从默认 12 个 configs 切到 expanded 384 个 configs。
- Hopper `mm_general_tma` 可以从默认 162 个 configs 切到 expanded 480 个 configs。
- `addmm` 没有接入 flagtune。
- `include="addmm"` 会被拒绝。
- `bmm` 的 YAML 路径正确指向 `general_ops_hopper_configs.yaml`。

脚本默认不真正 launch kernel 做完整 autotune，只检查 API、include scope 和 tuner configs 切换。这样可以避免首次运行 benchmark 大量 configs 导致耗时过长。

## 10. 如果以后要给新算子接入 flagtune

假设新增公开算子 `foo`。

### 10.1 把公开算子名注册到 registry

现在不要修改 `SUPPORTED_FLAGTUNE_OPS` / `DEFAULT_FLAGTUNE_INCLUDE` 这类名字。它们只是兼容旧代码的动态只读视图，真实状态来自 registry。

新增算子应通过 registry 注册：

```python
from flag_gems.runtime import register_flagtune_op

register_flagtune_op(
    "foo",
    default=False,
    description="example operator",
)
```

参数含义：

- `op_name`：用户在 `flag_gems.flagtune(include=...)` 里看到的公开算子名。
- `default=False`：默认 `flag_gems.flagtune()` 不启用它，用户需要显式写 `include="foo"`。
- `default=True`：默认 `flag_gems.flagtune()` 会启用它。
- `description`：给 registry 查询和调试用，不影响运行逻辑。
- `replace=False`：默认禁止用不同配置重复注册同名 op，避免两个模块静默覆盖彼此。

如果只是实验性接入，建议先设为：

```python
register_flagtune_op("foo", default=False)
```

然后用户显式启用：

```python
flag_gems.flagtune(include="foo")
```

如果确认它应该成为默认 FlagTune 范围，再改成：

```python
register_flagtune_op("foo", default=True)
```

### 10.2 在 kernel 的 libtuner 上加映射

如果公开算子名和 YAML key 一样：

```python
@libentry()
@libtuner(
    configs=runtime.get_tuned_config("foo"),
    key=["M", "N"],
    strategy=["align32", "align32"],
    flagtune_op_name="foo",
    flagtune_expand_op_name="foo",
)
@triton.jit
def foo_kernel(...):
    ...
```

如果公开算子名和内部 YAML key 不一样，比如 `mm`：

```python
flagtune_op_name="mm"
flagtune_expand_op_name="mm_general_tma"
```

含义：

- `flagtune_op_name`：用户 API 看到的名字。
- `flagtune_expand_op_name`：YAML 中对应的 key。

### 10.3 在 common.py 里加默认策略和 key 顺序

```python
DEFAULT_STRATEGIES["foo"] = ["align32", "align32"]
OP_KEY_ORDERS["foo"] = ["M", "N"]
```

### 10.4 在 configloader.py 里加 configs 生成逻辑

在 `_build_configs_by_op` 中新增：

```python
if op_name == "foo":
    return [
        triton.Config(
            {
                "BLOCK_M": block_m,
                "BLOCK_N": block_n,
            },
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for block_m in ranges["BLOCK_M"]
        for block_n in ranges["BLOCK_N"]
        for s in ranges["s"]
        for w in ranges["w"]
    ]
```

并在 `_build_expand_registry()` 里注册：

```python
"foo": self._build_single_expand_spec(
    "foo",
    expand_yaml_path=self._get_expand_config_path("foo"),
),
```

### 10.5 新增 backend/arch YAML

例如 Hopper：

```text
src/flag_gems/runtime/backend/_nvidia/hopper/foo_hopper_expand.yaml
```

内容示例：

```yaml
foo:
  - gen: true
    param_map:
      META:
        BLOCK_M: block_m
        BLOCK_N: block_n
      num_stages: stages
      num_warps: warps
    block_m:
      - 16
      - 32
      - 64
    block_n:
      - 16
      - 32
      - 64
    stages:
      - 3
      - 4
    warps:
      - 4
      - 8
  - strategy:
      M: align32
      N: align32
```

之后用户即可：

```python
flag_gems.flagtune(include="foo")
flag_gems.enable()
```

## 11. 当前限制和注意事项

1. 当前内置 registry 默认只注册：

```text
mm, bmm
```

新增 op 可以通过 `register_flagtune_op(...)` 注册，但“注册”只表示 API 层允许 include 这个名字；如果没有在对应 kernel 的 `@libtuner(...)` 上加 `flagtune_op_name`，也没有配置 YAML 和 config 生成逻辑，那么不会发生实际 tuner configs 切换。

2. Hopper `mm` 的 `streamk_mm` 和 `cluster_remote_mm` 当前没有接入本次 flagtune API。

3. `flag_gems.flagtune(...)` 不会自动注册 FlagGems op。仍然需要 `flag_gems.enable()` 或 `flag_gems.use_gems()`。

4. 测试脚本只验证 tuner 配置切换，不做真实 kernel autotune。真实 workload 第一次命中某个未缓存 shape 时，才会执行实际 autotune。

5. 如果用户直接在环境里手动设置：

```bash
USE_FLAGTUNE=1
```

但不设置 `FLAGTUNE_INCLUDE`，默认 include 是：

```text
mm, bmm
```

这个默认值来自 registry 中 `default=True` 的 op。当前内置默认是 `mm`、`bmm`，因此保持了一个合理默认行为，同时避免其他算子被误打开。

## 12. 总结

本次改造完成后，FlagTune 的启用方式从“全局环境变量粗粒度打开”变成了“Python API 精确指定算子范围”：

```python
flag_gems.flagtune(include="mm, bmm")
```

核心效果：

- `USE_FLAGTUNE=1` 不再让所有历史分支算子误启用。
- FlagTune 支持范围由 registry 管理，新增 op 可以通过 `register_flagtune_op(...)` 扩展。
- `mm` 和 `bmm` 可以分别、独立启用 FlagTune。
- `LibTuner` 支持运行时按 include 切换 configs。
- `bmm` 的 expanded YAML 已迁移到 Hopper backend 目录。
- `configloader.py` 已支持按 backend/arch 自动查找 YAML。
- 已在 `vllm0.17` 环境、NVIDIA Hopper 设备上验证 API 和路径解析符合预期。
