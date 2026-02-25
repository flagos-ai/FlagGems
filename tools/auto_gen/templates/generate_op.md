# FlagGems 算子生成任务

你需要为 FlagGems 项目实现一个新的 Triton 算子。

## 任务信息

- **算子名称**: {{OPERATOR}}
- **GPU ID**: {{GPU_ID}}
- **工作目录**: {{WORK_DIR}} (这是一个 git worktree)
- **Python 路径**: {{PYTHON_PATH}}

## 运行环境说明

**重要**：本项目**不需要** `pip install`。`pytest.ini` 已配置 `pythonpath = src`，因此在工作目录（worktree 根目录）下运行 pytest 时，会自动将 `<工作目录>/src` 加入 `sys.path`，从而正确导入当前 worktree 的 `flag_gems` 代码。

- **禁止**运行 `pip install -e .` 或任何形式的 `pip install flag-gems`
- **所有命令**必须在工作目录 `{{WORK_DIR}}` 下执行
- **GPU 指定**：所有涉及 GPU 的命令（pytest、python -c 中 import torch 等）必须加上 `CUDA_VISIBLE_DEVICES={{GPU_ID}}` 前缀
- 运行测试时使用：`CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -m pytest ...`

## FlagGems 项目结构

```
src/flag_gems/
├── __init__.py              # _FULL_CONFIG 注册表
├── ops/
│   ├── __init__.py          # import 所有算子
│   └── <operator>.py        # 各算子实现
├── utils/
│   ├── pointwise_dynamic.py # pointwise 装饰器
│   └── triton_lang_extension.py  # tl_extra_shim 跨后端兼容层
└── runtime/
    └── backend/_nvidia/
        └── tune_configs.yaml  # autotuning 配置
tests/
├── test_unary_pointwise_ops.py   # 一元 pointwise 测试
├── test_binary_pointwise_ops.py  # 二元 pointwise 测试
├── test_reduction_ops.py         # reduction 测试
├── test_norm_ops.py              # norm 测试
├── test_blas_ops.py              # BLAS 测试
├── test_special_ops.py           # 特殊算子测试
├── accuracy_utils.py             # 共享工具：POINTWISE_SHAPES, FLOAT_DTYPES, gems_assert_close 等
└── conftest.py
benchmark/
├── test_unary_pointwise_perf.py  # 一元 pointwise 性能
├── test_binary_pointwise_perf.py # 二元 pointwise 性能
├── test_reduction_perf.py
└── ...
pytest.ini                        # 配置 pythonpath = src，自动导入 worktree 的代码
```

## 执行步骤

请严格按照以下步骤执行：

### Step 1: 了解算子语义

运行以下命令了解 `{{OPERATOR}}` 的 PyTorch 接口：

```bash
CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -c "
import torch
for module_path in ['torch.{{OPERATOR}}', 'torch.nn.functional.{{OPERATOR}}']:
    try:
        fn = eval(module_path)
        help(fn)
        break
    except:
        pass
"
```

同时查阅 `torch.ops.aten` 中的 schema：

```bash
CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -c "
import torch
for op in dir(torch.ops.aten):
    if '{{OPERATOR}}' in op.lower():
        fn = getattr(torch.ops.aten, op)
        if hasattr(fn, 'default'):
            print(f'{op}: {fn.default._schema}')
"
```

### Step 2: 确定实现模式 & 阅读参考代码

FlagGems 有三种主要实现模式，根据算子类型选择：

**模式 A: pointwise_dynamic（逐元素操作）**
适用于：一元/二元逐元素操作（如 abs, relu, add, mul）
参考文件：`src/flag_gems/ops/abs.py`, `src/flag_gems/ops/ceil.py`, `src/flag_gems/ops/add.py`

```python
import logging
import triton
import triton.language as tl
from flag_gems.utils import pointwise_dynamic

logger = logging.getLogger(__name__)

@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")])
@triton.jit
def op_func(x):
    return ...  # Triton 标量逻辑

def op(A):
    logger.debug("GEMS OP")
    return op_func(A)

def op_(A):  # in-place 版本（如果需要）
    logger.debug("GEMS OP_")
    op_func(A, out0=A)
    return A
```

promotion_methods 常见值：
- `(0, "DEFAULT")` — 默认类型提升
- `(0, "INT_TO_FLOAT")` — 整数输入提升为浮点（三角函数等）
- `(0, "COMPLEX_TO_FLOAT")` — 复数输入返回浮点（abs 等）
- `(0, 1, "DEFAULT")` — 二元操作默认提升
- `(0, 1, "ALWAYS_BOOL")` — 输出始终为 bool（比较操作）

**模式 B: 手写 Triton kernel（reduction/scan/index 等）**
适用于：涉及跨元素计算的操作
参考文件：`src/flag_gems/ops/sum.py`, `src/flag_gems/ops/softmax.py`

**模式 C: 多 kernel + autograd.Function（需要反向传播）**
适用于：有前向+反向的操作
参考文件：`src/flag_gems/ops/layernorm.py`, `src/flag_gems/ops/rms_norm.py`

请阅读 2-3 个与目标算子**同类型**的已有实现作为参考。

**重要：跨后端兼容性**
- **禁止**直接调用 `tl.extra.cuda.libdevice`，这在非 NVIDIA 后端上会崩溃
- **必须**使用 `tl_extra_shim` 提供的跨后端兼容函数（如 `tl_extra_shim.nearbyint`, `tl_extra_shim.pow` 等）
- 参考 `src/flag_gems/ops/isnan.py` 和 `src/flag_gems/ops/ceil.py` 的写法
- 如果需要的数学函数在 `tl_extra_shim` 中不存在，使用 Triton 内置的 `tl.math` 或 `tl.` 函数

### Step 3: 实现算子代码

在 `src/flag_gems/ops/{{OPERATOR}}.py` 创建算子实现。

**要求：**
- 遵循已有算子的代码风格
- 必须有 `import logging` 和 `logger = logging.getLogger(__name__)`
- 函数名遵循已有命名规范
- 如果是 pointwise 操作，优先使用 `pointwise_dynamic`
- 对 float16/bfloat16 输入，做 `.to(tl.float32)` 计算后 `.to(x.dtype)` 转回（参考 ceil.py 的写法）

### Step 4: 注册算子

1. **在 `src/flag_gems/ops/__init__.py` 中添加 import 和 `__all__` 条目：**
   按字母顺序插入。注意字母顺序是严格的，例如 `sigmoid` < `signbit` < `silu` < `sin`。

2. **在 `src/flag_gems/__init__.py` 的 `_FULL_CONFIG` 中添加注册项：**
   按字母顺序插入，格式为：
   ```python
   ("aten_op_name", function_name),
   ```
   aten op name 需要与 Step 1 中查到的 schema 名一致。

### Step 5: 编写 accuracy 测试

**在 FlagGems 标准测试文件中添加测试用例**，不要写到 `/tmp` 或其他地方。

根据算子类型，选择对应的测试文件：
- 一元 pointwise → `tests/test_unary_pointwise_ops.py`
- 二元 pointwise → `tests/test_binary_pointwise_ops.py`
- reduction → `tests/test_reduction_ops.py`
- norm → `tests/test_norm_ops.py`
- 其他 → `tests/test_special_ops.py`

**先阅读对应测试文件**，了解现有测试的模式和使用的工具函数（如 `POINTWISE_SHAPES`, `FLOAT_DTYPES`, `to_reference`, `gems_assert_close`, `gems_assert_equal` 等），然后在文件末尾追加新的测试函数。

**一元 pointwise 测试模板**（参考 `test_accuracy_ceil` 和 `test_accuracy_ceil_`）：

```python
@pytest.mark.{{OPERATOR}}
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_{{OPERATOR}}(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp)

    ref_out = torch.{{OPERATOR}}(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.{{OPERATOR}}(inp)

    gems_assert_close(res_out, ref_out, dtype)


@pytest.mark.inplace
@pytest.mark.{{OPERATOR}}_
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_{{OPERATOR}}_(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = to_reference(inp.clone())

    ref_out = ref_inp.{{OPERATOR}}_()
    with flag_gems.use_gems():
        res_out = inp.{{OPERATOR}}_()

    gems_assert_close(res_out, ref_out, dtype)
```

**二元 pointwise 测试模板**（参考 `test_accuracy_add` 等）：

```python
@pytest.mark.{{OPERATOR}}
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_{{OPERATOR}}(shape, dtype):
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp1 = to_reference(inp1)
    ref_inp2 = to_reference(inp2)

    ref_out = torch.{{OPERATOR}}(ref_inp1, ref_inp2)
    with flag_gems.use_gems():
        res_out = torch.{{OPERATOR}}(inp1, inp2)

    gems_assert_close(res_out, ref_out, dtype)
```

**注意**：上面只是模板，你需要根据算子的实际接口和语义调整（输入数据生成方式、断言方式等）。对于精确运算（如 floor, round），应使用 `gems_assert_equal` 而非 `gems_assert_close`。

### Step 6: 运行 accuracy 测试

**必须在工作目录 `{{WORK_DIR}}` 下运行**（pytest.ini 的 `pythonpath = src` 会自动让 Python 导入当前 worktree 的 `flag_gems`）。

使用标准 pytest 命令，用 `-m` 指定算子 mark，**必须加 `--log-cli-level=DEBUG`** 以验证你的算子确实被调用了（你实现的算子中有 `logger.debug("GEMS XXX")` 日志）：

```bash
cd {{WORK_DIR}}
CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -m pytest tests/<test_file>.py -m {{OPERATOR}} -vs --log-cli-level=DEBUG 2>&1
```

例如：
```bash
# 一元 pointwise
CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -m pytest tests/test_unary_pointwise_ops.py -m {{OPERATOR}} -vs --log-cli-level=DEBUG

# 二元 pointwise
CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -m pytest tests/test_binary_pointwise_ops.py -m {{OPERATOR}} -vs --log-cli-level=DEBUG
```

**验证算子被调用**：在测试输出中检查是否出现了类似 `GEMS {{OPERATOR}}` 的 DEBUG 日志。如果没有出现，说明你的算子没有被正确注册或调用，需要检查 Step 4 的注册步骤。

**验证导入正确性**：如果需要确认导入的是 worktree 代码，可以运行：
```bash
CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -c "import sys; sys.path.insert(0, 'src'); import flag_gems; print(flag_gems.__file__)"
```
输出应包含 `{{WORK_DIR}}/src/flag_gems`。

**如果测试失败：**
1. 分析失败原因（类型不匹配、精度问题、逻辑错误）
2. 如果 DEBUG 日志中没有 `GEMS` 输出，检查算子注册是否正确
3. 修复 `src/flag_gems/ops/{{OPERATOR}}.py` 或测试代码
4. 重新运行测试，直到所有测试通过

### Step 7: 编写 benchmark 并运行

**在 FlagGems 标准 benchmark 文件中添加 benchmark 条目**。

根据算子类型，选择对应的 benchmark 文件：
- 一元 pointwise → `benchmark/test_unary_pointwise_perf.py`
- 二元 pointwise → `benchmark/test_binary_pointwise_perf.py`
- reduction → `benchmark/test_reduction_perf.py`
- 其他 → `benchmark/test_special_perf.py`

**先阅读对应 benchmark 文件**，了解 `forward_operations` 列表的格式，然后将新算子追加到合适位置。

**一元 pointwise benchmark 模板**（添加到 `forward_operations` 列表中）：
```python
("{{OPERATOR}}", torch.{{OPERATOR}}, FLOAT_DTYPES),
```

**二元 pointwise benchmark 模板**（添加到 `forward_operations` 列表中）：
```python
("{{OPERATOR}}", torch.{{OPERATOR}}, FLOAT_DTYPES),
```

运行 benchmark（同样必须在工作目录下）：
```bash
cd {{WORK_DIR}}
CUDA_VISIBLE_DEVICES={{GPU_ID}} {{PYTHON_PATH}} -m pytest benchmark/<benchmark_file>.py -m {{OPERATOR}} -vs 2>&1
```

**解析 benchmark 输出**：benchmark 输出格式为：
```
Operator: {{OPERATOR}}  Performance Test (dtype=torch.float16, mode=kernel,level=comprehensive)
SUCCESS    <torch_latency_ms>    <gems_latency_ms>    <speedup>    <gems_gbps>    [<shape>]
SUCCESS    ...
Operator: {{OPERATOR}}  Performance Test (dtype=torch.float32, ...)
SUCCESS    ...
```

请从输出中提取每一行 `SUCCESS` 的数据，按 dtype 分组记录到最终 JSON 中。

### Step 8: 输出结果

在所有步骤完成后，输出以下 JSON 格式的最终结果（用 ```json 代码块包裹）：

```json
{
  "operator": "{{OPERATOR}}",
  "status": "success 或 failed",
  "accuracy_passed": true/false,
  "files_created": [
    "src/flag_gems/ops/{{OPERATOR}}.py"
  ],
  "files_modified": [
    "src/flag_gems/ops/__init__.py",
    "src/flag_gems/__init__.py",
    "tests/test_xxx_ops.py",
    "benchmark/test_xxx_perf.py"
  ],
  "aten_ops_registered": ["aten_op_name1", "aten_op_name2"],
  "implementation_mode": "pointwise_dynamic 或 manual_kernel 或 autograd_function",
  "test_results": {
    "total": 12,
    "passed": 12,
    "failed": 0,
    "test_command": "python -m pytest tests/test_xxx_ops.py -m {{OPERATOR}} -vs"
  },
  "benchmark_results": {
    "benchmark_command": "python -m pytest benchmark/test_xxx_perf.py -m {{OPERATOR}} -vs",
    "data": [
      {
        "dtype": "torch.float16",
        "shape": "[1024, 1024]",
        "torch_latency_ms": 0.056,
        "gems_latency_ms": 0.057,
        "speedup": 0.987
      }
    ]
  },
  "error_message": "null 或错误描述",
  "notes": "实现说明或特殊处理"
}
```

**注意**：`benchmark_results.data` 数组中应包含 benchmark 输出中**每一行 SUCCESS** 的数据。如果 benchmark 运行失败或没有输出，`data` 可以为空数组 `[]`。

## 重要约束

1. **正确性优先**：必须通过 accuracy 测试
2. **代码风格**：严格遵循 FlagGems 已有代码风格
3. **标准测试**：测试和 benchmark 必须写入 FlagGems 标准文件中，使用标准 pytest 命令运行
4. **跨后端兼容**：禁止直接调用 `tl.extra.cuda.libdevice`，必须使用 `tl_extra_shim` 或 Triton 内置函数
5. **字母顺序**：所有注册（import、`__all__`、`_FULL_CONFIG`）必须严格按字母顺序插入
6. **最终代码保留**：无论成功失败，都保留修改的代码在 worktree 中
7. **不要删除或修改已有算子的代码和测试**
8. **JSON 结果必须输出**：即使失败也要输出 JSON，标明 status 为 failed
9. **禁止 pip install**：不要运行 `pip install -e .` 或任何安装命令，pytest.ini 已处理导入
10. **工作目录**：所有命令必须在 `{{WORK_DIR}}` 下执行，不要 cd 到其他目录
11. **禁止写临时文件**：不要将测试或代码写到 `/tmp` 或其他临时目录
