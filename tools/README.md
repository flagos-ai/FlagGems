# FlagGems 算子信息提取工具使用说明

## 工具概述

`extract_operators.py` 是一个从FlagGems源码自动提取算子信息的工具，可以：
- **双维度统计**：注册维度（_FULL_CONFIG）+ 测试覆盖维度（pytest marks）
- 从源码中提取所有算子信息
- 分析测试文件获取测试详情
- 检测后端支持情况（skip标记）
- 生成多种格式的输出

## 位置

```
FlagGems/tools/extract_operators.py
```

## 快速使用

### 1. 双维度统计（推荐）

```bash
# 在项目根目录执行，输出双维度统计报告
python tools/extract_operators.py --dual-stats

# 保存统计结果到JSON文件
python tools/extract_operators.py --dual-stats --output stats.json
```

**输出示例**：
```
======================================================================
FlagGems 算子统计报告（双维度）
======================================================================

【维度1: 注册维度】从 _FULL_CONFIG 提取
  - 注册的唯一算子数: 293个

【维度2: 测试覆盖维度】从 pytest marks 提取
  - 有测试覆盖的算子数: 310个

【差异分析】
  - 两个维度都有的算子: 246个
  - 仅在注册中有（无测试）: 47个
  - 仅在测试中有（未注册）: 64个
```

### 2. 生成算子详情表格

```bash
# 生成Markdown表格（默认）
python tools/extract_operators.py --output operators.md

# 生成CSV文件
python tools/extract_operators.py --format csv --output operators.csv

# 生成JSON文件
python tools/extract_operators.py --format json --output operators.json

# 显示详细信息
python tools/extract_operators.py --output operators.md --verbose
```

### 3. 使用便捷脚本

```bash
# 双维度统计
./tools/extract.sh --dual-stats
./tools/extract.sh --dual-stats json  # 保存JSON

# 生成表格
./tools/extract.sh          # 默认markdown
./tools/extract.sh csv      # CSV格式
```

## 命令行参数

| 参数 | 缩写 | 说明 | 默认值 |
|------|------|------|--------|
| `--path` | - | FlagGems项目路径 | 当前目录 |
| `--format` | - | 输出格式（markdown/csv/json） | markdown |
| `--output` | - | 输出文件路径 | 双维度模式可选 |
| `--dual-stats` | - | 双维度统计模式 | 否 |
| `--verbose` | - | 显示详细处理信息 | 否 |
| `--help` | -h | 显示帮助信息 | - |

## 输出格式说明

### 1. Markdown格式

**特点**：
- 包含完整的文档结构
- 按算子类别分组
- 包含后端支持统计

**输出内容**：
- 统计概览
- 后端支持统计表
- 算子详情表格（按类别分组）

**示例**：
```markdown
### 数学运算

| 算子名称 | 算子类别 | 算子所在文件 | ... | 支持的后端 |
|----------|----------|--------------|-----|-----------|
| abs | 数学运算 | abs.py | ... | 全部后端 |
| cos | 数学运算 | cos.py | ... | 全部后端（跳过HYGON） |
```

---

### 2. CSV格式

**特点**：
- 表格形式，易于导入Excel
- 每行一个测试
- 包含所有字段

**字段列表**：
1. 算子名称
2. 算子类别
3. 算子所在文件
4. 算子测例
5. 数据类型
6. 形状
7. 测试文件
8. 正确性测试命令
9. 性能测试命令
10. 支持的后端
11. 是否skip

**示例**：
```csv
算子名称,算子类别,算子所在文件,...
abs,数学运算,abs.py,test_accuracy_abs,FLOAT_DTYPES,POINTWISE_SHAPES,...
```

---

### 3. JSON格式

**特点**：
- 结构化数据
- 包含元数据
- 易于程序处理

**结构**：
```json
{
  "metadata": {
    "generated_at": "2026-04-08T19:16:58",
    "project_path": ".",
    "unique_operators": 291,
    "test_markers": 264,
    "total_tests": 392,
    "backend_skip_stats": {
      "HYGON": 4,
      "KUNLUNXIN": 2,
      ...
    }
  },
  "operators": {
    "abs": {
      "category": "数学运算",
      "source_file": "abs.py",
      "tests": [...]
    }
  }
}
```

## 使用场景

### 场景1：定期生成算子信息表

```bash
# 在CI中定期运行
0 0 * * * cd /path/to/FlagGems && python tools/extract_operators.py --output docs/operators.md
```

### 场景2：分析后端支持情况

```bash
# 生成JSON，然后用jq分析
python tools/extract_operators.py --format json --output /tmp/ops.json
cat /tmp/ops.json | jq '.metadata.backend_skip_stats'
```

### 场景3：导出到Excel分析

```bash
# 生成CSV
python tools/extract_operators.py --format csv --output operators.csv

# 在Excel中打开operators.csv
```

### 场景4：对比不同版本

```bash
# 生成v1版本
python tools/extract_operators.py --format json --output v1.json

# 更新代码后
git pull

# 生成v2版本
python tools/extract_operators.py --format json --output v2.json

# 对比
diff v1.json v2.json
```

## 输出内容说明

### 支持的后端

**"全部后端"**：
- 支持所有13个后端
- 无任何skip标记

**"全部后端（跳过XXX）"**：
- 支持大部分后端
- 在特定后端上有已知问题

### 后端列表

支持的后端包括：
- NVIDIA
- CAMBRICON（寒武纪）
- METAX（壁仞）
- ILUVATAR（天数智芯）
- MTHREADS（摩尔线程）
- KUNLUNXIN（昆仑芯）
- HYGON（海光）
- AMD
- AIPU
- ASCEND（华为昇腾）
- TSINGMICRO（清微智能）
- SUNRISE（日出东方）
- ENFLAME（燧原）

## 定制和扩展

### 修改算子类别

编辑脚本中的 `CATEGORY_MAP` 字典：

```python
CATEGORY_MAP = {
    'your_op': '新类别',
    ...
}
```

### 添加新的输出格式

在脚本中添加新的生成方法：

```python
def generate_xxx(self, operators, vendor_skip_stats):
    """生成XXX格式"""
    # 实现你的逻辑
    return content
```

### 修改后端列表

编辑 `ALL_BACKENDS` 列表：

```python
ALL_BACKENDS = [
    'NVIDIA', 'CAMBRICON', ...
]
```

## 故障排查

### 问题1：找不到项目路径

**错误**：`FileNotFoundError: FlagGems项目路径不存在`

**解决**：
```bash
# 确保在正确目录
cd /path/to/FlagGems
python tools/extract_operators.py --output operators.md

# 或使用--path参数
python tools/extract_operators.py --path /path/to/FlagGems --output operators.md
```

### 问题2：生成的算子数不对

**验证**：
```bash
# 检查_FULL_CONFIG条目数
grep -c '^\s*("' src/flag_gems/__init__.py

# 检查唯一算子名称
grep '^\s*("' src/flag_gems/__init__.py | sed 's/.*("\([^"]*\)".*/\1/' | cut -d'.' -f1 | sort -u | wc -l
```

### 问题3：输出文件太大

**解决**：
- 使用CSV格式（更小）
- 使用JSON格式（gzip压缩）

```bash
python tools/extract_operators.py --format csv --output operators.csv
gzip operators.csv
```

## 版本历史

- **v1.0** (2026-04-08)
  - 初始版本
  - 支持markdown/csv/json三种格式
  - 自动检测后端skip情况

## 维护者

FlagGems Team

## 许可证

Apache License 2.0