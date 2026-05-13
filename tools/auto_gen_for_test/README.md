# auto_gen — 批量自动生成 FlagGems 算子

通过 orchestrator 调度多个 Claude Code (CC) 实例，并行为 FlagGems 补充缺失算子。每个 CC 独立完成：算子实现 → 注册 → 测试 → benchmark。

## 前置条件

1. **Claude Code CLI** 已安装且可用（`claude --version`）
2. **conda 环境** 中已有 PyTorch + Triton，但 **不能** `pip install flag-gems`（依赖 `pytest.ini` 的 `pythonpath = src` 导入）
3. **GPU** 可用（脚本自动检测或在 config 中指定）

## 快速开始

```bash
cd tools/auto_gen

# 1. 编辑算子列表（每行一个，支持 aten::xxx 格式）
vim ops_list.txt

# 2. 修改配置（python_path、flaggems_dir 等）
vim config.yaml

# 3. 运行
python orchestrator.py ops_list.txt
```

## 文件结构

```
tools/auto_gen/
├── orchestrator.py          # 主调度脚本
├── device_manager.py        # GPU 分配（lock 文件互斥）
├── config.yaml              # 配置文件
├── ops_list.txt             # 待生成的算子列表
├── templates/
│   └── generate_op.md       # CC prompt 模板
├── .env.example             # 环境变量模板
├── .env                     # 实际环境变量（不提交）
└── results/
    ├── summary.json          # 运行汇总
    └── logs/
        └── <op>.jsonl        # CC 实时流式日志（stream-json）
```

## 用法

```bash
python orchestrator.py [ops_list] [-c config.yaml] [-v]
```

| 参数 | 说明 |
|------|------|
| `ops_list` | 算子列表文件路径（默认 `ops_list.txt`） |
| `-c, --config` | 配置文件路径（默认 `config.yaml`） |
| `-v, --verbose` | 开启 DEBUG 日志 |

## 算子列表格式

每行一个算子，支持以下格式：

```
aten::floor
aten::log1p
aten::signbit
round
fmod
```

`aten::` 前缀和 `.Tensor` 等重载后缀会自动去除。`#` 开头的行为注释。

## 配置说明

```yaml
flaggems_dir: /path/to/FlagGems        # FlagGems 仓库根目录
python_path: /path/to/python            # Python 解释器路径
claude_bin: claude                       # CC 可执行文件
max_retries: 3                           # 单算子最大重试次数
budget_per_op: 10.0                      # 单算子单次 budget 上限 (USD)
template: templates/generate_op.md       # prompt 模板路径
results_dir: results                     # 结果输出目录

device:
  lock_dir: /tmp/auto_gen_gpu_locks      # GPU 锁文件目录
  gpu_ids: null                          # null=自动检测, 或 [0,1,2,3]

poll_interval: 10                        # 进程轮询间隔 (秒)
```

## 工作流程

1. **读取算子列表**，放入任务队列
2. **获取空闲 GPU**（通过 lock 文件互斥）
3. **创建 git worktree**（`<flaggems_dir>/.worktrees/gen-<op>`），分支 `auto-gen/<op>`
4. **启动 CC 进程**，传入渲染后的 prompt 模板
5. CC 在 worktree 中独立完成：实现 → 注册 → 写测试 → 跑 pytest → 写 benchmark → 输出 JSON
6. **解析结果**：成功则记录，失败则重试（不超过 max_retries）
7. 释放 GPU，处理下一个算子
8. 全部完成后输出 `results/summary.json`

## 结果查看

```bash
# 实时查看某个 CC 的工作过程
tail -f results/logs/floor.jsonl

# 查看汇总
cat results/summary.json | python -m json.tool

# 查看生成的代码
ls /path/to/FlagGems/.worktrees/gen-floor/src/flag_gems/ops/floor.py

# 手动验证测试
cd /path/to/FlagGems/.worktrees/gen-floor
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_unary_pointwise_ops.py -m floor -vs --log-cli-level=DEBUG
```

## 注意事项

- 运行环境中 **不能** 有 `pip install -e .` 的 flag-gems，否则会覆盖 worktree 的代码
- 每个 worktree 独立运行，CC 之间不会互相干扰
- 支持 Ctrl+C 优雅退出，立即终止所有运行中的 CC 进程
- 再次 Ctrl+C 强制退出主进程
