# auto_gen backport TODO

从 `auto_fix_issues` 迁移回 `auto_gen`（及 `auto_gen_for_test`）的改进项。

> 注意：`auto_gen_for_test` 当前与 `auto_gen` 完全相同（零 diff），应考虑合并为一份代码或在迁移时同步更新。

---

## P0 — 必须迁移（基础设施 bug fix / 鲁棒性）

### 1. 原子 GPU 锁
- **文件**: `device_manager.py` — `acquire()`
- **现状**: 用 `open(lock_path, "w")` 创建锁文件，`os.path.exists()` 和 `open()` 之间存在 TOCTOU 竞态，两个 orchestrator 可能同时拿到同一块 GPU
- **目标**: 改用 `os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)` 原子创建；清理 stale lock 时加 `try/except FileNotFoundError`

### 2. 优雅进程终止
- **文件**: `orchestrator.py` — `_kill_cc_process()`
- **现状**: 直接 `SIGKILL`，等 5s
- **目标**: `SIGTERM` → 等 10s → `SIGKILL` → 等 5s，给 CC 保存状态的机会

### 3. API 断连自动恢复
- **文件**: `orchestrator.py` — 新增 `resume_cc()`、`is_api_stream_error()`、`extract_session_id()`
- **现状**: API 连接断开（ECONNRESET、Unexpected EOF）后整个任务从头重来
- **目标**: 检测流式错误 → 提取 session_id → `claude --resume <session_id>` 接续会话；识别 403/401 等认证错误直接跳过不重试
- **配置**: 新增 `max_stream_retries`（默认 3）

### 4. Summary 跨次运行保留
- **文件**: `orchestrator.py` — `Summary.__init__()`
- **现状**: 每次运行覆盖 `summary.json`，重跑失败项会丢失已成功的记录
- **目标**: 加载已有 `summary.json`，只覆盖本次运行的算子，保留历史结果

### 5. 可中断 sleep
- **文件**: `orchestrator.py` — 主循环 sleep
- **现状**: `time.sleep(poll_interval)` 阻塞，Ctrl+C 最多等 10s 才响应
- **目标**: 改为 1s 步进循环 + `shutdown_requested` 检查

### 6. 文件句柄 double-close 防护
- **文件**: `orchestrator.py` — `_kill_cc_process()`、`parse_cc_result()`
- **现状**: 两处都无条件 `close()`，可能 double-close
- **目标**: 加 `if not proc._stdout_file.closed` 检查

---

## P1 — 建议迁移（需少量适配）

### 7. JSON 解析增强
- **现状**: 用简单正则匹配 `` ```json...``` `` 或含 `"operator"+"status"` 的扁平对象，嵌套 JSON 会失败
- **目标**: 迁移 `_extract_json_object()` 函数（brace-counting + 字符串转义感知）

### 8. `needs_review` 状态
- **现状**: CC 输出不可解析时只有 success/failed
- **目标**: CC 正常退出（returncode=0）且 worktree 有改动但输出格式不对时标记为 `needs_review`，避免丢弃可能正确的代码

### 9. `base_branch` 可配置
- **现状**: `create_worktree()` 硬编码 `master`
- **目标**: 从 `config.yaml` 读取 `base_branch`，支持从其他分支生成算子

### 10. 后处理：格式化 + Co-Author 清理
- **现状**: CC 生成的代码风格不一致，commit message 带 `Co-Authored-By`
- **目标**: CC 完成后自动跑 `black`/`isort`/`flake8`；去掉 commit message 中的 `Co-Authored-By` 行

### 11. Timeline 生成
- **现状**: 只有原始 JSONL 日志，排查问题需手动解析
- **目标**: 迁移 `generate_timeline()` 函数，生成 `.timeline.txt` 人类可读日志

### 12. JSONL 读取编码防护
- **现状**: 读取 JSONL 时未指定错误处理，非 UTF-8 字节会导致崩溃
- **目标**: `open(..., errors="replace")`

---

## 不迁移

| 功能 | 原因 |
|------|------|
| YAML issue 输入格式 | 算子生成用 `ops_list.txt` 更自然 |
| `scope: repo` 支持 | issue 专属概念 |
| `fetch_issues.sh` / `fetch_test_results.sh` | 内部 issue 系统集成 |
| `create_prs.py` | 可作为独立工具，不必耦合 |
| `check_pytest_marks.py` | 可独立使用 |
| `GEMS_VENDOR` 环境变量 | 视算子生成是否需要多 vendor 再定 |

---

## 附注

- `auto_gen_for_test` 与 `auto_gen` 代码完全相同，迁移时需同步更新或考虑抽取公共模块
- 参考实现均在 `tools/auto_fix_issues/` 对应文件中
