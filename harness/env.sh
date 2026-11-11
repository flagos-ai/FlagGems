#!/usr/bin/env bash
# FlagGems Kunlunxin harness 环境装载
# 用法: source harness/env.sh
#
# 本机 conda 环境: /ssd2/v_liuyu60/fg_env/envs/py310
# docker 内:      /workspace/envs/py310
# 可用 FLAGGEMS_CONDA_ENV 覆盖。

CONDA_ENV="${FLAGGEMS_CONDA_ENV:-}"
if [ -z "$CONDA_ENV" ]; then
  # 优先容器内路径，其次宿主机路径（两者是同一 bind mount）
  for _candidate in /workspace/envs/py310 /ssd2/v_liuyu60/fg_env/envs/py310; do
    if [ -d "$_candidate" ]; then
      CONDA_ENV="$_candidate"
      break
    fi
  done
fi

if [ -n "$CONDA_ENV" ] && [ -d "$CONDA_ENV" ]; then
  export PATH="$CONDA_ENV/bin:$PATH"
  export CONDA_PREFIX="$CONDA_ENV"
else
  echo "[env.sh] 未找到环境: ${CONDA_ENV:-无} (可用 FLAGGEMS_CONDA_ENV 覆盖)" >&2
fi

# 注意: CUDA_VISIBLE_DEVICES / XPU_EVENT_KL3_ENABLE / PYTHONWARNINGS / FLAGGEMS_CACHE_DIR
#       由主 agent 选卡后按需 export，不在此固化（fg_env/AGENTS.md）。
