#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch / vLLM Trace 算子耗时分析脚本

功能:
    读取 `.pt.trace.json.gz` 文件，按 (算子名, Input Dims, Input Type) 
    聚合统计 CPU 和 GPU 的耗时，并生成对齐美观的文本报告。

前置条件 (Prerequisites):
    1. 服务端 vLLM 必须开启 Profiler 采集。
       - 确保在 config.env 中配置了 `ENABLE_PROFILER="true"` 及相关的 PROFILER 参数。
       - 通过 start.sh 启动服务，并通过 perf.sh 发起压测请求以触发采集。
    2. 依赖库：强烈建议安装 `ijson` (pip install ijson) 以流式解析大文件，防止 OOM。

使用方法 (Usage):
    # 默认按 GPU 耗时降序排序
    python3 getshape.py --trace /path/to/rank0.xxx.pt.trace.json.gz

    # 指定输出文件名
    python3 getshape.py --trace /path/to/trace.json.gz --output my_report.txt

    # 切换为按 CPU 耗时排序
    python3 getshape.py --trace /path/to/trace.json.gz --sort cpu

参数说明 (Arguments):
    --trace  : 必填，指定 .pt.trace.json.gz 文件的路径。
    --output : 选填，指定输出的报告文件名 (默认: shape_time_grouped_report.txt)。
    --sort   : 选填，选择排序主键：'cuda' 或 'cpu' (默认: cuda)。
"""

import argparse
import gzip
import json
import sys
from collections import defaultdict

try:
    import ijson
    HAVE_IJSON = True
except ImportError:
    HAVE_IJSON = False


DEFAULT_TRACE = "./rank0.1794462962407205869.pt.trace.json.gz"
DEFAULT_OUTPUT = "shape_time_grouped_report.txt"

_TOP_LEVEL_CACHE = {}

def detect_top_level(path):
    if path in _TOP_LEVEL_CACHE:
        return _TOP_LEVEL_CACHE[path]
    top = None
    with gzip.open(path, 'rb') as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            stripped = chunk.lstrip()
            if stripped:
                top = stripped[:1].decode('utf-8', 'ignore')
                break
    _TOP_LEVEL_CACHE[path] = top
    return top


def iter_events_streaming(path):
    top = detect_top_level(path)
    prefix = 'traceEvents.item' if top == '{' else 'item'
    with gzip.open(path, 'rb') as f:
        for e in ijson.items(f, prefix):
            yield e


def iter_events_fallback(path):
    print("[WARN] 未安装 ijson，回退到 json.load 全量解析，大 trace 可能 OOM。", file=sys.stderr)
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    if isinstance(data, dict):
        events = data.get('traceEvents', [])
    elif isinstance(data, list):
        events = data
    else:
        events = []
    for e in events:
        yield e


def iter_events(path):
    if HAVE_IJSON:
        yield from iter_events_streaming(path)
    else:
        yield from iter_events_fallback(path)


def get_ext_id(e):
    args = e.get('args') or {}
    for k in ('External id', 'correlation'):
        v = args.get(k)
        if v is not None:
            return v
    for k in ('External id', 'correlation'):
        v = e.get(k)
        if v is not None:
            return v
    return None


def is_kernel(e):
    cat = (e.get('cat') or '').lower()
    return 'kernel' in cat


def is_cpu_op(e):
    return e.get('cat') == 'cpu_op'


def _norm_json(v):
    if v is None:
        return ''
    try:
        return json.dumps(v, sort_keys=True, separators=(',', ':'), default=str)
    except TypeError:
        return str(v)


def collect_kernel_durations(path):
    cuda_durations = defaultdict(float)
    total_kernel_time = 0.0
    kernel_events = 0
    kernel_without_ext = 0

    for e in iter_events(path):
        if not is_kernel(e):
            continue
        kernel_events += 1
        dur = float(e.get('dur') or 0.0)
        total_kernel_time += dur
        ext = get_ext_id(e)
        if ext is None:
            kernel_without_ext += 1
        else:
            cuda_durations[ext] += dur

    return cuda_durations, total_kernel_time, kernel_events, kernel_without_ext


def collect_cpu_op_stats(path, cuda_durations):
    stats = defaultdict(lambda: {
        "calls": 0,
        "cpu_time_us": 0.0,
        "cuda_time_us": 0.0,
    })
    matched_ext_ids = set()

    for e in iter_events(path):
        if not is_cpu_op(e):
            continue
        args = e.get('args') or {}
        if 'Input Dims' not in args:
            continue

        name = e.get('name') or ''
        in_dims = _norm_json(args.get('Input Dims'))
        in_type = _norm_json(args.get('Input type') or args.get('Input dtype'))
        
        key = (name, in_dims, in_type)

        rec = stats[key]
        rec['calls'] += 1
        rec['cpu_time_us'] += float(e.get('dur') or 0.0)

        ext = get_ext_id(e)
        if ext is not None and ext in cuda_durations:
            rec['cuda_time_us'] += cuda_durations[ext]
            matched_ext_ids.add(ext)

    return stats, matched_ext_ids


def make_sort_key(primary):
    def key(item):
        name = item[0][0]
        s = item[1]
        if primary == 'cpu':
            d = s['cpu_time_us']
        else:
            d = s['cuda_time_us'] if s['cuda_time_us'] > 0 else s['cpu_time_us']
        return (name, -d)
    return key


def write_report(stats, path, sort_primary, trace_path,
                 total_kernel_time, kernel_events, kernel_without_ext,
                 matched_kernel_time):
    sorted_items = sorted(stats.items(), key=make_sort_key(sort_primary))
    
    widths = {
        "Op": 40,
        "InputDims": 90,
        "InputType": 70,
        "Calls": 8,
        "TotalCPU": 12,
        "AvgCPU": 10,
        "TotalCUDA": 12,
        "AvgCUDA": 10,
    }

    lines = []
    # 写入表头注释
    lines.append("# Shape-Time Grouped Report")
    lines.append(f"# Trace file      : {trace_path}")
    lines.append(f"# Sort primary    : {sort_primary} (secondary: op name asc)")
    lines.append(f"# Kernel events   : {kernel_events}, total kernel time: {total_kernel_time:.3f} us")
    lines.append(f"# Kernel without External id: {kernel_without_ext}")
    lines.append(f"# Kernel time matched to cpu_op: {matched_kernel_time:.3f} us")
    lines.append("#")
    lines.append("# Note 1: cuda_time_us = SUM of kernel durations per op. If kernels run on")
    lines.append("#         multiple streams concurrently, this exceeds wall-clock GPU time.")
    lines.append("# Note 2: AvgCUDA_us = matched cuda_time_us / calls. If some calls have no")
    lines.append("#         matched kernel, the average is underestimated.")
    lines.append("# Note 3: cpu_op events are flat; nested parent/child ops may be double-counted.")
    lines.append("#")

    def build_row(op_t, dims_t, type_t, calls_t, cpu_t, avg_cpu_t, cuda_t, avg_cuda_t):
        return (
            f"{op_t.ljust(widths['Op'])} | "
            f"{dims_t.ljust(widths['InputDims'])} | "
            f"{type_t.ljust(widths['InputType'])} | "
            f"{calls_t.rjust(widths['Calls'])} | "
            f"{cpu_t.rjust(widths['TotalCPU'])} | "
            f"{avg_cpu_t.rjust(widths['AvgCPU'])} | "
            f"{cuda_t.rjust(widths['TotalCUDA'])} | "
            f"{avg_cuda_t.rjust(widths['AvgCUDA'])}"
        )

    # 生成表头
    header_line = build_row(
        "Op", "InputDims", "InputType", 
        "Calls", "TotalCPU_us", "AvgCPU_us", "TotalCUDA_us", "AvgCUDA_us"
    )
    lines.append(header_line)
    # 表头下方用 '='
    lines.append("=" * len(header_line))

    def split_text(text, width):
        if not text:
            return [""]
        return [text[i:i+width] for i in range(0, len(text), width)]

    prev_op_name = None

    # 填充数据
    for (name, in_dims, in_type), s in sorted_items:
        # 【新增】分隔符逻辑
        if prev_op_name is not None:
            if name != prev_op_name:
                # 不同的算子之间，用 '=' 分隔，并且加一个空行作为大段落的分割
                lines.append("=" * len(header_line))
                lines.append("") 
            else:
                # 同一个算子的不同 Shape 之间，用 '-' 分隔
                lines.append("-" * len(header_line))
        prev_op_name = name
        
        calls = s['calls']
        avg_cpu = s['cpu_time_us'] / calls if calls else 0.0
        avg_cuda = s['cuda_time_us'] / calls if calls else 0.0
        
        op_parts = split_text(name, widths['Op'])
        dims_parts = split_text(in_dims, widths['InputDims'])
        type_parts = split_text(in_type, widths['InputType'])
        
        max_lines = max(len(op_parts), len(dims_parts), len(type_parts))
        
        for i in range(max_lines):
            op_t = op_parts[i] if i < len(op_parts) else ""
            dims_t = dims_parts[i] if i < len(dims_parts) else ""
            type_t = type_parts[i] if i < len(type_parts) else ""
            
            calls_t = str(calls) if i == 0 else ""
            cpu_t = f"{s['cpu_time_us']:.3f}" if i == 0 else ""
            avg_cpu_t = f"{avg_cpu:.3f}" if i == 0 else ""
            cuda_t = f"{s['cuda_time_us']:.3f}" if i == 0 else ""
            avg_cuda_t = f"{avg_cuda:.3f}" if i == 0 else ""
            
            lines.append(build_row(op_t, dims_t, type_t, calls_t, cpu_t, avg_cpu_t, cuda_t, avg_cuda_t))

    # 结尾再加一条线收尾
    lines.append("=" * len(header_line))

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已写入: {path}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch trace shape-time grouped report")
    parser.add_argument('--trace', default=DEFAULT_TRACE, help="输入 trace 路径")
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help="输出报告路径")
    parser.add_argument('--sort', choices=['cuda', 'cpu'], default='cuda', help="排序主键")
    args = parser.parse_args()

    print(f"Pass 1/2: collecting kernel durations from {args.trace} ...")
    cuda_durations, total_kernel_time, kernel_events, kernel_without_ext = collect_kernel_durations(args.trace)
    print(f"  kernel events={kernel_events}, unique ext ids={len(cuda_durations)}, total kernel time={total_kernel_time:.3f} us")

    print("Pass 2/2: aggregating cpu_op by (name, InputDims, InputType) ...")
    stats, matched_ext_ids = collect_cpu_op_stats(args.trace, cuda_durations)
    matched_kernel_time = sum(cuda_durations[e] for e in matched_ext_ids)
    print(f"  groups={len(stats)}, matched ext ids={len(matched_ext_ids)}, matched kernel time={matched_kernel_time:.3f} us")

    write_report(stats, args.output, args.sort, args.trace,
                 total_kernel_time, kernel_events, kernel_without_ext,
                 matched_kernel_time)


if __name__ == "__main__":
    main()