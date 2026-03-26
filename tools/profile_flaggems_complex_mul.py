import argparse
import os
import traceback
from datetime import datetime

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import flag_gems


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _write_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as file:
        file.write(text)


def _get_device_time_total(event) -> float:
    if hasattr(event, "device_time_total"):
        return float(event.device_time_total)
    if hasattr(event, "cuda_time_total"):
        return float(event.cuda_time_total)
    return 0.0


def _get_self_device_time_total(event) -> float:
    if hasattr(event, "self_device_time_total"):
        return float(event.self_device_time_total)
    if hasattr(event, "self_cuda_time_total"):
        return float(event.self_cuda_time_total)
    return 0.0


def _dump_events_with_stacks(prof, path: str):
    lines = []
    for idx, event in enumerate(prof.events()):
        lines.append(f"[{idx}] name={event.name}")
        lines.append(
            f"  cpu_time_total(us)={event.cpu_time_total:.3f}, device_time_total(us)={_get_device_time_total(event):.3f}"
        )
        if hasattr(event, "input_shapes"):
            lines.append(f"  input_shapes={event.input_shapes}")
        if event.stack:
            lines.append("  stack:")
            for frame in event.stack:
                lines.append(f"    {frame}")
        else:
            lines.append("  stack: <empty>")
        lines.append("")
    _write_text(path, "\n".join(lines))


def run_profile_for_dtype(args, dtype: torch.dtype, outdir: str):
    dtype_tag = _dtype_name(dtype)
    base = f"complex_mul_{dtype_tag}"

    trace_path = os.path.join(outdir, f"{base}_trace.json")
    stacks_cuda_path = os.path.join(outdir, f"{base}_stacks_cuda.txt")
    stacks_cpu_path = os.path.join(outdir, f"{base}_stacks_cpu.txt")
    table_path = os.path.join(outdir, f"{base}_key_averages.txt")
    events_path = os.path.join(outdir, f"{base}_events_full_stacks.txt")
    summary_path = os.path.join(outdir, f"{base}_summary.txt")

    device = flag_gems.device
    a = torch.randn((args.m, args.n), device=device, dtype=dtype)
    b = torch.randn((args.m, args.n), device=device, dtype=dtype)
    if args.conj_input:
        a = a.conj()
        b = b.conj()

    with flag_gems.use_gems():
        for _ in range(args.warmup):
            _ = torch.mul(a, b)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with flag_gems.use_gems():
            for i in range(args.iters):
                with record_function(
                    f"CALL torch.mul dtype={dtype_tag} iter={i} @ profile_flaggems_complex_mul.py"
                ):
                    out = torch.mul(a, b)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    prof.export_stacks(stacks_cuda_path, "self_cuda_time_total")
    prof.export_stacks(stacks_cpu_path, "self_cpu_time_total")

    table = prof.key_averages(group_by_stack_n=16).table(
        sort_by="self_cuda_time_total", row_limit=args.row_limit
    )
    _write_text(table_path, table)
    _dump_events_with_stacks(prof, events_path)

    suspicious_keywords = ("mul", "copy", "conj", "real", "imag", "redispatch")
    suspicious = []
    for event in prof.key_averages(group_by_input_shape=False):
        name = event.key
        if any(keyword in name for keyword in suspicious_keywords):
            suspicious.append(
                f"{name:70s} | calls={event.count:<4d} | cpu_self(us)={event.self_cpu_time_total:10.3f} | device_self(us)={_get_self_device_time_total(event):10.3f}"
            )

    summary_lines = [
        f"timestamp={datetime.now().isoformat()}",
        f"dtype={dtype}",
        f"shape=({args.m}, {args.n})",
        f"conj_input={args.conj_input}",
        f"trace={trace_path}",
        f"stacks_cuda={stacks_cuda_path}",
        f"stacks_cpu={stacks_cpu_path}",
        f"table={table_path}",
        f"events={events_path}",
        "",
        "[suspicious key averages]",
        *suspicious,
    ]
    _write_text(summary_path, "\n".join(summary_lines))

    print(f"[OK] dtype={dtype}")
    print(f"  trace   : {trace_path}")
    print(f"  stacks  : {stacks_cuda_path}")
    print(f"  stacks  : {stacks_cpu_path}")
    print(f"  table   : {table_path}")
    print(f"  events  : {events_path}")
    print(f"  summary : {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Profile FlagGems complex mul with full stack information"
    )
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    parser.add_argument("--row-limit", type=int, default=200)
    parser.add_argument(
        "--conj-input",
        action="store_true",
        help="Use conjugated complex tensors as input",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/workspace/FlagGems/profiler_out/complex_mul_profiles",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    dtypes = [torch.complex64, torch.complex128]
    errors = []
    for dtype in dtypes:
        try:
            run_profile_for_dtype(args, dtype, args.outdir)
        except Exception as exc:
            error_msg = (
                f"[FAIL] dtype={dtype}: {type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )
            print(error_msg)
            errors.append(error_msg)

    if errors:
        failed_path = os.path.join(args.outdir, "failed_dtypes.txt")
        _write_text(failed_path, "\n\n".join(errors))
        print(f"[WARN] some dtypes failed, see: {failed_path}")


if __name__ == "__main__":
    main()