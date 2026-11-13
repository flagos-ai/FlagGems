# benchmark/run_all_bench.py
import argparse

from bench_conv2d import bench_conv2d
from bench_glu import bench_glu
from bench_mm import bench_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="output csv path")
    args = parser.parse_args()
    csv_fd = open(args.csv, "a") if args.csv else None

    def log(line):
        print(line)
        if csv_fd:
            csv_fd.write(line + "\n")
            csv_fd.flush()

    shapes_mm = [
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
    ]
    for M, N, K in shapes_mm:
        t_gems, t_ref, sp = bench_mm(M, N, K)
        log(f"mm,{M},{N},{K},{t_gems*1e6:.2f},{t_ref*1e6:.2f},{sp:.4f}")

    shapes_conv = [
        (16, 64, 32, 32),
        (8, 128, 64, 64),
    ]
    for N, C, H, W in shapes_conv:
        t_gems, t_ref, sp = bench_conv2d(N, C, H, W)
        log(f"conv2d,{N},{C},{H},{W},{t_gems*1e6:.2f},{t_ref*1e6:.2f},{sp:.4f}")

    shapes_glu = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ]
    for B, D in shapes_glu:
        t_gems, t_ref, sp = bench_glu(B, D)
        log(f"glu,{B},,{D},{t_gems*1e6:.2f},{t_ref*1e6:.2f},{sp:.4f}")

    if csv_fd:
        csv_fd.close()


if __name__ == "__main__":
    main()
