# benchmark/test_is_strides_like_format.py
import torch
from torch.utils.benchmark import Timer

import flag_gems


def main():
    shapes = [
        (2, 3),  # 2D
        (4, 5, 6),  # 3D
        (2, 3, 4, 5),  # 4D contiguous & channels_last
        (8, 3, 224, 224),  # 大4D
    ]
    formats = ["contiguous", "channels_last", "any"]

    #  NPU doesn't support channels_last,skip
    if flag_gems.device == "npu":
        if "channels_last" in formats:
            print(
                "ERROR: npu not support channels_last format, skipping this benchmark"
            )
            formats = [fmt for fmt in formats if fmt != "channels_last"]

    print("Benchmarking is_strides_like_format (time in microseconds)")
    print("-" * 60)

    for shape in shapes:
        x = torch.randn(shape).to(flag_gems.device)

        # only turn to channels_last and dim equals 4D
        x_cl = None
        if "channels_last" in formats and len(shape) == 4:
            try:
                x_cl = x.contiguous(memory_format=torch.channels_last)
            except RuntimeError:
                # if false,exit
                print(
                    f"ERROR: npu not support channels_last for shape {shape}, removing format"
                )
                formats = [fmt for fmt in formats if fmt != "channels_last"]
                x_cl = None

        for fmt in formats:
            if fmt == "channels_last" and x_cl is not None:
                tensor = x_cl
            else:
                tensor = x

            if fmt == "channels_last" and len(shape) != 4:
                continue

            timer = Timer(
                stmt="is_strides_like_format(t, fmt)",
                globals={
                    "is_strides_like_format": flag_gems.is_strides_like_format,
                    "t": tensor,
                    "fmt": fmt,
                },
                num_threads=1,
            )
            time_us = timer.timeit(1000).median * 1e6
            print(
                f"Shape: {str(shape):<20} | Format: {fmt:<15} | Time: {time_us:.3f} us"
            )


if __name__ == "__main__":
    main()
