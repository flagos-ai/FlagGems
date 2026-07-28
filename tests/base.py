import time

import torch


class Benchmark:
    def __init__(self, op_name, torch_op, dtypes, device=None):
        self.op_name = op_name
        self.torch_op = torch_op
        self.dtypes = dtypes
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.shapes = []
        self.set_shapes()

    def set_shapes(self, shape_file_path=None):
        pass

    def get_input_iter(self, cur_dtype):
        raise NotImplementedError

    def run(self):
        print(f"\n=== Benchmarking {self.op_name} on {self.device} ===")

        for dtype in self.dtypes:
            print(f"\n--- dtype: {dtype} ---")
            print(f"{'Shape':<20} {'Torch (ms)':<14} {'Gems (ms)':<14} {'Speedup':<10}")
            print("-" * 60)

            test_data = []
            for inputs in self.get_input_iter(dtype):
                test_data.append(inputs)

            for inputs in test_data:
                for _ in range(10):
                    self.torch_op(*inputs)
                    if hasattr(self, "gems_op"):
                        self.gems_op(*inputs)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                torch_times = []
                for _ in range(100):
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    self.torch_op(*inputs)
                    if self.device == "cuda":
                        torch.cuda.synchronize()
                    torch_times.append((time.perf_counter() - start) * 1000)
                torch_time = sum(torch_times) / len(torch_times)

                if hasattr(self, "gems_op"):
                    gems_times = []
                    for _ in range(100):
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        self.gems_op(*inputs)
                        if self.device == "cuda":
                            torch.cuda.synchronize()
                        gems_times.append((time.perf_counter() - start) * 1000)
                    gems_time = sum(gems_times) / len(gems_times)
                    speedup = torch_time / gems_time
                else:
                    gems_time = None
                    speedup = None

                shape_str = str(list(inputs[0].shape))
                if gems_time is not None:
                    print(
                        f"{shape_str:<20} {torch_time:<14.6f} {gems_time:<14.6f} {speedup:<10.2f}"
                    )
                else:
                    print(
                        f"{shape_str:<20} {torch_time:<14.6f} {'N/A':<14} {'N/A':<10}"
                    )
