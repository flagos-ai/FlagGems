from pathlib import Path

import torch_musa
from setuptools import setup
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension

torch_musa_root = Path(torch_musa.__file__).resolve().parent
setup(
    name="native_index_put_ext",
    ext_modules=[
        MUSAExtension(
            "native_index_put_ext",
            ["bindings.cpp", "kernel.mu"],
            include_dirs=[str(torch_musa_root)],
            extra_compile_args={"cxx": ["-O3", "-std=c++17"], "mcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
