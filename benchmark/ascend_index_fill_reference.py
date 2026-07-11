"""Direct ACLNN reference used only by the Ascend index_fill benchmark."""

import os
from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


_MODULE_NAME = "flaggems_ascend_index_fill_reference"
_BUILD_DIR_ENV = "FLAGGEMS_INDEX_FILL_REFERENCE_BUILD_DIR"
_VERBOSE_ENV = "FLAGGEMS_INDEX_FILL_REFERENCE_VERBOSE"


def _cann_root():
    configured_root = os.environ.get("ASCEND_HOME_PATH")
    candidates = (
        Path(configured_root) if configured_root else None,
        Path("/usr/local/Ascend/cann-8.5.0"),
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "aarch64-linux").is_dir():
            return candidate
    raise RuntimeError("Could not locate a CANN installation with aarch64-linux")


def _torch_npu_root():
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("torch_npu is required for the ACLNN reference") from error
    return Path(torch_npu.__file__).resolve().parent


def _build_dir():
    return Path(
        os.environ.get(
            _BUILD_DIR_ENV,
            "/tmp/flaggems-index-fill-aclnn-reference",
        )
    )


@lru_cache(maxsize=1)
def load_reference():
    """Build and load the direct ACLNN extension in a temporary directory."""
    cann_root = _cann_root()
    cann_arch_root = cann_root / "aarch64-linux"
    cann_lib = cann_arch_root / "lib64"
    torch_npu_root = _torch_npu_root()
    torch_npu_lib = torch_npu_root / "lib"
    source = Path(__file__).with_name("csrc") / "ascend_index_fill_reference.cpp"
    build_directory = _build_dir()
    build_directory.mkdir(parents=True, exist_ok=True)

    # Shared 910B hosts can be memory constrained while compiling C++ extensions.
    os.environ.setdefault("MAX_JOBS", "1")
    return load(
        name=_MODULE_NAME,
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=[
            str(cann_root / "include"),
            str(cann_arch_root / "include"),
            str(torch_npu_root / "include"),
        ],
        extra_ldflags=[
            f"-L{cann_lib}",
            "-lopapi",
            "-lascendcl",
            "-lruntime",
            f"-L{torch_npu_lib}",
            "-ltorch_npu",
            f"-Wl,-rpath,{cann_lib}",
            f"-Wl,-rpath,{torch_npu_lib}",
        ],
        build_directory=str(build_directory),
        verbose=os.environ.get(_VERBOSE_ENV) == "1",
    )


def index_fill(input, dim, index, value):
    return load_reference().index_fill(input, dim, index, value)


def index_fill_(input, dim, index, value):
    return load_reference().index_fill_(input, dim, index, value)
