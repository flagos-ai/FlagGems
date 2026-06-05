# FlagGems Dev Containers

Each subdirectory is an independent VS Code Dev Container targeting one hardware backend.
When you open this repo in VS Code, it will prompt you to select a configuration.

## Available Backends

| Directory    | Backend     | CMake Flag              | Hardware         |
|--------------|-------------|-------------------------|------------------|
| `nvidia/`    | CUDA        | `FLAGGEMS_BACKEND=CUDA` | NVIDIA GPU       |
| `iluvatar/`  | IX (Iluvatar)| `FLAGGEMS_BACKEND=IX`  | 天数智芯 GPU     |
| `metax/`     | GCU (MetaX) | `FLAGGEMS_BACKEND=GCU`  | 沐曦 GPU         |
| `mthreads/`  | MUSA (Moore Threads) | `FLAGGEMS_BACKEND=MUSA` | 摩尔线程 GPU |

## Structure

```
.devcontainer/
├── README.md                          # this file
├── common/
│   └── scripts/
│       └── install-flaggems.sh        # shared install logic, consumes env vars
└── <backend>/
    ├── devcontainer.json              # VS Code Dev Container config
    ├── Dockerfile                     # base image + build dependencies
    ├── flaggems.env                   # backend-specific CMAKE_ARGS and env vars
    └── scripts/
        └── install-dev-tools.sh       # source flaggems.env → call common script
```

## Adding a New Backend

1. Create a new directory under `.devcontainer/<backend>/`
2. Copy the structure from an existing backend (e.g., `nvidia/`)
3. Update `flaggems.env` with the appropriate `FLAGGEMS_BACKEND` and `CMAKE_ARGS`
4. Update `Dockerfile` to use the correct base image and pip index URL
5. Update `devcontainer.json` with the correct device mount and container name

## Backend-to-CMake Mapping

The `FLAGGEMS_BACKEND` values come from `CMakeLists.txt`:

- `CUDA`  → `FLAGGEMS_USE_CUDA`  (also used by Iluvatar IX backend)
- `IX`    → `FLAGGEMS_USE_IX`    (天数智芯)
- `MUSA`  → `FLAGGEMS_USE_MUSA`  (摩尔线程)
- `NPU`   → `FLAGGEMS_USE_NPU`   (Ascend)
- `GCU`   → `FLAGGEMS_USE_GCU`   (MetaX / Enflame)
