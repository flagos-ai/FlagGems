# FlagGems Dev Containers

[中文版](README_CN.md)

Each subdirectory is an independent VS Code Dev Container targeting one hardware backend.
When you open this repo in VS Code, it will prompt you to select a configuration.

## Available Backends

| Directory | Backend | CMake Flag               | Hardware   |
|-----------|---------|--------------------------|------------|
| `nvidia/` | CUDA    | `FLAGGEMS_BACKEND=CUDA`  | NVIDIA GPU |
| `hygon/`  | DTK     | `FLAGGEMS_BACKEND=CUDA`  | HYGON DCU  |

## Structure

```
.devcontainer/
├── README.md                          # this file
├── README_CN.md                       # Chinese version
├── common/
│   └── scripts/
│       └── install-flaggems.sh        # shared install logic, consumes env vars
└── <backend>/
    ├── devcontainer.json              # VS Code Dev Container config
    ├── Dockerfile                     # base image + build dependencies
    ├── flaggems.env                   # backend-specific CMAKE_ARGS and env vars
    └── scripts/
        └── install-dev-tools.sh       # sources flaggems.env, then calls the common script
```

## Quick Start

1. Install [VS Code](https://code.visualstudio.com/) and the
   [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension.
2. Open this repository in VS Code.
3. When prompted, select **Reopen in Container** and choose your hardware backend.
4. The container builds automatically and runs `install-dev-tools.sh` on first start,
   which installs FlagGems in editable mode.

## Adding a New Backend

1. Create a new directory under `.devcontainer/<backend>/`.
2. Copy the structure from an existing backend (e.g., `nvidia/`).
3. Update `flaggems.env` with the appropriate `FLAGGEMS_BACKEND` and `CMAKE_ARGS`.
4. Update `Dockerfile` to use the correct base image and pip index URL.
5. Update `devcontainer.json` with the correct device mounts and container name.

## Backend-to-CMake Mapping

The `FLAGGEMS_BACKEND` values are consumed by `CMakeLists.txt`:

| Value  | CMake Variable        | Hardware          |
|--------|-----------------------|-------------------|
| `CUDA` | `FLAGGEMS_USE_CUDA`   | NVIDIA / HYGON    |
| `IX`   | `FLAGGEMS_USE_IX`     | Iluvatar          |
| `MUSA` | `FLAGGEMS_USE_MUSA`   | Moore Threads     |
| `NPU`  | `FLAGGEMS_USE_NPU`    | Ascend            |
| `GCU`  | `FLAGGEMS_USE_GCU`    | Enflame           |
