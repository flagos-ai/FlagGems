# FlagGems Dev Containers

Each subdirectory is an independent VS Code Dev Container targeting one hardware backend.
When you open this repo in VS Code, it will prompt you to select a configuration.

## Available Backends

| Directory    | Backend| CMake Flag              | Hardware         |
|--------------|--------|-------------------------|------------------|
| `nvidia/`    | CUDA   | `FLAGGEMS_BACKEND=CUDA` | NVIDIA GPU       |
| `iluvatar/`  | IX     | `FLAGGEMS_BACKEND=IX`   | Iluvatar GPU     |
| `Enflame/`   | GCU    | `FLAGGEMS_BACKEND=GCU`  | Enflame GCU      |
| `mthreads/`  | MUSA   | `FLAGGEMS_BACKEND=MUSA` | Moore Threads GPU |

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
- `IX`    → `FLAGGEMS_USE_IX`    (Iluvatar)
- `MUSA`  → `FLAGGEMS_USE_MUSA`  (Moore Threads)
- `NPU`   → `FLAGGEMS_USE_NPU`   (Ascend)
- `GCU`   → `FLAGGEMS_USE_GCU`   (Enflame)
