# FlagGems 开发容器

每个子目录是一个独立的 VS Code Dev Container，对应一种硬件后端。
在 VS Code 中打开本仓库时，会提示你选择一个配置。

## 可用后端

| 目录        | 后端  | CMake 标志               | 硬件        |
|-------------|-------|--------------------------|-------------|
| `nvidia/`   | CUDA  | `FLAGGEMS_BACKEND=CUDA`  | NVIDIA GPU  |
| `hygon/`    | DTK   | `FLAGGEMS_BACKEND=CUDA`  | 海光 DCU    |

## 目录结构

```
.devcontainer/
├── Dockerfile                          # 基础镜像与构建依赖
├── README.md                           # 英文说明文档
├── README_CN.md                        # 本文档（中文）
├── common/
│   └── scripts/
│       └── install-flaggems.sh         # 共享安装逻辑脚本
└── <backend>/                          # 特定后端目录（例如 hygon, nvidia-cuda12.8 等）
    └── devcontainer.json               # VS Code Dev Container 配置文件
```

## 快速开始

1. 安装 [VS Code](https://code.visualstudio.com/) 和
   [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 扩展。
2. 在 VS Code 中打开本仓库。
3. 出现提示时选择 **在容器中重新打开（Reopen in Container）**，并选择对应的硬件后端。
4. 首次启动时，VS Code 会自动从镜像仓库拉取预构建的 Dev Container 镜像（若本地不存在），拉取完成后即可直接进入开发环境。
5. 容器启动后，`postCreateCommand` 以可编辑模式安装 FlagGems。

## 添加新后端

1. 在 `.devcontainer/<backend>/` 下创建新目录。
2. 参考已有后端（如 `nvidia/`）复制目录结构。
3. 修改 `flaggems.env`，填写对应的 `FLAGGEMS_BACKEND` 和 `CMAKE_ARGS`。
4. 修改 `Dockerfile`：将默认的 `IMAGE` 构建参数设置为由
   [`build-infra`](.devcontainer/build-infra) 构建的 runtime 镜像，
   命名规则为 `flagos-runtime-{vendor}-{backend}:latest`（见 `.devcontainer/build-infra/configs.yaml`）。
5. 修改 `devcontainer.json`，配置正确的设备挂载和容器名称。

## 后端与 CMake 变量对照

`FLAGGEMS_BACKEND` 的值由 `CMakeLists.txt` 读取：

| 值     | CMake 变量            | 硬件            |
|--------|-----------------------|-----------------|
| `CUDA` | `FLAGGEMS_USE_CUDA`   | NVIDIA / 海光   |
| `IX`   | `FLAGGEMS_USE_IX`     | 天数智芯        |
| `MUSA` | `FLAGGEMS_USE_MUSA`   | 摩尔线程        |
| `NPU`  | `FLAGGEMS_USE_NPU`    | 昇腾            |
| `GCU`  | `FLAGGEMS_USE_GCU`    | 燧原科技        |
