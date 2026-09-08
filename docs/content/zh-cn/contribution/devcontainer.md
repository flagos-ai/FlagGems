---
title: 开发容器
weight: 15
---

<!--
# Dev Container (Recommended)

FlagGems ships VS Code Dev Container configurations for each supported hardware backend.
A Dev Container gives you a ready-to-use development environment — the correct base image,
drivers, pip dependencies, and FlagGems installed in editable mode — without touching your
host system.
-->
# 开发容器（推荐）

FlagGems 为每种支持的硬件后端提供了 VS Code Dev Container 配置。
开发容器能够为你提供一个开箱即用的开发环境——包括正确的基础镜像、驱动、pip 依赖，
以及以可编辑模式（editable mode）安装好的 FlagGems——无需对宿主机做任何配置。

<!--
## Prerequisites

- [VS Code](https://code.visualstudio.com/) with the
  [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- Docker (with GPU pass-through support for your hardware)
-->
## 前提条件

- 安装了 [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) 扩展的 [VS Code](https://code.visualstudio.com/)
- Docker（需支持对应硬件的 GPU 透传）

<!--
## Available backends

| Directory | Backend | Hardware   |
|-----------|---------|------------|
| `nvidia/` | CUDA    | NVIDIA GPU |
| `hygon/`  | DTK     | HYGON DCU  |

See [`.devcontainer/README.md`](https://github.com/FlagOpen/FlagGems/blob/master/.devcontainer/README.md)
for details on the directory layout and instructions for adding a new backend.
-->
## 可用后端

| 目录        | 后端  | 硬件       |
|-------------|-------|------------|
| `nvidia/`   | CUDA  | NVIDIA GPU |
| `hygon/`    | DTK   | 海光 DCU   |

更多目录结构说明及新增后端的指引，请参阅
[`.devcontainer/README_CN.md`](https://github.com/FlagOpen/FlagGems/blob/master/.devcontainer/README_CN.md)。

---

<!--
## Scenario A — Local machine

Use this path when Docker and your GPU are on the same machine as VS Code.
-->
## 场景 A — 本地机器

适用于 Docker、GPU 和 VS Code 均在同一台机器上的情况。

```
┌─────────────────────────────────────────────────────────────┐
│                        本地机器                              │
│                                                             │
│  1. 克隆仓库              2. 在 VS Code 中打开               │
│  git clone …              文件 › 打开文件夹…                 │
│       │                          │                          │
│       └──────────────┬───────────┘                          │
│                      ▼                                      │
│         VS Code 检测到 .devcontainer/                        │
│                      │                                      │
│                      ▼                                      │
│      弹出提示："在容器中重新打开？"                            │
│      → 点击  在容器中重新打开                                 │
│      → 选择后端（nvidia / hygon）                            │
│                      │                                      │
│                      ▼                                      │
│      Docker 构建镜像（仅首次）                                │
│                      │                                      │
│                      ▼                                      │
│      执行 postCreateCommand                                  │
│      install-dev-tools.sh → pip install -e .                │
│                      │                                      │
│                      ▼                                      │
│      ✓  VS Code 工作区在容器内打开                            │
└─────────────────────────────────────────────────────────────┘
```

<!--
**Step-by-step:**

1. Clone the repository and open the folder in VS Code.
2. VS Code detects `.devcontainer/` and shows the **Reopen in Container** notification.
   Click it (or run **Dev Containers: Reopen in Container** from the Command Palette).
3. Select your hardware backend when prompted.
4. Docker builds the image on the first launch. Subsequent opens reuse the cached image.
5. After the build, `install-dev-tools.sh` runs automatically and installs FlagGems in
   editable mode. The VS Code workspace is now inside the container.
-->
**操作步骤：**

1. 克隆仓库，在 VS Code 中打开对应文件夹。
2. VS Code 检测到 `.devcontainer/` 后会弹出**在容器中重新打开**的提示，点击即可
   （也可在命令面板中运行 **Dev Containers: Reopen in Container**）。
3. 在提示中选择硬件后端。
4. Docker 在首次启动时构建镜像，后续启动会复用已缓存的镜像。
5. 构建完成后，`install-dev-tools.sh` 会自动执行，以可编辑模式安装 FlagGems。
   此后 VS Code 工作区即运行在容器内部。

---

<!--
## Scenario B — Remote server via SSH

Use this path when the GPU is on a remote server. VS Code connects to the server over SSH
and then starts the Dev Container on the server side.
-->
## 场景 B — 通过 SSH 登录远端服务器后进入容器

适用于 GPU 部署在远端服务器的情况。VS Code 先通过 SSH 连接到服务器，再在服务器端启动 Dev Container。

```
┌─────────────────────┐          ┌──────────────────────────────────────────┐
│      本地机器        │   SSH    │                远端服务器                │
│                     │◄────────►│                                          │
│  VS Code            │          │  1. 仓库已克隆至服务器（或现在克隆）       │
│  + Remote-SSH 扩展  │          │  2. 服务器已安装 Docker 和 GPU 驱动       │
│  + Dev Containers   │          │                                          │
│        │            │          │                                          │
│        │  SSH 连接   │          │                                          │
│        ├────────────►│          │                                          │
│        │            │          │   VS Code Server 在服务器上运行           │
│        │            │          │              │                           │
│        │            │          │              ▼                           │
│        │            │          │   VS Code 检测到 .devcontainer/           │
│        │            │          │              │                           │
│        │◄───────────┤          │              ▼                           │
│  弹出提示：          │          │   弹出提示："在容器中重新打开？"            │
│  在容器中重新打开    │          │   → 选择后端（nvidia / hygon）             │
│        │            │          │              │                           │
│        │            │          │              ▼                           │
│        │            │          │   Docker 在服务器上构建镜像               │
│        │            │          │              │                           │
│        │            │          │              ▼                           │
│        │            │          │   执行 postCreateCommand                 │
│        │            │          │   install-dev-tools.sh                   │
│        │            │          │              │                           │
│        │            │          │              ▼                           │
│        │◄───────────┤          │   ✓  容器在服务器上运行                   │
│  本地渲染 VS Code UI │          │      VS Code UI 转发至本地显示            │
└─────────────────────┘          └──────────────────────────────────────────┘
```

<!--
**Step-by-step:**

1. Install the
   [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
   extension in VS Code (in addition to Dev Containers).
2. Connect to the remote server: open the Command Palette and run
   **Remote-SSH: Connect to Host…**, then enter `user@hostname`.
3. Once connected, open the repository folder on the remote server
   (**File › Open Folder…**).
4. VS Code detects `.devcontainer/` on the remote host and shows the
   **Reopen in Container** notification. Click it and select your backend.
5. Docker builds the image **on the remote server**. Port forwarding, file I/O,
   and terminal sessions all run remotely; only the UI is rendered locally.
6. `install-dev-tools.sh` runs automatically after the build and installs FlagGems
   in editable mode on the remote container.
-->
**操作步骤：**

1. 在 VS Code 中安装
   [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)
   扩展（与 Dev Containers 扩展配合使用）。
2. 连接远端服务器：打开命令面板，运行 **Remote-SSH: Connect to Host…**，
   输入 `user@hostname`。
3. 连接成功后，在远端服务器上打开仓库所在文件夹（**文件 › 打开文件夹…**）。
4. VS Code 检测到远端的 `.devcontainer/` 并弹出**在容器中重新打开**的提示，
   点击并选择后端。
5. Docker 在**远端服务器**上构建镜像。端口转发、文件 I/O 和终端会话均在远端执行，
   仅 UI 界面渲染在本地。
6. 构建完成后，`install-dev-tools.sh` 在远端容器内自动运行，以可编辑模式安装 FlagGems。
