---
title: Dev Container
weight: 15
---
# Dev Container (Recommended)

FlagGems ships VS Code Dev Container configurations for each supported hardware backend.
A Dev Container gives you a ready-to-use development environment — the correct base image,
drivers, pip dependencies, and FlagGems installed in editable mode — without touching your
host system.

## Prerequisites

- [VS Code](https://code.visualstudio.com/) with the
  [Dev Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) extension
- Docker (with GPU pass-through support for your hardware)

## Available backends

| Directory | Backend | Hardware   |
|-----------|---------|------------|
| `nvidia/` | CUDA    | NVIDIA GPU |
| `hygon/`  | DTK     | HYGON DCU  |

See [`.devcontainer/README.md`](https://github.com/FlagOpen/FlagGems/blob/master/.devcontainer/README.md)
for details on the directory layout and instructions for adding a new backend.

---

## Scenario A — Local machine

Use this path when Docker and your GPU are on the same machine as VS Code.

```
┌─────────────────────────────────────────────────────────────┐
│                     Local machine                           │
│                                                             │
│  1. Clone repo          2. Open in VS Code                  │
│  git clone …            File › Open Folder…                 │
│       │                        │                            │
│       └───────────┬────────────┘                            │
│                   ▼                                         │
│        VS Code detects .devcontainer/                       │
│                   │                                         │
│                   ▼                                         │
│     "Reopen in Container?" prompt appears                   │
│     → click  Reopen in Container                            │
│     → select backend  (nvidia / hygon)                      │
│                   │                                         │
│                   ▼                                         │
│     Docker builds image (first time only)                   │
│                   │                                         │
│                   ▼                                         │
│     postCreateCommand runs                                  │
│     install-dev-tools.sh → pip install -e .                 │
│                   │                                         │
│                   ▼                                         │
│     ✓  VS Code workspace opens inside container             │
└─────────────────────────────────────────────────────────────┘
```

**Step-by-step:**

1. Clone the repository and open the folder in VS Code.
2. VS Code detects `.devcontainer/` and shows the **Reopen in Container** notification.
   Click it (or run **Dev Containers: Reopen in Container** from the Command Palette).
3. Select your hardware backend when prompted.
4. Docker builds the image on the first launch. Subsequent opens reuse the cached image.
5. After the build, `install-dev-tools.sh` runs automatically and installs FlagGems in
   editable mode. The VS Code workspace is now inside the container.

---

## Scenario B — Remote server via SSH

Use this path when the GPU is on a remote server. VS Code connects to the server over SSH
and then starts the Dev Container on the server side.

```
┌──────────────────────┐          ┌────────────────────────────────────────┐
│    Local machine     │   SSH    │            Remote server               │
│                      │◄────────►│                                        │
│  VS Code             │          │  1. Repo already cloned (or clone now) │
│  + Remote-SSH ext.   │          │  2. Docker + GPU drivers installed     │
│  + Dev Containers    │          │                                        │
│        │             │          │                                        │
│        │  SSH connect │          │                                        │
│        ├─────────────►│          │                                        │
│        │             │          │  VS Code Server runs on remote host    │
│        │             │          │           │                            │
│        │             │          │           ▼                            │
│        │             │          │  VS Code detects .devcontainer/        │
│        │             │          │           │                            │
│        │◄────────────┤          │           ▼                            │
│   "Reopen in         │          │  "Reopen in Container?" prompt         │
│    Container?"       │          │  → select backend (nvidia / hygon)     │
│        │             │          │           │                            │
│        │             │          │           ▼                            │
│        │             │          │  Docker builds image on remote host    │
│        │             │          │           │                            │
│        │             │          │           ▼                            │
│        │             │          │  postCreateCommand runs                │
│        │             │          │  install-dev-tools.sh                  │
│        │             │          │           │                            │
│        │             │          │           ▼                            │
│        │◄────────────┤          │  ✓  Container running on remote host   │
│  VS Code UI renders  │          │     VS Code UI forwarded to local      │
└──────────────────────┘          └────────────────────────────────────────┘
```

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
