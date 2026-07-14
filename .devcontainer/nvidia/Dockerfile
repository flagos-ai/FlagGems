# Base image produced by build-infra (see .devcontainer/build-infra/runtime/Containerfile).
# Naming convention: flagos-runtime-{vendor}-{backend}:latest
# Override this arg to pin to a specific registry or tag without editing this file.
ARG IMAGE=flagos-runtime-nvidia-cuda12.8:latest

FROM ${IMAGE}

# ── Dev-container user ─────────────────────────────────────────
# VS Code expects a non-root user; UID/GID are re-aligned to the host
# user at container start via updateRemoteUserUID.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# The nvidia runtime image is based on ubuntu and runs as root.
# Create the vscode user and configure passwordless sudo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd  --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
