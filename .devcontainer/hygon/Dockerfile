# Base image produced by build-infra (see .devcontainer/build-infra/runtime/Containerfile).
# Naming convention: flagos-runtime-{vendor}-{backend}:latest
# Override this arg to pin to a specific registry or tag without editing this file.
ARG IMAGE=flagos-runtime-hygon-dtk26.04:latest

FROM ${IMAGE}

# ── Dev-container user ─────────────────────────────────────────
# VS Code expects a non-root user named vscode; UID/GID are re-aligned to
# the host user at container start via updateRemoteUserUID.
ARG USERNAME=vscode

# The hygon runtime image runs as root; it also ships with ubuntu:1000.
# Rename that user/group to vscode and configure passwordless sudo
# rather than creating a duplicate GID.
#
# chmod 755 on the apt partial dir is required on Ubuntu 24.04: the runtime
# image clears /var/lib/apt/lists/* and the partial/ directory is left owned
# by _apt with mode 700, which makes apt-get install fail with exit code 4
# when running as root inside a non-privileged build context.
RUN chmod 755 /var/cache/apt/archives/partial \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        sudo \
        git \
        curl \
        ca-certificates \
    && groupmod --new-name $USERNAME ubuntu \
    && usermod  --login $USERNAME --move-home --home /home/$USERNAME ubuntu \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
