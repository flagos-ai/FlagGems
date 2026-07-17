# Base image produced by build-infra (see .devcontainer/build-infra/runtime/Containerfile).
# Naming convention: flagos-runtime-{vendor}-{backend}:latest
# Override this arg to pin to a specific registry or tag without editing this file.
ARG IMAGE=flagos-runtime-nvidia-cuda13.3:latest

FROM ${IMAGE}

# ── Dev-container user ─────────────────────────────────────────
# VS Code expects a non-root user named vscode; UID/GID are re-aligned to
# the host user at container start via updateRemoteUserUID.
ARG USERNAME=vscode

# The nvidia runtime image is based on ubuntu and runs as root.
# It already ships with ubuntu:1000 — rename that user/group to vscode
# and configure passwordless sudo rather than creating a duplicate GID.
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

# The runtime image installs uv-managed Python under /root/.local/share/uv/,
# making it inaccessible to non-root users.  Reinstall to a shared location
# (/opt/uv-python), update pyvenv.cfg, and repoint the /flagos/bin/python*
# symlinks so the venv works for all users (including vscode after
# updateRemoteUserUID).
RUN UV_PYTHON_VERSION=$(awk '/^version_info/{print $3}' /flagos/pyvenv.cfg) \
    && UV_PYTHON_INSTALL_DIR=/opt/uv-python \
       uv python install "${UV_PYTHON_VERSION}" --python-preference only-managed -q \
    && NEW_BIN=$(ls -d /opt/uv-python/cpython-${UV_PYTHON_VERSION}*/bin 2>/dev/null | head -1) \
    && NEW_PYTHON="${NEW_BIN}/python3.${UV_PYTHON_VERSION#*.}" \
    && sed -i "s|^home = .*|home = ${NEW_BIN}|" /flagos/pyvenv.cfg \
    && ln -sf "${NEW_PYTHON}" /flagos/bin/python \
    && chmod -R a+rX /opt/uv-python \
    && chown -R $USERNAME:$USERNAME /flagos

USER $USERNAME
