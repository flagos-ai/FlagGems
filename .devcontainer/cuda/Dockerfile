FROM nvcr.io/nvidia/pytorch:25.05-py3

# 创建 vscode 用户，UID/GID 由 VSCode 在容器启动时通过 updateRemoteUserUID 自动对齐宿主机用户
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 基础镜像已有 ubuntu:1000:1000，重命名为 vscode 并配置 sudo
RUN usermod -l $USERNAME ubuntu \
    && groupmod -n $USERNAME ubuntu \
    && usermod -d /home/$USERNAME -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# ********************************************************
# * install requirements *
# ********************************************************
RUN pip install --no-cache-dir \
    --index-url https://resource.flagos.net/repository/flagos-pypi-hosted/simple \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host resource.flagos.net \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    "torch>=2.2.0" \
    "flagtree==0.5.0" \
    "scikit-build-core==0.12.2" \
    "pybind11==3.0.3" \
    "ninja" \
    "cmake" \
    "PyYAML==6.0.3" \
    "packaging>=24.0,<27.0" \
    "sqlalchemy==2.0.48" \
    "greenlet>=3.0.0" \
    "numpy==1.26.4"

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

