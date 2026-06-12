# FROM harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.10
FROM harbor.baai.ac.cn/flaggems/hygon-flaggems-test-bw1000:20260526

# Create the vscode user; UID/GID is automatically aligned to the host user by VSCode at container start via updateRemoteUserUID
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# The base image runs as root; directly create the vscode user and configure sudo
RUN apt-get update \
    && apt-get install -y sudo \
    && groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# ********************************************************
# * install requirements *
# ********************************************************
RUN pip install --no-cache-dir \
    --index-url https://resource.flagos.net/repository/flagos-pypi-hygon/simple \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host resource.flagos.net \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    "torch==2.9.0+das.opt1.dtk2604" \
    "flagtree==0.5.1+hcu3.1" \
    "scikit-build-core==0.11" \
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
