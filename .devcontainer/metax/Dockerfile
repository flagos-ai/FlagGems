FROM registry.metax-tech.com/metax/pytorch:2.8.0-metax3.5.3.9-py310

ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# ********************************************************
# * install requirements *
# ********************************************************
RUN pip install --no-cache-dir \
    --index-url https://resource.flagos.net/repository/flagos-pypi-metax/simple \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host resource.flagos.net \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    "flagtree==0.5.1+metax3.0" \
    "scikit-build-core==0.12.2" \
    "pybind11==3.0.3" \
    "ninja" \
    "cmake"

USER $USERNAME
