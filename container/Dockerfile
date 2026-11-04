# container/Dockerfile
# Base: Ubuntu22.04，适配Iluvatar SDK、FlagTree编译器
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# 基础系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    ca-certificates \
    vim \
    libssl-dev \
    pkg-config \
    libgtest-dev \
    google-perftools \
    && rm -rf /var/lib/apt/lists/*

# Python3.10 + pip
RUN apt-get update && apt-get install -y python3.10 python3.10-dev python3-pip python3.10-venv && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 安装gtest编译（系统gtest是源码包，需要编译）
RUN cd /usr/src/gtest && cmake CMakeLists.txt && make -j$(nproc) && cp lib/*.so /usr/lib/

# PyPI基础包
RUN pip install --upgrade pip setuptools wheel && \
    pip install pybind11==2.11.1 numpy pytest pytest-xdist

# 【重点】Iluvatar Runtime SDK（这里为占位，部署时替换成内部离线包地址）
# 实际场景：内部registry挂载/离线tar包拷贝
ENV ILUVATAR_SDK_ROOT=/opt/iluvatar
RUN mkdir -p ${ILUVATAR_SDK_ROOT}
# COPY ./offline_pkg/iluvatar-sdk.tar.gz /tmp/
# RUN tar -xf /tmp/iluvatar-sdk.tar.gz -C ${ILUVATAR_SDK_ROOT} --strip-components=1
ENV PATH="${ILUVATAR_SDK_ROOT}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${ILUVATAR_SDK_ROOT}/lib64:${ILUVATAR_SDK_ROOT}/lib:${LD_LIBRARY_PATH}"

# 【重点】FlagTree 编译器（Triton for Iluvatar，内部定制编译器）
ENV FLAGTREE_ROOT=/opt/flagtree
RUN mkdir -p ${FLAGTREE_ROOT}
# COPY ./offline_pkg/flagtree.tar.gz /tmp/
# RUN tar -xf /tmp/flagtree.tar.gz -C ${FLAGTREE_ROOT} --strip-components=1
ENV PATH="${FLAGTREE_ROOT}/bin:${PATH}"

# PyTorch & Triton（Iluvatar适配版本，内部源优先）
RUN pip install torch==2.3.1 triton==2.2.0

WORKDIR /workspace/FlagGems

# 默认启动shell
CMD ["/bin/bash"]
