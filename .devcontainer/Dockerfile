FROM mcr.microsoft.com/devcontainers/base:1-ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        clang-18 \
        cmake \
        curl \
        gdb \
        git \
        git-lfs \
        jq \
        libcurl4-openssl-dev \
        libopenblas-dev \
        libomp-18-dev \
        lld-18 \
        ninja-build \
        numactl \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        shellcheck \
        time \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir pip==25.2 uv==0.8.22 \
    && chown -R vscode:vscode /opt/venv \
    && install -d -o vscode -g vscode \
        /home/vscode/.cache/huggingface \
        /home/vscode/.cache/pip

COPY requirements.lock /opt/embedded-jev/requirements.lock

RUN /opt/venv/bin/python -m pip install --no-cache-dir --require-hashes \
        -r /opt/embedded-jev/requirements.lock \
    && /opt/venv/bin/python -m pip check \
    && chown -R vscode:vscode /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    CC=clang-18 \
    CXX=clang++-18 \
    HF_HOME=/home/vscode/.cache/huggingface \
    XDG_CACHE_HOME=/home/vscode/.cache \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1