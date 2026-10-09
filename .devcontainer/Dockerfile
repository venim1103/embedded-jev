FROM mcr.microsoft.com/devcontainers/base:1-ubuntu-24.04

ENV DEBIAN_FRONTEND=noninteractive

# Host roots from prepare-build.sh, e.g. TLS-inspection issuers; empty keeps defaults.
COPY host-ca-certificates.crt /tmp/host-ca-certificates.crt
RUN set -eu; \
    if grep -q -- '-----BEGIN CERTIFICATE-----' /tmp/host-ca-certificates.crt; then \
        dir=/usr/local/share/ca-certificates/host; \
        install -d "$dir"; \
        awk -v dir="$dir" '{ sub(/\r$/, "") } /-----BEGIN CERTIFICATE-----/ { n++; file = sprintf("%s/host-%04d.crt", dir, n) } file { print > file } /-----END CERTIFICATE-----/ { close(file); file = "" }' /tmp/host-ca-certificates.crt; \
        for cert in "$dir"/*.crt; do \
            openssl x509 -noout -in "$cert" 2>/dev/null || rm -f "$cert"; \
        done; \
        update-ca-certificates; \
    fi; \
    rm -f /tmp/host-ca-certificates.crt

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    NODE_EXTRA_CA_CERTS=/etc/ssl/certs/ca-certificates.crt

RUN sed -i 's|^URIs: http://|URIs: https://|' /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
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