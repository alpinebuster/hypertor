#!/bin/bash

if ! command -v tcpreplay >/dev/null 2>&1; then
    echo "[INFO] tcpreplay not found, installing from source..."

    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        autogen \
        autoconf \
        automake \
        libtool \
        pkg-config \
        libpcap-dev \
        git

    WORKDIR="/tmp/tcpreplay-build"
    TC_PREPLAY_REPO="https://github.com/appneta/tcpreplay.git"

    echo "[INFO] Preparing build directory: $WORKDIR"
    rm -rf "$WORKDIR"
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"

    echo "[INFO] Cloning tcpreplay..."
    git clone "$TC_PREPLAY_REPO"
    cd tcpreplay

    echo "[INFO] Generating configure script..."
    ./autogen.sh

    echo "[INFO] Configuring..."
    ./configure --disable-local-libopts --enable-test-hexdump

    echo "[INFO] Building..."
    make -j"$(nproc)"

    echo "[INFO] Installing..."
    sudo make install

    echo "[INFO] tcpreplay installation completed!"
else
    echo "[INFO] tcpreplay already installed!"
fi
tcpreplay --version

if ! command -v tcpdump >/dev/null 2>&1; then
    echo "[INFO] tcpdump not found, installing..."

    sudo apt-get install -y tcpdump

    echo "[INFO] tcpdump installed!"
else
    echo "[INFO] tcpdump already installed!"
fi
tcpdump --version

if ! command -v ip >/dev/null 2>&1; then
    echo "[INFO] ip command not found, installing iproute2..."
    sudo apt-get install -y iproute2
    echo "[INFO] iproute2 installed!"
else
    echo "[INFO] ip already installed!"
fi
ip -V || true
