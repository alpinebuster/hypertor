FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

RUN --mount=type=cache,target=/var/cache/apt \
    --mount=type=cache,target=/var/lib/apt \
    apt-get update && apt-get install -y \
    git curl wget build-essential htop \
    libgl1 libglib2.0-0 \
    python3.12 python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /root/workspace

COPY pyproject.toml poetry.lock* /root/workspace/
RUN poetry install --no-root --no-interaction
RUN echo "source $(poetry env info --path)/bin/activate" >> /root/.bashrc

CMD ["/bin/bash"]
