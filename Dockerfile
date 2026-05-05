FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install poetry
RUN poetry config virtualenvs.create false

WORKDIR /app
COPY pyproject.toml poetry.lock* .
RUN poetry install --no-interaction --no-ansi --no-root
