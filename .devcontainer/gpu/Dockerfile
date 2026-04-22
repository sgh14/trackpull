# =============================================================================
# BASE IMAGE
# Using slim Python - ML frameworks bundle their own CUDA
# =============================================================================
FROM python:3.11-slim

# =============================================================================
# ENVIRONMENT VARIABLES
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/home/vscode/workspace/.venv/bin:${PATH}"

# =============================================================================
# SYSTEM DEPENDENCIES (as root)
# =============================================================================
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# CREATE NON-ROOT USER
# =============================================================================
RUN useradd -m -s /bin/bash -u 1000 vscode
USER vscode
WORKDIR /home/vscode

# =============================================================================
# INSTALL UV PACKAGE MANAGER
# =============================================================================
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# =============================================================================
# WORKSPACE SETUP
# =============================================================================
RUN mkdir -p /home/vscode/workspace /home/vscode/.ssh && chmod 700 /home/vscode/.ssh
WORKDIR /home/vscode/workspace

# =============================================================================
# DEPENDENCIES (uncomment to bake into image)
# =============================================================================
# COPY --chown=vscode:vscode pyproject.toml ./
# RUN uv venv .venv && uv pip install -e ".[dev]"

# =============================================================================
# LABELS FOR TRACEABILITY
# =============================================================================
ARG GIT_SHA=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.revision="${GIT_SHA}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
ENV CONTAINER_GIT_SHA=${GIT_SHA}

CMD ["bash"]
