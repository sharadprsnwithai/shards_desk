# Production slim Dockerfile for Sharad's Option Selling Desk
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

WORKDIR /app

# 1. Install minimal system packages & timezone
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tzdata \
    ca-certificates \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Clone Kronos model library and strip git history/docs to save disk space
RUN git clone --depth 1 https://github.com/shiyu-coder/Kronos.git /app/kronos_lib \
    && rm -rf /app/kronos_lib/.git /app/kronos_lib/figures /app/kronos_lib/examples /app/kronos_lib/finetune* /app/kronos_lib/webui

# 3. Install lightweight CPU-only PyTorch first (~180MB instead of 4.5GB CUDA bloat)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 4. Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y git && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# 5. Copy application source code (respecting .dockerignore)
COPY . /app

# 6. Make entrypoint executable
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["scheduler"]
