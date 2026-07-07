FROM python:3.11-slim-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates unzip \
    && curl -fsSL https://deno.land/install.sh | DENO_INSTALL=/usr/local sh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY setup.py ./
COPY auto_subtitle ./auto_subtitle

# CPU-only torch avoids multi-GB CUDA wheels that exhaust small VPS disks.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -e .

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/data/jobs /app/data/whisper-cache \
    && chown -R appuser:appuser /app/data

USER appuser

ENV DRAKONSUB_HOST=0.0.0.0 \
    DRAKONSUB_PORT=8000 \
    DRAKONSUB_JOBS_ROOT=/app/data/jobs \
    XDG_CACHE_HOME=/app/data/whisper-cache

EXPOSE 8000

CMD ["drakonsub-web"]
