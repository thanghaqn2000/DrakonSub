#!/usr/bin/env bash
# Collect sanitized server inventory for DrakonSub production deploy (Phase 1).
# Run on the target server after SSH login. Do not paste secrets.
set -euo pipefail

echo "=== DrakonSub server inventory ==="
echo "timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo

whoami
hostname
pwd
uname -a
echo

if [[ -f /etc/os-release ]]; then
  cat /etc/os-release
fi
echo

df -h
echo
free -h
echo
echo "nproc: $(nproc)"
lscpu | head -40
echo

echo "docker: $(which docker 2>/dev/null || echo 'not found')"
docker --version 2>/dev/null || true
docker compose version 2>/dev/null || true
echo

echo "ffmpeg: $(which ffmpeg 2>/dev/null || echo 'not found')"
ffmpeg -version 2>/dev/null | head -3 || true
echo "ffprobe: $(which ffprobe 2>/dev/null || echo 'not found')"
python3 --version 2>/dev/null || true
echo

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi: not available"
fi

echo
echo "=== end inventory ==="
