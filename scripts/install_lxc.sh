#!/usr/bin/env bash
set -euo pipefail

# Turnkey TPUaaS Installer for Debian 12 (Bookworm) / Ubuntu 22.04+ / Proxmox LXC
# Automatically resolves the broken Google Coral apt repository and Python 3.9 pycoral dependency
# using uv + direct libedgetpu1-std extraction.

INSTALL_DIR="${TPUAAS_BASE_DIR:-/opt/tpuaas}"
echo "==> Installing TPUaaS into ${INSTALL_DIR}..."

apt-get update
apt-get install -y curl ca-certificates git libusb-1.0-0

# 1. Install libedgetpu.so.1.0 directly if missing
if [ ! -f /usr/lib/x86_64-linux-gnu/libedgetpu.so.1 ]; then
  echo "==> Installing libedgetpu.so.1.0 for x86_64..."
  TMP_DIR=$(mktemp -d)
  git clone --depth 1 https://github.com/google-coral/libedgetpu.git "$TMP_DIR/libedgetpu" || true
  if [ -f "$TMP_DIR/libedgetpu/direct/k8/libedgetpu.so.1.0" ]; then
    cp "$TMP_DIR/libedgetpu/direct/k8/libedgetpu.so.1.0" /usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0
    ln -sf /usr/lib/x86_64-linux-gnu/libedgetpu.so.1.0 /usr/lib/x86_64-linux-gnu/libedgetpu.so.1
    ldconfig
  fi
  rm -rf "$TMP_DIR"
fi

# 2. Install uv for standalone Python 3.9 management
if ! command -v uv >/dev/null 2>&1 && [ ! -f "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# 3. Copy files and create Python 3.9 virtualenv
mkdir -p "$INSTALL_DIR/models" "$INSTALL_DIR/data"
cp -f engine.py server.py client.py dashboard.html AGENT_GUIDE.md AGENT_BOARD.md "$INSTALL_DIR/"
if [ ! -f "$INSTALL_DIR/.env" ] && [ -f .env.example ]; then
  cp .env.example "$INSTALL_DIR/.env"
fi

uv venv --python 3.9 "$INSTALL_DIR/.venv"
uv pip install --python "$INSTALL_DIR/.venv/bin/python" \
  --extra-index-url https://google-coral.github.io/py-repo/ \
  pycoral~=2.0 tflite-runtime fastapi "uvicorn[standard]" python-multipart pillow "numpy<2" httpx

# 4. Download models
bash scripts/download_models.sh "$INSTALL_DIR/models"

# 5. Create and enable systemd service
cat >/etc/systemd/system/tpuaas.service <<EOF
[Unit]
Description=TPUaaS — Coral EdgeTPU-as-a-Service
After=network.target

[Service]
Type=simple
WorkingDirectory=${INSTALL_DIR}
EnvironmentFile=-${INSTALL_DIR}/.env
Environment="TPUAAS_BASE_DIR=${INSTALL_DIR}"
ExecStart=${INSTALL_DIR}/.venv/bin/python ${INSTALL_DIR}/server.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now tpuaas.service
echo "==> TPUaaS is live on http://0.0.0.0:8090"
