FROM python:3.9-slim-bullseye

WORKDIR /opt/tpuaas

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates gnupg libusb-1.0-0 \
    && echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" > /etc/apt/sources.list.d/coral-edgetpu.list \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y --no-install-recommends libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --extra-index-url https://google-coral.github.io/py-repo/ \
    "pycoral~=2.0" tflite-runtime fastapi "uvicorn[standard]" python-multipart pillow "numpy<2" httpx

COPY scripts/download_models.sh /opt/tpuaas/scripts/download_models.sh
RUN bash /opt/tpuaas/scripts/download_models.sh /opt/tpuaas/models

COPY engine.py server.py client.py dashboard.html AGENT_GUIDE.md AGENT_BOARD.md .env.example /opt/tpuaas/

ENV TPUAAS_BASE_DIR=/opt/tpuaas
ENV TPUAAS_PORT=8090
EXPOSE 8090

CMD ["python", "/opt/tpuaas/server.py"]
