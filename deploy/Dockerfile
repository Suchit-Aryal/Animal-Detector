# Hugging Face Space (Docker SDK). CPU only. Serves on port 7860.
# Built from the folder that deploy/build_space.sh creates (deploy/hf_space/).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOSTED=1 \
    YOLO_OFFLINE=1 \
    YOLO_CONFIG_DIR=/tmp/Ultralytics \
    INSTANCE_DIR=/tmp/animal-watch \
    HOME=/home/user

# libgl1 and libglib2.0-0: OpenCV's runtime libraries. ffmpeg: re-encodes results to H.264 for browsers.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 user

WORKDIR /app

# 1. CPU-only torch and torchvision first, from the CPU index. The default PyPI wheels are built for CUDA
#    and fail on a CPU torch ("operator torchvision::nms does not exist"). Then a check that they work together.
# 2. Then the rest, pinned. ultralytics finds torchvision already installed and does not replace it.
COPY requirements-container.txt /app/requirements-container.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.1 \
 && python -c "import torch, torchvision; torch.ops.torchvision.nms(torch.zeros(2, 4), torch.zeros(2), 0.5); print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'ok')" \
 && pip install -r /app/requirements-container.txt \
 && python -c "import torchvision, torch; torch.ops.torchvision.nms(torch.zeros(2, 4), torch.zeros(2), 0.5); print('torchvision after install', torchvision.__version__, 'ok')"

# Application, the animal model and the person model. Baked in at build time; nothing is downloaded at run time.
COPY web/ /app/web/
COPY results/weights/best.pt /app/results/weights/best.pt
COPY yolov8n.pt /app/yolov8n.pt
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh \
 && mkdir -p /tmp/animal-watch /tmp/Ultralytics \
 && chown -R 1000:1000 /tmp/animal-watch /tmp/Ultralytics

USER user
EXPOSE 7860

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:7860/api/summary', timeout=5).status == 200 else 1)"

CMD ["/app/start.sh"]
