FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch CPU first
RUN pip install --no-cache-dir \
    "torch==2.2.2+cpu" \
    "torchaudio==2.2.2+cpu" \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install everything in ONE shot with exact pins so nothing conflicts
RUN pip install --no-cache-dir \
    "huggingface_hub==0.23.4" \
    "starlette==0.41.3" \
    "fastapi==0.115.12" \
    "gradio==4.44.1" \
    "gradio-client==1.3.0" \
    "uvicorn==0.29.0" \
    "aiofiles==23.2.1" \
    "httpx==0.27.2" \
    "pydantic==2.7.4" \
    "python-multipart==0.0.9" \
    "websockets==11.0.3" \
    "orjson==3.10.7" \
    "tomlkit==0.12.0" \
    "typer==0.12.5" \
    "ffmpy==0.3.2" \
    "pydub==0.25.1" \
    "pillow==10.4.0" \
    "markupsafe==2.1.5" \
    "semantic-version==2.10.0" \
    "pandas==2.2.2" \
    "ruff==0.5.7" \
    "importlib-resources==6.4.4" \
    "deepfilternet" \
    "noisereduce" \
    "soundfile" \
    "imageio-ffmpeg" \
    "scipy" \
    "numpy<2.0"

COPY app.py .
COPY audio_processor.py .

EXPOSE 10000

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=10000
ENV PORT=10000

CMD ["python", "app.py"]
