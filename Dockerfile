FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    wget \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY requirements.txt .

# Instalar dependências Python
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ultralytics opencv-python

# Copiar código
COPY lpr_ocr/ ./lpr_ocr/

# Criar diretórios para modelos
RUN mkdir -p models

# Baixar modelos (serão substituídos por volumes em produção)
# YOLO será montado via volume
# CRNN será montado via volume

# Copiar script de pipeline
COPY lpr_pipeline.py .

# Diretório para imagens
RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "lpr_pipeline.py"]
CMD ["--help"]
