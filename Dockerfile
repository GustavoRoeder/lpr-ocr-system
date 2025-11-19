# Dockerfile para LPR OCR System (CPU)
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y wget libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY lpr_ocr/ ./lpr_ocr/
RUN mkdir -p lpr_ocr/models
RUN wget -q https://github.com/GustavoRoeder/lpr-ocr-system/releases/download/v1.0/best_model.pth -O lpr_ocr/models/best_model.pth
RUN mkdir -p /data
ENV PYTHONUNBUFFERED=1
COPY docker-entrypoint.sh /
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["--help"]
