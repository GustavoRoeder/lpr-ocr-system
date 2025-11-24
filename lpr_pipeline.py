#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from ultralytics import YOLO
import cv2

# Importar predictor
from lpr_ocr.predictor import LPRPredictor

# Configurações
YOLO_MODEL = os.getenv("YOLO_MODEL", "/app/models/yolo_best.pt")
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.5"))

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--help":
        print("=" * 70)
        print("🚗 SISTEMA LPR COMPLETO - YOLO + CRNN")
        print("=" * 70)
        print()
        print("Uso: docker run -v $(pwd)/data:/data lpr-complete /data/imagem.jpg")
        print()
        print("Variáveis de ambiente:")
        print("  CONFIDENCE: Confiança mínima YOLO (padrão: 0.5)")
        print()
        return

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"❌ Imagem não encontrada: {image_path}")
        sys.exit(1)

    print("=" * 70)
    print("🚗 SISTEMA LPR COMPLETO")
    print("=" * 70)
    print()

    # Carregar modelos
    print("📥 Carregando modelos...")
    yolo = YOLO(YOLO_MODEL)
    crnn = LPRPredictor()
    print("✅ Modelos carregados!")
    print()

    # Processar
    print(f"📷 Processando: {Path(image_path).name}")
    print()

    # Detectar
    print("🔍 Detectando placas...")
    results = yolo.predict(image_path, conf=CONFIDENCE, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        print("❌ Nenhuma placa detectada")
        print("=" * 70)
        sys.exit(1)

    print(f"✅ {len(boxes)} placa(s) detectada(s)")
    print()

    # Processar cada placa
    img = cv2.imread(image_path)
    for idx, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])

        print(f"📋 Placa {idx}:")
        print(f"   Bbox: ({x1}, {y1}, {x2}, {y2})")
        print(f"   Confiança: {conf:.2%}")

        # Recortar
        plate = img[y1:y2, x1:x2]
        temp_path = f"/tmp/plate_{idx}.jpg"
        cv2.imwrite(temp_path, plate)

        # Reconhecer
        text = crnn.predict(temp_path)
        print(f"   🚗 Texto: {text}")
        print()

    print("=" * 70)
    print("✅ PROCESSAMENTO CONCLUÍDO")
    print("=" * 70)

if __name__ == "__main__":
    main()
