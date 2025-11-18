# 🚗 LPR OCR - Brazilian License Plate Recognition

Sistema de reconhecimento de placas veiculares brasileiras usando modelo CRNN.

## 📊 Performance
- **Acurácia**: 87%+ (sequência completa)
- **Velocidade**: ~600 imagens/segundo (GPU)
- **Modelo**: CRNN com 5.77M parâmetros (67MB)

## 🚀 Instalação

```bash
git clone https://github.com/GustavoRoeder/lpr-ocr-system.git
cd lpr-ocr-system
pip install -r requirements.txt

# Baixar modelo
wget https://github.com/GustavoRoeder/lpr-ocr-system/releases/download/v1.0/best_model.pth -O lpr_ocr/models/best_model.pth
from lpr_ocr import LPRPredictor

predictor = LPRPredictor( )
plate = predictor.predict("imagem.jpg")
print(f"Placa: {plate}")

Execute!
