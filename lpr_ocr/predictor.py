#!/usr/bin/env python3
"""
Script de inferência LPR com modelo CRNN treinado
Uso: python3 lpr_inference_final.py <imagem.jpg>
Retorna: Texto da placa reconhecida
"""
import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import sys
from pathlib import Path

# Caminho do modelo (relativo ou absoluto)
if Path("/home/testelpr/lpr_models/best_model.pth").exists():
    MODEL_PATH = Path("/home/testelpr/lpr_models/best_model.pth")
else:
    MODEL_PATH = Path(__file__).parent / "models" / "best_model.pth"
IMG_HEIGHT = 32
IMG_WIDTH = 128
HIDDEN_SIZE = 256

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
NUM_CLASSES = len(CHARS) + 1

class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=256):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d(2, 2),
        )
        self.rnn = nn.LSTM(512 * 2, hidden_size, 2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        conv = conv.permute(0, 3, 1, 2).reshape(b, w, c * h)
        return self.fc(self.rnn(conv)[0])

class LPRPredictor:
    """Classe para predição de placas com modelo CRNN"""

    def __init__(self, model_path=MODEL_PATH, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CRNN(NUM_CLASSES, HIDDEN_SIZE).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def decode_ctc(self, pred):
        """Decodifica predição CTC"""
        result = []
        prev = -1
        for p in pred:
            if p != 0 and p != prev:
                if p in IDX_TO_CHAR:
                    result.append(IDX_TO_CHAR[p])
            prev = p
        return ''.join(result)

    def predict(self, image_path):
        """Prediz placa de uma imagem (path)"""
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        return self._predict_from_array(img)

    def predict_from_array(self, img_array):
        """Prediz placa de um array numpy (para integração com API)"""
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

        return self._predict_from_array(img_array)

    def _predict_from_array(self, img_gray):
        """Predição interna"""
        img_tensor = self.transform(img_gray).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_tensor)
            _, pred = output.max(2)
            pred = pred.squeeze(0).cpu().numpy()

        return self.decode_ctc(pred)

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 lpr_inference_final.py <imagem.jpg>", file=sys.stderr)
        sys.exit(1)

    image_path = sys.argv[1]

    if not Path(image_path).exists():
        print(f"Erro: Arquivo não encontrado: {image_path}", file=sys.stderr)
        sys.exit(1)

    predictor = LPRPredictor()
    result = predictor.predict(image_path)

    if result:
        print(result)
    else:
        print("Erro ao processar imagem", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
