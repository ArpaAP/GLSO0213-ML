import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import platform
import torch.nn.functional as F

# 이미지 전처리 함수 정의 (기존과 동일)
def custom_transform(image):
    image = image.resize((128, 128))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5  # Normalize to range [-1, 1]
    image = torch.tensor(image).permute(2, 0, 1)  # Change to (C, H, W)
    return image

# 모델 정의 (기존과 동일)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.fc2 = nn.Linear(512, 5)  # 5개의 클래스 (개, 고양이, 자동차, 물고기, 새)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 모델 초기화 및 로드
device = torch.device('mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin' else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = CNNClassifier().to(device)
model_path = 'cnn_classifier.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 클래스 이름 로드 (데이터셋 디렉토리의 서브폴더 이름 사용)
root_dir = './datasets'
class_names = sorted(os.listdir(root_dir))

# 이미지 예측 함수 정의
def predict_image(image_path):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')
    image = custom_transform(image)
    image = image.unsqueeze(0)  # 배치 차원 추가
    image = image.to(device)
    
    # 모델을 통해 예측 수행
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_class = class_names[predicted.item()]
    return predicted_class

# 실행 부분
if __name__ == '__main__':
    image_path = input('이미지 경로를 입력하세요: ')
    predicted_class = predict_image(image_path)
    print(f'예측된 클래스: {predicted_class}')
