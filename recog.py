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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # 필터 수 증가
        self.bn1 = nn.BatchNorm2d(64)  # 배치 정규화 추가
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 필터 수 증가
        self.bn2 = nn.BatchNorm2d(128)  # 배치 정규화 추가
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 필터 수 증가
        self.bn3 = nn.BatchNorm2d(256)  # 배치 정규화 추가
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # 추가 합성곱 계층
        self.bn4 = nn.BatchNorm2d(512)  # 배치 정규화 추가
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)  # 드롭아웃 추가 (50% 확률)
        self.fc1 = nn.Linear(512 * 8 * 8, 1024)  # 출력 채널 수 및 뉴런 수 증가
        self.fc2 = nn.Linear(1024, 512)  # 추가 전결합 계층
        self.fc3 = nn.Linear(512, 5)  # 최종 클래스 수 유지

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # 드롭아웃 적용
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.fc3(x)
        return x

# 모델 초기화 및 로드
device = torch.device('mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin' else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = CNNClassifier().to(device)
model_path = 'model.pth'
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
    folder_path = "./test"
    output_file = 'results.txt'

    with open(output_file, 'w') as f:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(folder_path, filename)
                try:
                    predicted_class = predict_image(image_path)
                    f.write(f'{filename}: {predicted_class}\n')
                    print(f'{filename}: {predicted_class}')
                except Exception as e:
                    print(f'이미지 처리 중 오류 발생 - {filename}: {e}')
                    f.write(f'{filename}: Error\n')

    print(f'\n결과가 {output_file} 파일에 저장되었습니다.')
