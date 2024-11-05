import torch
import torch.nn as nn
import numpy as np

from PIL import Image
import os
import platform
import torch.nn.functional as F

# 이미지 전처리 함수 정의
def custom_transform(image):
    image = image.resize((128, 128))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5  # Normalize to range [-1, 1]
    image = torch.tensor(image).permute(2, 0, 1)  # Change to (C, H, W)
    return image

# 모델 정의
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.15)
        self.fc1 = nn.Linear(256 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 256 * 16 * 16)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# 모델 초기화 및 로드
device = torch.device('mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin' else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = CNNClassifier().to(device)
model_path = 'model.pth'
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# 클래스 이름 로드 (데이터셋 디렉토리의 서브폴더 이름 사용)
root_dir = './datasets'
class_names = sorted(os.listdir(root_dir))
print(class_names)

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
        for filename in sorted(os.listdir(folder_path)):
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
