import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import platform
from tqdm import tqdm
from PIL import Image
import os
import numpy as np

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 0.001
epochs = 10
train_ratio = 0.8  # 학습 데이터와 테스트 데이터 비율 설정
image_size = (128, 128)  # 이미지 크기 설정

# 사용자 정의 데이터셋 클래스 정의
class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 이미지 전처리 함수 정의
def custom_transform(image):
    # 이미지 크기 조정 및 텐서 변환
    image = image.resize(image_size)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5  # Normalize to range [-1, 1]
    image = torch.tensor(image).permute(2, 0, 1)  # Change to (C, H, W)
    return image

# 데이터셋 준비
dataset = CustomImageDataset(root_dir='./datasets', transform=custom_transform)
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의 (더 큰 모델로 수정, 드롭아웃 및 배치 정규화 추가)
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

# 모델 초기화
device = torch.device('mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin' else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = CNNClassifier().to(device)

# 기존에 저장된 모델이 있으면 불러오기
model_path = 'model.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print(f'Model loaded from {model_path}')
except FileNotFoundError:
    print(f'No saved model found at {model_path}, training from scratch.')

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)  # 학습률 스케줄러 추가

# 모델 학습
def train():
    model.train()
    for epoch in range(epochs):
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 학습률 스케줄러 업데이트
        scheduler.step()

    # 모델 저장
    torch.save(model.state_dict(), 'model.pth')

# 모델 테스트
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

# 실행
if __name__ == "__main__":
    train()
    test()
