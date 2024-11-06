import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import platform
from tqdm import tqdm
import os
import numpy as np

# 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
epochs = 12
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15  # 학습 데이터와 테스트 데이터 비율 설정
image_size = (128, 128)  # 이미지 크기 설정
patience = 5  # 조기 종료를 위한 검증 손실 개선 횟수 제한

# 이미지 전처리 함수 정의 (torchvision 사용)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋 준비
dataset = ImageFolder(root='./datasets', transform=transform)

# 데이터셋 준비 (훈련, 검증, 테스트로 분리)
train_size = int(train_ratio * len(dataset))
val_size = int(val_ratio * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True)

# 모델 정의 (더 큰 모델로 수정, 드롭아웃 및 배치 정규화 추가)
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.1)
        self.fc1 = nn.Linear(512 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

# 모델 초기화
device = torch.device('mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin' else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = CNNClassifier().to(device)

# 클래스 가중치 계산
class_counts = np.bincount([label for _, label in dataset.imgs])
# class_weights = torch.tensor([len(dataset) / count if count > 0 else 0 for count in class_counts], dtype=torch.float).to(device)
class_weights = torch.tensor([5.2674, 4.8724, 5.0905, 4.7624, 5.9443], dtype=torch.float).to(device)

# 기존에 저장된 모델이 있으면 불러오기
model_path = 'model.pth'
if __name__ == "__main__":  # 메인 프로세스에서만 실행
    try:
        print(class_weights)

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        print(f'Model loaded from {model_path}')
    except FileNotFoundError:
        print(f'No saved model found at {model_path}, training from scratch.')

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)  # 학습률 스케줄러 추가

best_val_loss = np.inf
patience_counter = 0

# 모델 학습
def train():
    global best_val_loss, patience_counter
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 손실 및 정확도 계산
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # 에포크별 손실과 정확도 출력
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
        
        # 검증 손실 확인 후 조기 종료
        val_loss = validate()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'model.pth')  # 모델 저장
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break
        
        # 학습률 스케줄러 업데이트
        scheduler.step()

# 모델 검증
def validate():
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    return running_loss / len(val_loader)

# 모델 테스트
def test():
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 실행
if __name__ == "__main__":
    train()
    test()
