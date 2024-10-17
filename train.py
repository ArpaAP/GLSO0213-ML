import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import platform

# 하이퍼파라미터 설정
batch_size = 32
learning_rate = 0.001
epochs = 10

# 데이터셋 준비
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.ImageFolder(root='./datasets/train', transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root='./datasets/test', transform=data_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 모델 정의
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

# 모델 초기화
device = torch.device('mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin' else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = CNNClassifier().to(device)

# 기존에 저장된 모델이 있으면 불러오기
model_path = 'cnn_classifier.pth'
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f'Model loaded from {model_path}')
except FileNotFoundError:
    print(f'No saved model found at {model_path}, training from scratch.')

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 모델 학습
def train():
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)

            # 옵티마이저 초기화
            optimizer.zero_grad()

            # 순전파 + 역전파 + 최적화
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 통계 출력
            running_loss += loss.item()
            if i % 100 == 99:  # 매 100 미니배치마다 출력
                print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

    # 모델 저장
    torch.save(model.state_dict(), 'cnn_classifier.pth')

# 모델 테스트
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
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
