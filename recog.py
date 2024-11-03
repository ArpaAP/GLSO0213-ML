import torch
import os
from PIL import Image
import numpy as np
import platform

# 이미지 전처리 함수 정의
def custom_transform(image):
    image = image.resize((128, 128))
    image = np.array(image).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5  # Normalize to range [-1, 1]
    image = torch.tensor(image).permute(2, 0, 1)  # Change to (C, H, W)
    return image.unsqueeze(0)  # Add batch dimension

# 모델 정의 (기존 모델과 동일하게 유지)
class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc1 = torch.nn.Linear(512 * 8 * 8, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 5)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.nn.functional.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 512 * 8 * 8)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 클래스 정의
classes = ["새", "자동차", "고양이", "개", "물고기"]

# 모델 초기화 및 가중치 불러오기
device = torch.device('mps' if torch.backends.mps.is_available() and platform.system() == 'Darwin' else ('cuda' if torch.cuda.is_available() else 'cpu'))
model = CNNClassifier().to(device)
model.load_state_dict(torch.load('model_11031842.pth', map_location=device))
model.eval()

# 이미지 폴더 경로 입력받기
def classify_images_in_folder(folder_path):
    with open('classification_results.txt', 'w') as result_file:
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            if img_path.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    image = Image.open(img_path).convert('RGB')
                    input_tensor = custom_transform(image).to(device)
                    with torch.no_grad():
                        output = model(input_tensor)
                        _, predicted = torch.max(output, 1)
                        class_name = classes[predicted.item()]
                        result_file.write(f"{img_name}: {class_name}\n")
                        print(f"{img_name}: {class_name} (Class {predicted.item()})")
                except Exception as e:
                    print(f"Error processing {img_name}: {e}")

# 실행
if __name__ == "__main__":
    folder_path = "./test"
    classify_images_in_folder(folder_path)
