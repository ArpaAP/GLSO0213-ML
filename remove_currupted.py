import os
from PIL import Image
from tqdm import tqdm

def remove_corrupted_images(directory):
    for filename in tqdm(os.listdir(directory)):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with Image.open(filepath) as img:
                    img.verify()  # 이미지가 손상되었는지 확인
            except (IOError, SyntaxError):
                os.remove(filepath)

# 사용 예시
directory = r'D:\\Datasets\\CarsFlatten'
remove_corrupted_images(directory)