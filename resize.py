import os
from PIL import Image
from tqdm import tqdm
import traceback

def resize_images_in_folder(input_folder_path, output_folder_path):
    # 출력 폴더가 존재하지 않으면 생성
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    # 이미지 폴더 내 모든 파일을 반복
    for filename in tqdm(os.listdir(input_folder_path)):
        input_file_path = os.path.join(input_folder_path, filename)
        
        # 파일이 이미지인지 확인
        if os.path.isfile(input_file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(input_file_path) as img:
                    # 원래 이미지 크기 가져오기
                    width, height = img.size
                    
                    # 짧은 변이 128px이 되도록 비율 유지하여 리사이징
                    if width < height:
                        new_width = 128
                        new_height = int((height / width) * new_width)
                    else:
                        new_height = 128
                        new_width = int((width / height) * new_height)
                    
                    # 리사이징 수행
                    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # 출력 파일 경로 설정
                    output_file_path = os.path.join(output_folder_path, filename)
                    
                    # 새로운 파일명으로 저장
                    resized_img.save(output_file_path)

            except Exception as e:
                print(f"Error processing file: {filename}")
                print(traceback.format_exc())

# 폴더 경로 설정
input_folder_path = r"D:\Datasets\CarsFlattenRandom"  # 입력 폴더 경로 변경 필요
output_folder_path = r"D:\Datasets\CarsFlattenRandomResized"  # 출력 폴더 경로 변경 필요
resize_images_in_folder(input_folder_path, output_folder_path)
