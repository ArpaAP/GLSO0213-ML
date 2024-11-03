import os
import shutil

def flatten_and_copy(src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    file_count = {}

    for root, _, files in os.walk(src_folder):
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_name = file

            # Check if the file name already exists in the destination folder
            if dest_file_name in file_count:
                file_count[dest_file_name] += 1
                name, ext = os.path.splitext(file)
                dest_file_name = f"{name}_{file_count[dest_file_name]}{ext}"
            else:
                file_count[dest_file_name] = 0

            dest_file_path = os.path.join(dest_folder, dest_file_name)
            shutil.copy2(src_file_path, dest_file_path)

# 예시 사용법
src_folder = 'D:\\Datasets\\Birds\\train'  # 소스 폴더 경로
dest_folder = 'D:\\Datasets\\BirdsFlatten'  # 대상 폴더 경로

flatten_and_copy(src_folder, dest_folder)
