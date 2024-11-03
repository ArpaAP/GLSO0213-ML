import os
import shutil
import random

def copy_random_files(src_folder, dest_folder, num_files):
    # 소스 폴더에 있는 모든 파일 목록을 가져옵니다.
    all_files = os.listdir(src_folder)

    print(len(all_files))

    # 선택하려는 파일 수가 전체 파일 수보다 많을 경우 조정합니다.
    num_files = min(num_files, len(all_files))
    
    # 파일을 랜덤하게 선택합니다.
    selected_files = random.sample(all_files, num_files)
    
    # 선택된 파일을 대상 폴더로 복사합니다.
    for file_name in selected_files:
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy(src_path, dest_path)
        print(f"Copied: {file_name} to {dest_folder}")

src_folder = 'D:\\Datasets\\BirdsFlatten'  # 소스 폴더 경로
dest_folder = 'D:\\Datasets\\BirdsRandom'  # 대상 폴더 경로
copy_random_files(dest_folder, dest_folder, 10000)