import os
import shutil
import random
from tqdm import tqdm

def copy_random_files(src_folder, dest_folder, num_files):
    # 대상 폴더가 존재하지 않으면 생성합니다.
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # 소스 폴더에 있는 모든 파일 목록을 가져옵니다.
    all_files = os.listdir(src_folder)

    print(len(all_files))

    # 선택하려는 파일 수가 전체 파일 수보다 많을 경우 조정합니다.
    num_files = min(num_files, len(all_files))
    
    # 파일을 랜덤하게 선택합니다.
    selected_files = random.sample(all_files, num_files)
    
    # 선택된 파일을 대상 폴더로 복사합니다.
    for file_name in tqdm(selected_files):
        src_path = os.path.join(src_folder, file_name)
        dest_path = os.path.join(dest_folder, file_name)
        shutil.copy(src_path, dest_path)

src_folder = r'D:\\Datasets\\CarsFlatten'  # 소스 폴더 경로
dest_folder = r'D:\\Datasets\\CarsFlattenRandom'  # 대상 폴더 경로
copy_random_files(src_folder, dest_folder, 20000)