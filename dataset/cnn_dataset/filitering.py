import os
import shutil
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

def is_valid_folder(folder_name):
    try:
        num = int(folder_name)
        return 0 <= num <= 7
    except ValueError:
        return False

def filter_and_copy_images(root_directory, output_directory, min_size, max_size):
    matching_files_count = 0
    total_png_count = 0
    folder_stats = OrderedDict()

    # 먼저 0-8 폴더를 순서대로 생성
    for i in range(9):
        folder_stats[str(i)] = {"matching": 0, "total": 0}

    for root, dirs, files in tqdm(list(os.walk(root_directory)), desc="폴더 처리 중"):
        current_folder = os.path.basename(root)
        if root == root_directory or is_valid_folder(current_folder):
            folder_matching = 0
            folder_total = 0
            for filename in files:
                if filename.lower().endswith('.png'):
                    folder_total += 1
                    file_path = os.path.join(root, filename)
                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                            if min_size <= width <= max_size and min_size <= height <= max_size:
                                matching_files_count += 1
                                folder_matching += 1
                                
                                # 새로운 폴더 경로 생성
                                relative_path = os.path.relpath(root, root_directory)
                                new_folder_path = os.path.join(output_directory, relative_path)
                                os.makedirs(new_folder_path, exist_ok=True)
                                
                                # 파일 복사
                                new_file_path = os.path.join(new_folder_path, filename)
                                shutil.copy2(file_path, new_file_path)
                    except IOError:
                        print(f"Error opening {file_path}")
            
            if current_folder in folder_stats:
                folder_stats[current_folder]["matching"] += folder_matching
                folder_stats[current_folder]["total"] += folder_total
            total_png_count += folder_total
        else:
            dirs.clear()

    return matching_files_count, total_png_count, folder_stats

# 사용 예시
root_directory = './'  # 현재 디렉토리
output_directory = './filtered_images'  # 필터링된 이미지를 저장할 디렉토리
min_size = 64  # 원하는 최소 크기
max_size = 120  # 원하는 최대 크기

matching_count, total_count, folder_stats = filter_and_copy_images(root_directory, output_directory, min_size, max_size)

print(f"\n필터링 크기 범위: {min_size}x{min_size} 에서 {max_size}x{max_size}")
print(f"\n필터링된 이미지 저장 위치: {output_directory}")
print("\n폴더별 통계:")
for folder, stats in folder_stats.items():
    matching = stats["matching"]
    total = stats["total"]
    if total > 0:
        percentage = (matching / total) * 100
        print(f"폴더 {folder}: {matching}/{total} 파일 조건 만족 ({percentage:.2f}%)")
    else:
        print(f"폴더 {folder}: PNG 파일 없음")

print(f"\n총 PNG 파일 수: {total_count}")
print(f"크기 조건을 만족하는 총 PNG 파일 수: {matching_count}")
if total_count > 0:
    overall_percentage = (matching_count / total_count) * 100
    print(f"전체 만족 비율: {overall_percentage:.2f}%")
else:
    print("처리된 PNG 파일이 없습니다.")