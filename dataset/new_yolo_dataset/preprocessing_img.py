import os
import shutil

# 기존 이미지 경로 (전체 이미지가 있는 곳)
img_dir = "/Users/hyunoh/Documents/vscode/embedded/traffic_sign_recognition/dataset/new_yolo_dataset/train/img"

# 라벨이 저장된 경로
label_dir = "/Users/hyunoh/Documents/vscode/embedded/traffic_sign_recognition/dataset/new_yolo_dataset/labels_yolo"

# 새 이미지 복사 경로
filtered_img_dir = "/Users/hyunoh/Documents/vscode/embedded/traffic_sign_recognition/dataset/new_yolo_dataset/images_yolo"
os.makedirs(filtered_img_dir, exist_ok=True)

# 라벨이 존재하는 이미지 이름 추출
for label_file in os.listdir(label_dir):
    if not label_file.endswith(".txt"):
        continue

    # 라벨 파일 이름에서 .txt 제거 → 이미지 확장자까지 포함된 이름이 나옴
    img_filename = label_file[:-4]  # 예: "0003.png"

    img_path = os.path.join(img_dir, img_filename)
    if os.path.exists(img_path):
        shutil.copy(img_path, os.path.join(filtered_img_dir, img_filename))
    else:
        print(f"⚠️ 이미지 없음: {img_path}")
