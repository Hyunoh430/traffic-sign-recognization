import os

src_label_folder = r'C:\Users\2019124074\Desktop\embedded\traffic-sign-recognization\dataset\new_yolo_dataset\labels_yolo'

# 변환된 라벨을 저장할 새 폴더 (원하면 아래 경로도 수정 가능)
dst_label_folder = r'C:\Users\2019124074\Desktop\embedded\traffic-sign-recognization\dataset\new_yolo_dataset\converted_labels'

os.makedirs(dst_label_folder, exist_ok=True)

original_classes = [0, 1, 2, 3, 4, 5, 6, 7]
target_class = 3

for filename in os.listdir(src_label_folder):
    if filename.endswith('.txt'):
        src_path = os.path.join(src_label_folder, filename)
        dst_path = os.path.join(dst_label_folder, filename)

        with open(src_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            if class_id in original_classes:
                parts[0] = str(target_class)
            new_lines.append(' '.join(parts))

        with open(dst_path, 'w') as f:
            f.write('\n'.join(new_lines))

print("✅ 기존 라벨은 유지하고, 변환된 라벨은 'converted_labels' 폴더에 저장했습니다.")
