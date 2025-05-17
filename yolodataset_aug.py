import os
import cv2
import albumentations as A
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 증강 개수 설정
AUGMENT_PER_IMAGE = 3
# YOLO 데이터셋 경로 설정
# 내 실제 YOLO 데이터 경로
BASE_PATH = r'C:\Users\2019124074\Desktop\embedded\traffic-sign-recognization\dataset\new_yolo_dataset'

IMG_DIR = os.path.join(BASE_PATH, 'images_yolo')
LBL_DIR = os.path.join(BASE_PATH, 'labels_yolo')
OUT_IMG_DIR = os.path.join(BASE_PATH, 'aug_images_yolo')
OUT_LBL_DIR = os.path.join(BASE_PATH, 'aug_labels_yolo')


os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Albumentations 증강 파이프라인
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.MotionBlur(p=0.2),
    A.GaussNoise(p=0.2),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# 이미지와 라벨 매칭 처리
for filename in os.listdir(IMG_DIR):
    if not filename.endswith('.png'):
        continue

    img_path = os.path.join(IMG_DIR, filename)
    label_path = os.path.join(LBL_DIR, filename + '.txt')  # 예: 00003.png.txt

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(img_path)
    height, width, _ = image.shape

    # YOLO 라벨 읽기
    with open(label_path, 'r') as f:
        lines = f.readlines()

    bboxes = []
    class_labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, x, y, w, h = parts
        bboxes.append([float(x), float(y), float(w), float(h)])
        class_labels.append(int(cls))

    for i in range(AUGMENT_PER_IMAGE):
        augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)

        out_img_name = filename.replace('.png', f'_aug{i}.png')
        out_lbl_name = filename.replace('.png', f'_aug{i}.png.txt')

        # 이미지 저장
        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_img_name), augmented['image'])

        # 라벨 저장
        with open(os.path.join(OUT_LBL_DIR, out_lbl_name), 'w') as f:
            for bbox, cls in zip(augmented['bboxes'], class_labels):
                x, y, w, h = bbox
                f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
