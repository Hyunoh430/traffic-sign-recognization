import json
import os

# 대상 클래스
target_classes = {
    "speed limit 20": 0,
    "speed limit 30": 1,
    "speed limit 50": 2,
    "speed limit 60": 3,
    "speed limit 70": 4,
    "speed limit 80": 5,
    "speed limit 100": 6,
    "speed limit 120": 7
}

# 경로 설정
json_dir = "/Users/hyunoh/Documents/vscode/embedded/traffic_sign_recognition/dataset/new_yolo_dataset/train/ann"
output_dir = "/Users/hyunoh/Documents/vscode/embedded/traffic_sign_recognition/dataset/new_yolo_dataset/labels_yolo/"
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(json_dir):
    if not file.endswith(".json"):
        continue

    with open(os.path.join(json_dir, file)) as f:
        data = json.load(f)

    img_w = data["size"]["width"]
    img_h = data["size"]["height"]

    label_lines = []

    for obj in data["objects"]:
        class_name = obj["classTitle"]
        if class_name not in target_classes:
            continue  # 🚨 speed limit 아닌 건 skip

        x1, y1 = obj["points"]["exterior"][0]
        x2, y2 = obj["points"]["exterior"][1]

        # 중심점 및 박스 크기 (YOLO 상대좌표)
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h

        cls_id = target_classes[class_name]
        label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    # 저장
    if label_lines:
        out_file = os.path.join(output_dir, file.replace(".json", ".txt"))
        with open(out_file, "w") as out:
            out.write("\n".join(label_lines))
