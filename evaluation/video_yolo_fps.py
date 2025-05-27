from ultralytics import YOLO
import cv2
import time

# YOLOv8 모델 로드 (예: yolov8n)
model = YOLO("/Users/hyunoh/Documents/vscode/embedded/traffic-sign-recognization/saved_models/yolo_226/best.pt")  # 또는 커스텀 모델 사용: "runs/detect/train/weights/best.pt"

# 실시간 영상 소스: 0 = 웹캠, 또는 동영상 경로 입력
video_path = 0  # 예: "test_video.mp4" 또는 0
cap = cv2.VideoCapture(video_path)

# 비디오 정보 출력
assert cap.isOpened(), "Cannot open video source"
print(f"Video FPS: {cap.get(cv2.CAP_PROP_FPS)}, Resolution: {cap.get(3)}x{cap.get(4)}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 추론 시작 시간
    start = time.time()

    # YOLO 추론 (auto-resize됨, NMS 포함)
    results = model(frame, verbose=False)

    # 추론 종료 시간
    end = time.time()
    fps = 1 / (end - start)

    # 결과 시각화 (바운딩 박스 포함된 프레임 가져오기)
    annotated_frame = results[0].plot()

    # FPS 표시
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 화면 출력
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    # 종료 조건: ESC 키
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
