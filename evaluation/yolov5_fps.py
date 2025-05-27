import torch
import cv2
import time

# 1. YOLOv5 모델 로드 (Torch Hub)
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='/Users/hyunoh/Documents/vscode/embedded/traffic-sign-recognization/saved_models/128_128.pt',
                       force_reload=True)

# 2. GPU or CPU 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 3. 비디오 캡처 시작 (0은 웹캠 / 또는 'your_video.mp4')
video_path = 0  # 또는 'video.mp4'
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Video capture failed"

# 4. FPS 측정 변수
frame_count = 0
total_time = 0

# 5. 루프
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    # 6. YOLO 추론 (자동 resize + NMS 포함)
    results = model(frame)

    end = time.time()
    inference_time = end - start
    fps = 1 / inference_time
    total_time += inference_time
    frame_count += 1

    # 7. 결과 시각화 (읽기 가능한 복사본 사용)
    annotated_frame = results.render()[0].copy()

    # 8. FPS 텍스트 추가
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 9. 화면 출력
    cv2.imshow("YOLOv5 Detection", annotated_frame)

    # 10. ESC 키로 종료
    if cv2.waitKey(1) == 27:
        break

# 11. 종료 처리
cap.release()
cv2.destroyAllWindows()

# 12. 평균 FPS 출력
print(f"Average FPS: {frame_count / total_time:.2f}")
