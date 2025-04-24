import torch
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T

def inference_and_save(model, input_image_path, output_image_path, threshold=0.3):
    # 이미지 로드 및 전처리
    img = Image.open(input_image_path).convert("RGB")
    img_tensor = T.ToTensor()(img).unsqueeze(0)  # 이미지 텐서로 변환 및 배치 차원 추가

    # 모델 추론
    with torch.no_grad():
        predictions = model(img_tensor.to(torch.device('cpu')))  # CPU로 모델 실행
    
    # 이미지 변환 및 그리기
    img_np = np.array(img)  # PIL 이미지를 numpy 배열로 변환
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)  # OpenCV 형식으로 변환

    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i].item()
        if score > threshold:  # 신뢰도 점수가 임계값 이상인 경우만 표시
            x1, y1, x2, y2 = map(int, box)
            label = predictions[0]['labels'][i].item()
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, f'Label: {label} Score: {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 이미지 저장
    cv2.imwrite(output_image_path, img_np)
    print(f'Result saved at {output_image_path}')

# 사용 예시
input_image_path = 'test_video.mp4'  # 입력 이미지 경로
output_image_path = 'r.mp4'  # 출력 이미지 저장 경로

# 모델 로드
model_path = '15road_best_model.pt'
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# 추론 및 저장
inference_and_save(model, input_image_path, output_image_path, threshold=0.3)