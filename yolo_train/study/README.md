# YOLOv5 모델 구조 및 전이학습 개념 정리

## 1. YOLOv5 전체 구조 개요

YOLOv5는 아래와 같은 세 부분으로 구성됨:

```
[입력 이미지]
      ↓
  ┌──────────────┐
  │  Backbone    │  ← 특징 추출기 (ex. CSPDarknet)
  └──────────────┘
         ↓
  ┌──────────────┐
  │     Neck     │  ← 다중 스케일 피처 융합 (ex. PANet)
  └──────────────┘
         ↓
  ┌──────────────┐
  │     Head     │  ← 바운딩 박스, 클래스, 확률 출력
  └──────────────┘
         ↓
[탐지 결과: box + class + confidence]
```

---

## 2. 각 구성요소 설명

### 2.1 Backbone
- 입력 이미지로부터 시각적 특징(feature)을 추출
- YOLOv5에서는 CSPDarknet 구조 사용
- 계층적으로 Conv → C3 → SPPF 등으로 구성됨
- 일반적인 텍스처, 모서리, 윤곽선 등을 학습함

### 2.2 Neck
- 다양한 크기의 객체를 탐지하기 위한 피처맵 융합
- PANet(FPN 기반) 사용
- 고해상도 feature와 저해상도 feature를 결합

### 2.3 Head
- 바운딩 박스 좌표, 클래스 확률, 객체 확률 예측
- 클래스 수에 따라 자동으로 구조 조정됨
- 예: 3개의 커스텀 클래스 → Head는 3개의 출력을 가지게 됨

---

## 3. Pretrained Weight란?

- COCO 같은 대규모 공개 데이터셋으로 **사전 학습된 가중치 파일**
- 예: `yolov5n.pt`, `yolov5s.pt`
- Backbone과 Neck의 weight가 저장됨
- 전이학습 시 성능 향상과 빠른 수렴에 유리

---

## 4. 전이학습 (Transfer Learning)

### 전이학습 작동 방식
- 기존 pretrained weight 로딩
- Head는 내 클래스 수에 맞춰 새로 초기화
- Backbone은 그대로 유지 → 일반적 특징 재활용
- 결과적으로 학습 속도 향상, 성능 안정

### 학습 명령어 예시
```bash
python train.py --img 128 --batch 16 --epochs 100 --data data.yaml --weights yolov5n.pt --name my_project
```

### data.yaml 예시
```yaml
train: ./dataset/train/images
val: ./dataset/val/images
nc: 3
names: ['speed30', 'speed50', 'speed60']
```

---

## 5. YOLO 구조 수정

- YOLO 구조는 `yolov5/models/*.yaml` 파일로 정의됨
- 수정 가능 항목:
  - `depth_multiple`: 레이어 깊이 조절
  - `width_multiple`: 채널 수 조절
- 구조를 바꾸면 pretrained weight는 **호환되지 않음**
  - → 처음부터 학습 (from scratch)

---

## 6. 발표용 요약 멘트 예시

> YOLOv5는 backbone, neck, head의 3단 구조로 구성되며, pretrained weight를 이용한 전이학습을 통해 적은 양의 데이터로도 효과적인 탐지가 가능합니다. Head는 클래스 수에 맞게 조정되어, 저희만의 속도 표지판 데이터셋에도 잘 적응할 수 있었습니다.
