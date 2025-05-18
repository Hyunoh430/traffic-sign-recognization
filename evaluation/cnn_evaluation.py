import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# 폴더 번호 ↔ 클래스 라벨 매핑
folder_to_label = {
    '2': '30',
    '3': '40',
    '4': '50',
    '5': '60',
    '6': '70',
    '7': '80'
}
class_labels = list(folder_to_label.values())

# TensorFlow Lite 모델 로딩
interpreter = tf.lite.Interpreter(model_path='saved_models/speed_sign_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 이미지 전처리 함수
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 128, 128, 1)
    return img

# 분류 함수
def classify_image(image_path):
    img = preprocess_image(image_path)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output_data, axis=1)[0]
    return class_labels[pred_idx]

# 평가용 리스트
y_true = []
y_pred = []

# 데이터셋 루트
dataset_root = 'dataset/cnn_dataset/'

# 반복 평가
for folder in folder_to_label:
    label = folder_to_label[folder]
    folder_path = os.path.join(dataset_root, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)

        # 예측 및 기록
        pred = classify_image(img_path)
        y_pred.append(pred)
        y_true.append(label)

# 정확도 출력
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision (macro):", precision_score(y_true, y_pred, average='macro'))
print("Recall (macro):", recall_score(y_true, y_pred, average='macro'))
print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
