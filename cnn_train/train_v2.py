import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 이미지 불러오기 및 전처리 함수
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            images.append(img)
            labels.append(label)
    return images, labels

# 각 폴더에 대한 라벨링 (예: 30 = 0, 40 = 1, ...)
folders = {
    'Maximum_Speed_30': 0,
    'Maximum_Speed_40': 1,
    'Maximum_Speed_50': 2,
    'Maximum_Speed_60': 3,
    'Maximum_Speed_70': 4,
    'Maximum_Speed_80': 5,
    'Maximum_Speed_90': 6
}

all_images = []
all_labels = []

# 각 폴더에서 이미지 불러오기
for folder_name, label in folders.items():
    folder_path = os.path.join('/home/hyunoh/train', folder_name)  # 경로 수정 필요
    images, labels = load_images_from_folder(folder_path, label)
    all_images.extend(images)
    all_labels.extend(labels)

# 이미지를 numpy 배열로 변환
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# CNN 모델을 위해 4차원 배열로 reshape
all_images = all_images.reshape(all_images.shape[0], 128, 128, 1)

# 데이터셋 분리
X_train, X_val, y_train, y_val = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

# 데이터 증강
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
val_generator = val_datagen.flow(X_val, y_val, batch_size=32)

# CNN 모델 구성
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(2048, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(7, activation='softmax')  # 7개의 클래스 (30, 40, 50, 60, 70, 80, 90)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 훈련
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator
)

# 모델 평가
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_acc}")

# 모델 저장 (옵션)
model.save('speed_sign_cnn_model.h5')
