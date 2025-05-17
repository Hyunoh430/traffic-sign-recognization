# Speed Sign Detection System (YOLOv5 + CNN + TFLite)

End-to-end speed sign recognition pipeline:
1. Detect speed signs using a YOLOv5 grayscale model
2. Crop and preprocess the detection
3. Classify cropped sign using a lightweight CNN (TensorFlow Lite)

---

## 📁 Project Structure

```
project_root/
├── run.py                       # Runs YOLOv5 detection and crops highest-confidence result
├── detect_image_utils.py        # (optional) Contains reusable detect_image() logic
├── 128_128.pt                   # YOLOv5 grayscale model (128x128)
├── image1.jpg                   # Sample input image
├── result_image.jpg             # YOLOv5 result with bounding boxes
├── cropped_result.jpg           # Cropped ROI for CNN input
├── speed_sign_model.h5          # Trained CNN model (Keras)
├── speed_sign_model.tflite      # Quantized TFLite model (optional)
├── train_speed_sign_cnn.py      # CNN training script using grayscale images
├── changing_yaml_to_gray.ipynb  # Notebook for YOLO grayscale config + training
├── /home/hyunoh/train/          # Folder of labeled training images by class
│   ├── Maximum_Speed_30/
│   ├── Maximum_Speed_40/
│   └── ...
└── README.md                    # ← this file
```

---

## 🔍 Step Overview

### 1. Train YOLOv5 (Grayscale, 128x128)
- Train YOLOv5 with custom data using `changing_yaml_to_gray.ipynb`
- Output model: `128_128.pt`

### 2. Run Detection
```bash
python run.py
```
- Loads `image1.jpg`
- Runs YOLOv5
- Saves:
  - `result_image.jpg` (with bounding box)
  - `cropped_result.jpg` (for CNN input)

### 3. Train CNN Classifier
```bash
python train_speed_sign_cnn.py
```
- Trains CNN using grayscale cropped images (7 classes)
- Saves model as `speed_sign_model.h5`

### 4. (Optional) Convert to TFLite
```python
import tensorflow as tf
model = tf.keras.models.load_model("speed_sign_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("speed_sign_model.tflite", "wb") as f:
    f.write(tflite_model)
```

---

## 🛠 Requirements

```bash
pip install torch torchvision opencv-python tensorflow scikit-learn
```

---

## 🎯 Goals

- Fast and lightweight speed sign recognition system for embedded devices (e.g., Raspberry Pi)
- YOLOv5 + CNN architecture
- Optimized input size: 128×128 grayscale

---

## 📌 Notes

- `run.py` avoids `detect.py` and gives full control over detection and post-processing
- CNN model is trained separately using speed sign class folders

--수정해야할듯 아직 미완성--
