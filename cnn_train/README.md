
---

```markdown
# Speed Sign Classification - CNN (Grayscale, 128x128)

This project trains a CNN to classify grayscale speed sign images (30–90 km/h) using TensorFlow/Keras.

---

## 🗂 Directory Structure

```
/home/hyunoh/train/
├── Maximum_Speed_30/
├── Maximum_Speed_40/
├── Maximum_Speed_50/
├── Maximum_Speed_60/
├── Maximum_Speed_70/
├── Maximum_Speed_80/
└── Maximum_Speed_90/
```
- Each folder contains `.jpg` images labeled by class.

---

## 🚀 How to Run

```bash
python train_speed_sign_cnn.py
```

---

## ⚙️ Model Details

- Input: 128x128 grayscale images
- Output: 7-class softmax (speed signs: 30–90)
- Architecture:
  - 4× Conv2D + MaxPooling
  - Dense layers: 2048 → 512 → 128 → 7

---

## 📦 Output

- Trained model saved as:  
  `speed_sign_cnn_model.h5`

---

## 📊 Requirements

```bash
pip install tensorflow opencv-python scikit-learn
```

---

## 📌 Notes

- Data augmentation is used only for training set.
- Preprocessing includes grayscale conversion and resizing to (128,128,1).
```

---
