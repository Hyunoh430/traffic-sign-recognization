
---

## ✅ README.md (핵심 요약 중심)

```markdown
# YOLOv5 + TFLite Speed Sign Detection

This project performs speed limit sign detection using a YOLOv5 model and crops the highest-confidence detection for further classification via a lightweight CNN (e.g., TensorFlow Lite).

---

## 🔧 Features

- Lightweight YOLOv5 model (`128x128.pt`) for real-time detection
- Automatically crops the most confident detected region
- Resizes and converts cropped region to grayscale for TFLite input
- Saves both full detection result and cropped image

---

## 📁 File Structure

```
.
├── run.py                    # Main script to run YOLOv5 detection and crop result
├── 128_128.pt                # YOLOv5 grayscale model (128x128 input)
├── speed_sign_model.tflite   # TFLite classifier model (optional, not invoked in run.py)
├── image1.jpg                # Input test image
├── result_image.jpg          # Full detection image with bounding box
├── cropped_result.jpg        # Cropped + grayscale image (TFLite-ready)
└── changing_yaml_to_gray-2.ipynb  # Training + preprocessing notebook
```

---

## 🚀 Quick Start

```bash
python run.py
```

> Default config uses:
> - `weights='128_128.pt'`
> - `img_path='image1.jpg'`
> - Output saved to `result_image.jpg`, `cropped_result.jpg`

---

## 🛠 Requirements

```bash
pip install torch opencv-python
```

---

## 🧠 Notes

- YOLOv5 inference is handled manually using `DetectMultiBackend`, not `detect.py`, for full control.
- Cropped image (128×128 grayscale) is ready for inference with a TFLite classifier like `speed_sign_model.tflite`.

---

## 📌 Author

Developed by [HyunOh] for speed sign recognition on embedded systems (e.g., Raspberry Pi).
```

---

