# 🧠 SkinGuard: Multi-YOLO Skin Cancer Detection System

SkinGuard is an advanced AI-powered system designed to detect skin cancer using a **multi-model object detection approach**.
It leverages **YOLOv8, YOLOv9, and YOLOv10** to perform detection and comparative analysis on skin lesion images.

The goal of this project is to explore how different YOLO architectures perform in the medical imaging domain and identify the most effective model for **early skin cancer detection**.

---

## 🚀 Key Highlights

* 🔬 Multi-YOLO architecture (YOLOv8, YOLOv9, YOLOv10)
* 📊 Comparative performance analysis across models
* 🧠 Deep learning-based lesion detection
* ⚡ Fast and efficient inference
* 📈 Evaluation using standard metrics (mAP, Precision, Recall)
* 🖼 Supports dermoscopic skin images

---

## 🏗️ Project Workflow

```
Input Image
     ↓
Preprocessing
     ↓
 ┌───────────────┬───────────────┬───────────────┐
 │   YOLOv8      │   YOLOv9      │   YOLOv10     │
 └───────────────┴───────────────┴───────────────┘
     ↓
Prediction Outputs (Bounding Boxes + Confidence)
     ↓
Model Comparison & Evaluation
     ↓
Final Analysis / Visualization
```

---

## 🤖 Models Used

| Model   | Description                                 |
| ------- | ------------------------------------------- |
| YOLOv8  | Lightweight and fast baseline detector      |
| YOLOv9  | Improved feature extraction and accuracy    |
| YOLOv10 | Optimized architecture for high performance |

---

## 📊 Evaluation Metrics

The models are evaluated using:

* **mAP (Mean Average Precision)**
* **Precision**
* **Recall**
* **Inference Time**
* **Confidence Score Analysis**

---

## 🧪 Dataset

* Skin lesion dataset (e.g., ISIC or custom dataset)

> Note: Ensure dataset is properly labeled in YOLO format.

---

## ⚙️ Tech Stack

* Python
* PyTorch
* YOLO (v8, v9, v10)
* OpenCV
* NumPy
* Pandas

---

## 📁 Project Structure

```
SkinGuard/
│── models/
│   ├── yolov8/
│   ├── yolov9/
│   ├── yolov10/
│
│── dataset/
│   ├── images/
│   ├── labels/
│
│── results/
│
│── detect.py
│── compare.py
│── requirements.txt
│── README.md
```

---

## 🧩 Installation

```bash
git clone https://github.com/your-username/skinguard.git
cd skinguard

pip install -r requirements.txt
```

---

## ▶️ Usage

### Run detection with a specific model:

```bash
python detect.py --model yolov8 --image sample.jpg
python detect.py --model yolov9 --image sample.jpg
python detect.py --model yolov10 --image sample.jpg
```

### Run comparison:

```bash
python compare.py --image sample.jpg
```

---

## 💡 Future Scope

* 🌐 Web or mobile deployment
* 🎥 Real-time detection using camera
* 🤖 Integration with AI agents for smart diagnosis
* 🧠 Ensemble learning for improved accuracy

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.
It should not be used as a substitute for professional medical advice or diagnosis.

---

## 👨‍💻 Author

**Dhruv Lokadiya**
---
**B.Tech CSE | AI/ML Enthusiast**

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!

---
