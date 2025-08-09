# Tableware Object Detection with YOLO

## 📌 Project Overview
This project trains a YOLO-based object detection model to detect **tableware** objects (e.g., plates, cups, bowls) using a custom-labeled dataset.  
The dataset follows the YOLO format, including training, validation, and testing splits.  
The training is performed on **Google Colab** with the dataset stored in **Google Drive**.

---

## 📂 Dataset Structure
tableware/  
│── train/  
│ ├── images/  
│ └── labels/  
│── valid/  
│ ├── images/  
│ └── labels/  
│── test/  
│ ├── images/  
│ └── labels/  
└── data.yaml  


- **images/** → contains `.jpg` or `.png` images  
- **labels/** → contains `.txt` label files in YOLO format  
- **data.yaml** → dataset configuration file for YOLO

---

## ⚙ Environment Setup (Google Colab)
### 1️⃣ Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2️⃣ Install YOLOv8 (Ultralytics)
```!pip install ultralytics -q```

### 📝 Data Configuration (data.yaml)
```
train: /content/drive/MyDrive/tableware/train/images
val: /content/drive/MyDrive/tableware/valid/images
test: /content/drive/MyDrive/tableware/test/images
```
nc: 8  
names: ['chopsticks', 'dessert fork', 'dessert spoon', 'dinner fork', 'meal knife', 'salad fork', 'soup spoon', 'spoon']  

### 🚀 Training the Model
```
from ultralytics import YOLO
%env WANDB_DISABLED=true  # disable Weights & Biases logging

model = YOLO('yolov8n.pt')  # load pre-trained YOLOv8 nano model
model.train(
    data='/content/drive/MyDrive/tableware/data.yaml',
    imgsz=640,
    epochs=50,
    batch=16,
    project='runs_custom',
    name='tableware_model'
)
```
### 📊 Evaluation
```model.val()```

### 🔍 Inference
```results = model.predict('/content/drive/MyDrive/tableware/test/images/example.jpg', save=True)```

Detected images will be saved in:  
```runs_custom/tableware_model/predict/```

### 📌 Future Improvements
* Increase dataset size for better generalization
* Apply data augmentation techniques
