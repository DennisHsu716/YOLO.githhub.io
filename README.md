# Tableware Object Detection with YOLO

## 📌 Project Overview
This project trains a YOLO-based object detection model to detect **tableware** objects (e.g., plates, cups, bowls) using a custom-labeled dataset.  
The dataset follows the YOLO format, including training, validation, and testing splits.  
The training is performed on **Google Colab** with the dataset stored in **Google Drive**.

---
### 📂 Repository Structure
YOLO.githhub.io/     
│   
├── requirements.txt             # Required Python packages    
├── data.yaml                    # YOLO dataset config (classes + paths)    
│  
├── datasets/                    # Small sample dataset for quick test    
│     │── Train/   
│     │ ├── Images/    
│     │ └── Labels/   
│     │── Valid/   
│     │ ├── Images/    
│     │ └── Labels/   
│     │── Test/  
│     │ ├── Images/  
│     │ └── Labels/  
├── Weights/                     # Model weights   
│   └── best.pt   
│   
├── Results/                     # Training results summary    
│   ├── results.png   
│   ├── confusion_matrix.png   
│   └── PR_curve.png   
│  
├── Samples/                     # Demo images  
│   ├── demo.jpg  
│   └── demo_result.jpg  
│  
└── Scripts/           
    └── Untitled0.ipynb  

## 📂 Dataset Structure
Tableware/  
│── Train/  
│ ├── Images/  
│ └── Labels/  
│── Valid/  
│ ├── Images/  
│ └── Labels/  
│── Test/  
│ ├── Images/  
│ └── Labels/  
└── Data.yaml  


- **images/** → contains `.jpg` images    
- **labels/** → contains `.txt` label files in YOLO format   
- **data.yaml** → dataset configuration file for YOLO   

---

## ⚙ Environment Setup 
```python
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```
### 🚀 Training the Model
```
from google.colab import drive
drive.mount('/content/drive')

!pip install -r requirements.txt

!python train.py \
  --data /content/drive/MyDrive/tableware/data.yaml \
  --weights yolov5s.pt \
  --img 640 \
  --batch 16 \
  --epochs 50 \
  --project runs_custom \
  --name tableware_model
```
### 📊 Evaluation
```
python val.py --weights runs_custom/tableware_model/weights/best.pt \  
              --data /content/drive/MyDrive/tableware/data.yaml
```

### 🔍 Inference
```
python detect.py \
  --weights runs_custom/tableware_model/weights/best.pt \
  --source /content/drive/MyDrive/tableware/test/images \
  --img 640 \
  --conf 0.25
```
Detected images will be saved in:  
```runs/detect/exp/```

### 📦 Download Full Dataset
Due to file size, the full training dataset is stored externally:  
📥 Google Drive Link ((https://drive.google.com/drive/folders/1czJoabUyuPLusa56is1rayl_yV-xZARB?usp=sharing)  

### 📌 Future Improvements
* Increase dataset size for better generalization
* Apply data augmentation techniques
* Try YOLOv8 for higher accuracy and faster inference
