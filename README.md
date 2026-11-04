# TBXNet: Automated Disease Diagnosis from Medical Reports using Deep Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.12+](https://img.shields.io/badge/TensorFlow-2.12%2B-orange.svg)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An official implementation of **TBXNet**, an ultra-lightweight Deep Convolutional Neural Network (~2.3M parameters) engineered for high-precision screening and differential diagnosis of **Tuberculosis, Pneumonia, and COVID-19** from chest radiographs (CXRs) and medical reports.

---

## 📌 Project Overview

- **Project Title:** Design and Development of Disease Diagnosis from Medical Reports using Deep Learning
- **Authors:** Abhishek Tomar (BTAI2401003) & Adarsh (BTAI2401005)
- **Supervision:** Dr. Pooja Tripathi (Assistant Professor, Centre for AI, MITS Gwalior)
- **Institution:** Madhav Institute of Technology & Science (MITS), Gwalior (M.P.), India
- **Department:** Centre for Artificial Intelligence (AI)

---

## ✨ Key Features & Innovations

- **Specialized Feature Fusion Layer ($\mathcal{F}_{fused}$):** Combines low-level high-resolution spatial texture maps directly with abstract high-level semantic feature representations to preserve fine parenchymal details critical for early tuberculosis detection.
- **Borderline SMOTE (BL-SMOTE):** Integrates boundary-focused oversampling to eliminate class imbalancing across public medical image archives, significantly reducing false-negative rates.
- **Ultra-Lightweight Footprint:** Maintains total trainable parameters under **2.5 million (~2.3M)**, eliminating VRAM bottlenecks and allowing deployment on low-resource edge workstations.
- **Real-Time Latency:** Achieves an average inference execution latency of **14.2 milliseconds per radiograph** on standard hardware.

---

## 📊 Benchmark Performance

| Evaluation Cohort | Classification Type | Accuracy | Precision | F1-Score | Key Highlights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Dataset A** | Binary (TB vs. Normal) | **98.98%** | 99.02% | 98.95% | High-precision baseline TB screening |
| **Dataset B** | Expanded Binary Cohort | **99.17%** | 99.20% | 99.15% | Robustness against anatomical noise |
| **Dataset C** | 4-Class Generalization | **95.10%** | 95.25% | 95.08% | **96.2% TB Sensitivity**, **0.982 COVID-19 AUC** |

---

## 📐 System Architecture

$$\mathcal{F}_{fused} = \text{Concat}(\Gamma_1(\mathcal{F}_{block1}), \Gamma_2(\mathcal{F}_{block2}), \Gamma_3(\mathcal{F}_{block3}))$$

```text
[Input Radiograph Image: 128x128x3] 
       │
       ▼
[Conv Block 1: 3x3 Conv, 32 Filters, ReLU] ──► Max-Pooling 2D (2x2)
       │
       ▼
[Conv Block 2: 3x3 Conv, 64 Filters, ReLU] ──► Max-Pooling 2D (2x2)
       │
       ▼
[Conv Block 3: 3x3 Conv, 128 Filters, ReLU] ──► Max-Pooling 2D (2x2)
       │
       ▼
[SPECIALIZED FEATURE FUSION LAYER] ◄── Concatenates Multi-Scale Spatial & Semantic Maps
       │
       ▼
[Dense Block: 512 Units + Dropout (p=0.3)]
       │
       ▼
[SoftMax Output Classification Layer: Target Classes (TB / Normal / Pneumonia / COVID-19)]
```

---

## 🛠 Tech Stack

- **Language:** Python v3.10+
- **Deep Learning Frameworks:** TensorFlow v2.12.0+, Keras, PyTorch Vision
- **Image & Audio Processing:** OpenCV, Pillow (PIL), Librosa, SciPy
- **Machine Learning & Data Balancing:** Scikit-Learn, Imbalanced-Learn (`imblearn`)
- **Visualization:** Matplotlib, Seaborn

---

## 📁 Repository Structure

```text
.
├── data_pipeline/         # Preprocessing and BL-SMOTE resampling modules
├── models/                # TBXNet architecture & custom fusion layer definitions
├── notebooks/             # Exploratory Data Analysis (EDA) & performance curves
├── sample_radiographs/    # Sample chest radiographs for inference testing
├── data_loader.py         # Data loading, image normalization, & CWT conversion
├── balance_data.py        # Boundary-focused oversampling using BL-SMOTE
├── tbxnet_model.py        # Neural network graph & feature concatenation layers
├── train.py               # Training loop with SGDM optimizer & Cross-Entropy loss
├── prediction.py          # Command-line inference & prediction execution
├── requirements.txt       # Software dependencies
└── README.md              # Project documentation
```

---

## 🚀 Installation & Quick Start

### 1. Clone the Repository & Install Dependencies
```bash
git clone https://github.com/your-username/tbxnet-disease-diagnosis.git
cd tbxnet-disease-diagnosis
pip install -r requirements.txt
```

### 2. Preprocess & Balance Dataset
```bash
python balance_data.py --apply_blsmote True
```

### 3. Train the Model
```bash
python train.py --epochs 40 --batch_size 32 --lr 0.05
```

### 4. Run Model Inference on Sample Radiograph
```bash
python prediction.py --image_path sample_radiographs/Tuberculosis-8.png
```

**Output:**
```text
C:\Users\Public\tbxnet_chest_x_ray>python prediction.py
Prediction Result: TUBERCULOSIS
```

---

## 📜 References & Citations

If you use this repository or TBXNet framework in your research, please cite:

1. **Lakhani, P., & Sundaram, B. (2017).** Deep learning at chest radiography: Automated classification of pulmonary tuberculosis by using convolutional neural networks. *Radiology*, 284(2), 574–582.
2. **Jaeger, S., et al. (2014).** Two public chest X-ray datasets for computer-aided screening of pulmonary diseases. *Quantitative Imaging in Medicine and Surgery*, 4(6), 475–477.
3. **Frontiers in Medicine (2022).** A systematic review of deep learning techniques for tuberculosis detection from chest radiographs. *Frontiers in Medicine*, 9, 830515.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
