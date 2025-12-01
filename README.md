# DermAI – Comparative Analysis of Skin Cancer Classification Models

DermAI is a machine-learning project focused on analyzing, comparing, and evaluating multiple traditional and deep-learning algorithms for binary skin-cancer classification (Benign vs. Malignant).  
The goal of this repository is to provide a comprehensive comparison between classical ML models, custom CNNs, and state-of-the-art transfer-learning architectures, and to justify the selection of the final model used in the main DermAI system.

---

## Project Objectives
- Compare a wide range of ML and DL models for skin cancer classification.
- Evaluate performance using multiple metrics: Accuracy, Precision, Recall, and F1-Score.
- Analyze strengths and weaknesses of each model in the context of medical imaging.
- Provide a clear justification for selecting ResNet50 as the final classifier for DermAI.

---

## Models Compared

### Traditional ML Models
- K-Nearest Neighbors (KNN)  
- Artificial Neural Network (ANN)

### Deep Learning (Custom)
- Custom CNN (from scratch)

### Transfer Learning Architectures
- ResNet50 (final chosen model for DermAI)  
- VGG16  
- DenseNet121  
- InceptionV3  
- Xception  
- EfficientNetB0

---

## Results Summary

Final performance metrics for all evaluated models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| DenseNet121 | 85.79% | 77.40% | 78.37% | 77.88% |
| VGG16 | 85.29% | 86.44% | 63.95% | 73.51% |
| InceptionV3 | 83.98% | 76.24% | 72.41% | 74.28% |
| Xception | 83.98% | 79.34% | 67.40% | 72.88% |
| ResNet50 | 81.68% | 76.56% | 61.44% | 68.17% |
| KNN | 81.38% | 81.52% | 53.92% | 64.91% |
| ANN | 79.18% | 77.07% | 49.53% | 60.31% |
| Custom CNN | 73.57% | 86.67% | 20.38% | 32.99% |
| EfficientNetB0 | 68.07% | 0% | 0% | 0% |

---

## Interpretation of Results

Several observations can be made based on the evaluation:

- DenseNet121 achieved the highest F1-score and recall, making it the strongest performer in raw detection capability.  
- VGG16 achieved the highest precision but lower recall, indicating that it is more selective but may miss some malignant cases.  
- Custom CNN and traditional ML models showed weaker performance due to the complexity of dermoscopic images and limited representational power.  
- EfficientNetB0 struggled significantly and failed to detect malignant cases, likely due to class imbalance or training instability.

---

## Justification for Selecting ResNet50

Although DenseNet121 achieved the highest numerical performance, ResNet50 was selected as the final model for the DermAI system for several practical and technical reasons:

1. ResNet50 is widely adopted in medical imaging research, which makes it easier to reference, compare, and justify academically.  
2. It offers a stable balance between accuracy, generalization, and training behavior.  
3. ResNet50 is lighter and faster than DenseNet and Inception models, making it more suitable for deployment in web or mobile environments.  
4. The DermAI system architecture, backend design, and SRS were developed around ResNet50, making it the most practical engineering choice.

---

## Dataset

- HAM10000 dermoscopic dataset.  
- Highly imbalanced (benign cases significantly outnumber malignant cases).  
- Preprocessing included:
  - Resizing  
  - Normalization  
  - Augmentation  
  - Class weighting  

---

## Visual Comparisons

The repository contains visual plots comparing all models across the four main metrics (Accuracy, Precision, Recall, F1-score).  
These visualizations are located in the notebook.

---

## Notebook

Complete implementation is provided in:

DermAI_Comparative_Algorithms.ipynb

---

## Conclusion

This repository documents the full evaluation process that led to selecting ResNet50 as the final classifier for the DermAI system.  
The comparison highlights trade-offs between multiple architectures and demonstrates a clear, scientific methodology for selecting a model suitable for medical image classification.

## Developers
 Raghad Odwan _ AI Engineer 


