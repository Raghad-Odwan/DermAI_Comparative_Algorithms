# DermAI – Comparative Analysis of Skin Lesion Classification Models

This repository contains the **comparative evaluation phase** of the DermAI project.
It focuses on experimentally comparing multiple machine learning and deep learning models for **binary skin lesion classification (Benign vs. Malignant)** in order to support an informed and justified model selection.

The purpose of this repository is **analysis and comparison**, not deployment or final training.

---

## Objectives

* Experimentally compare multiple classification models on the same dataset
* Evaluate models using standard performance metrics
* Analyze trade-offs between different architectures
* Provide empirical evidence to support the selection of a final model for the DermAI system

---

## Models Evaluated

### Traditional / Baseline Models

* K-Nearest Neighbors (KNN)
* Artificial Neural Network (ANN)

### Deep Learning (From Scratch)

* Custom Convolutional Neural Network (CNN)

### Transfer Learning Architectures

* ResNet50
* VGG16
* DenseNet121
* InceptionV3
* Xception
* EfficientNetB0

---

## Evaluation Metrics

All models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-score

These metrics were chosen to provide a balanced view of performance, particularly under class imbalance.

---

## Results Summary

| Model          | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------- | --------- | ------ | -------- |
| DenseNet121    | 85.79%   | 77.40%    | 78.37% | 77.88%   |
| VGG16          | 85.29%   | 86.44%    | 63.95% | 73.51%   |
| InceptionV3    | 83.98%   | 76.24%    | 72.41% | 74.28%   |
| Xception       | 83.98%   | 79.34%    | 67.40% | 72.88%   |
| ResNet50       | 81.68%   | 76.56%    | 61.44% | 68.17%   |
| KNN            | 81.38%   | 81.52%    | 53.92% | 64.91%   |
| ANN            | 79.18%   | 77.07%    | 49.53% | 60.31%   |
| Custom CNN     | 73.57%   | 86.67%    | 20.38% | 32.99%   |
| EfficientNetB0 | 68.07%   | 0%        | 0%     | 0%       |

---

## Interpretation of Results

* Transfer learning models generally outperformed traditional machine learning and custom CNN approaches.
* DenseNet121 achieved the highest overall F1-score and recall among the evaluated models.
* VGG16 demonstrated high precision but comparatively lower recall.
* Traditional ML models and the custom CNN showed limited performance, likely due to the complexity of dermoscopic image features.
* EfficientNetB0 failed to produce meaningful predictions under the current training configuration.

This analysis highlights the variability in model behavior and the importance of considering multiple metrics rather than accuracy alone.

---

## Model Selection Rationale

Although DenseNet121 achieved the strongest numerical performance in this comparison, **ResNet50 was selected for subsequent stages of the DermAI project** based on a combination of experimental results and system-level considerations:

* Consistent and stable training behavior across experiments
* Favorable balance between performance and computational complexity
* Compatibility with the overall system architecture and deployment constraints
* Strong support for explainability techniques (e.g., Grad-CAM) used in later stages

The selection was therefore based on **both empirical evidence and engineering constraints**, rather than peak metric values alone.

---
### Preprocessing Steps

* Image resizing
* Normalization
* Data augmentation
* Class weighting during training

---

## Visual Analysis

The repository includes visual comparisons of model performance across all evaluation metrics.
These plots are generated and presented within the Jupyter notebook.

---

## Notebook

All experiments and analyses are implemented in:

```
DermAI_Comparative_Algorithms.ipynb
```

The notebook documents the full experimental workflow and result interpretation.

---

## Role in the DermAI Project

This repository represents the **model comparison and selection phase** of the DermAI AI pipeline and directly supports:

* Cross-validation experiments
* Final model training
* Explainability analysis

---

بس احكي 👍
