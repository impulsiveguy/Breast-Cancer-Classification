# Breast Cancer Classification Using Machine Learning

This project implements multiple **Machine Learning algorithms** for **Breast Cancer Diagnosis Prediction** using the **Breast Cancer Wisconsin Dataset**.  
It includes **data preprocessing, visualization, model comparison, hyperparameter tuning, and evaluation** — optimized for accuracy, recall, and interpretability.

---

## Overview

The goal of this project is to predict whether a tumor is **malignant (M)** or **benign (B)** based on cell nucleus features computed from digitized breast mass images.

A wide range of models are compared and fine-tuned:
- **Logistic Regression**
- **XGBoost**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Artificial Neural Network (ANN / MLPClassifier)**

Each model is optimized using **RandomizedSearchCV** with **Recall as the refit metric**, which is crucial for medical prediction tasks where false negatives must be minimized.

---

## Features

- Comprehensive **Exploratory Data Analysis (EDA)**  
- Correlation and feature importance heatmaps  
- Distribution and violin plots for class-wise comparison  
- Model training and accuracy comparison across multiple algorithms  
- **Hyperparameter tuning** with Repeated Stratified K-Fold cross-validation  
- **Feature importance** visualization for tree-based and linear models  
- **ROC Curves** and **AUC evaluation** for best-performing models  
- Model saving with **Joblib** for future inference  

---

## Model Performance Summary

| Model                | Accuracy | CV Recall | Key Hyperparameters |
|----------------------|-----------|------------|----------------------|
| Logistic Regression  | 0.9737    | 0.9643     | C=0.1, L1, balanced  |
| XGBoost              | 0.9561    | 0.9464     | n=300, depth=3, lr=0.05 |
| Random Forest        | 0.9561    | 0.9446     | n=300, depth=10 |
| KNN                  | 0.9649    | 0.9308     | k=7, p=1 |
| ANN (MLPClassifier)  | 0.9737    | 0.9525     | hidden=(100,), adaptive |

**Best Model (Recall-Focused): Logistic Regression**

---

## Key Visualizations

- **Box plots** and **violin plots** for feature distribution  
- **Pair plots** showing feature relationships by diagnosis  
- **Correlation heatmap** identifying key predictive variables  
- **ROC curves** comparing AUC across models  
- **Feature importance** for linear and ensemble models  

---

## Dependencies

Install required dependencies using:

```bash
pip install -r requirements.txt
