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

