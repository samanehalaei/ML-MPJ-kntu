# ML-MPJ3 Project
Description of ML-MPJ3 project.
## Project Overview

This project explores the implementation and evaluation of advanced classification models using Python, focusing on tasks such as testing the SVM algorithm on the Gholzanebq dataset and building an autoencoder for credit card fraud detection.

## Table of Contents
1. [Introduction](#introduction)
2. [First Question: SVM on Gholzanebq Dataset](#first-question-svm-on-gholzanebq-dataset)
3. [Third Question: Credit Card Fraud Detection using Autoencoder](#third-question-credit-card-fraud-detection-using-autoencoder)
4. [Dependencies](#dependencies)
5. [Setup Instructions](#setup-instructions)
6. [Usage](#usage)
7. [References](#references)

## Introduction

The ML-MPJ3 project is structured to test different machine learning techniques on various datasets. Specifically, it involves:
1. Testing Support Vector Machine (SVM) on different samples from the Gholzanebq dataset.
2. Implementing an autoencoder neural network for credit card fraud detection and summarizing the key sections of the research paper on this topic.

## First Question: SVM on Gholzanebq Dataset

### Objective
Evaluate the performance of the SVM algorithm on various samples from the Gholzanebq dataset.

### Steps
1. Data Preparation: 
    - Load and preprocess the Gholzanebq dataset.
    - Normalize the data to ensure uniform feature contribution.

2. Dimensionality Reduction:
    - Apply Principal Component Analysis (PCA) to reduce the dataset to 2 principal components.
    - Use t-Distributed Stochastic Neighbor Embedding (t-SNE) for further visualization in a lower-dimensional space.

3. SVM Classification:
    - Train an SVM classifier with a linear kernel on the reduced data.
    - Experiment with polynomial kernels of varying degrees.
    - Evaluate the performance of the model using metrics such as accuracy, confusion matrix, etc.

### Deliverables
- Report including the code and analysis.
- Visualizations and performance metrics for each SVM model.

## Third Question: Credit Card Fraud Detection using Autoencoder

### Objective
Summarize the key sections of the research paper "Credit Card Fraud Detection Using Autoencoder Neural Network" and implement the methodology.

### Steps
1. Data Preparation:
    - Load and preprocess the credit card fraud dataset.
    - Add Gaussian noise to the training data for robustness.

2. Autoencoder Implementation:
    - Build a denoising autoencoder using PyTorch.
    - Train the autoencoder on the noisy data to learn clean representations.

3. Classifier Implementation:
    - Define and train a neural network classifier on the denoised data.
    - Evaluate the classifier using metrics such as accuracy, recall, precision, F1-score, and confusion matrix.

4. Summarize the Paper:
    - Briefly report on the problem design, method introduction, and summary sections of the research paper.

### Deliverables
- Report including the code, analysis, and implementation.
- Summary of the research paper.
- Visualizations and performance metrics for the autoencoder and classifier.

## Dependencies

The project requires the following libraries and tools:
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- PyTorch
- imbalanced-learn
