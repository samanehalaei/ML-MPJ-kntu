# ML-MPJ2 Project
Description of ML-MPJ2 project.
Classification Models and Techniques

This project focuses on designing and evaluating various classification models and techniques using Python. The main tasks involve creating neural networks, building Multi-Layer Perceptrons (MLPs), designing decision trees, and applying Bayes classification for medical diagnosis.

## Project Description

### Technologies Used
- Python
- PyCharm
- scikit-learn
- Kaggle datasets

### Key Concepts
- Neural networks
- Multi-Layer Perceptrons (MLP)
- Decision trees
- Bayesian classification
- Model evaluation

### Datasets
- CWRU Bearing Fault Diagnosis Dataset
- Forest Cover Classification Dataset
- Heart Disease Dataset

## First Question

Design a network for a two-class classification problem with the last two layers being ReLU and sigmoid. Use a simple neuron model, such as a McCulloch-Pitts neuron, for this design.

### Overview:

1. McCulloch-Pitts Neuron:
   - Description:
     - The McCulloch-Pitts neuron is a simple binary threshold unit that can be used to design the network.
   - Function:
     - Implement a basic neuron model that computes a weighted sum of inputs and applies a threshold function to determine the output.

2. Network Design:
   - Layers:
     - Implement the final layers using ReLU (Rectified Linear Unit) for activation and a sigmoid function for output.
   - Activation Functions:
     - ReLU introduces non-linearity and helps in learning complex patterns.
     - Sigmoid function maps outputs to probabilities in the range [0, 1].

## Second Question

Build a Multi-Layer Perceptron (MLP) model with 2 or more hidden layers using the CWRU Bearing Fault Diagnosis dataset.

### Overview:

1. Data Preparation:
   - Load Data:
     - Import the fault diagnosis dataset and preprocess it.
   - Normalization:
     - Scale the data to ensure all features contribute equally to the model.

2. Model Design:
   - MLP Configuration:
     - Design an MLP with two hidden layers, using ReLU activation and an appropriate solver.
   - Training:
     - Train the MLP model using the preprocessed dataset.
   - Evaluation:
     - Assess the model's performance using accuracy, classification reports, and confusion matrices.

## Third Question

Design a decision tree classifier for a dataset related to forest cover or medicine. Analyze the decision tree both programmatically and manually. Explain how methods like Random Forest and AdaBoost can improve results.

### Overview:

1. Data Preparation:
   - Load Dataset:
     - Use datasets such as the Forest Cover Type or medical datasets for classification tasks.
   - Preprocessing:
     - Handle missing values, normalize or standardize features as necessary.

2. Decision Tree Model:
   - Design:
     - Implement a decision tree classifier and visualize its structure.
   - Analysis:
     - Analyze the decision tree's performance using metrics like accuracy and cross-validation scores.

3. Ensemble Methods:
   - Random Forest:
     - Combines multiple decision trees to improve classification accuracy and control overfitting.
   - AdaBoost:
     - Enhances model performance by combining weak classifiers into a strong classifier, focusing on difficult-to-classify instances.

## Fourth Question

Diagnose heart disease using the Bayes classification algorithm. After dividing the data into training and testing sets and performing necessary preprocessing, evaluate the model. Explain the difference between Macro and Micro averaging in classification metrics.

### Overview:

1. Data Preparation:
   - Preprocessing:
     - Standardize or normalize the dataset.
   - Train-Test Split:
     - Divide the data into training and testing sets.

2. Bayes Classification:
   - Model:
     - Use the Gaussian Naive Bayes classifier for prediction.
   - Evaluation:
     - Evaluate the classifier using metrics like accuracy, classification report, and confusion matrix.
