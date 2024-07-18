# ML-MPJ1 Project
Description of ML-MPJ1 project.
# ML-MPJ1: Implementation of Basic Machine Learning Models

This project focuses on implementing fundamental machine learning models. The primary goals include generating synthetic datasets, working with real-world fault diagnosis datasets, and analyzing weather data using linear regression techniques.

## Project Description

### Technologies Used
- Python
- Google Colab
- scikit-learn

### Key Concepts
- Data preprocessing
- Linear regression
- Model evaluation

### Datasets
- CWRU Bearing Fault Diagnosis Dataset
- Weather in Szeged 2006-2016

## First Question

Using sklearn.datasets, generate a dataset with 1000 samples, 4 classes, and 3 features. Implement at least two ready-made Python linear layers from sklearn.linear_model.

### Steps:

1. Generate Dataset:
   - Use make_classification to generate a synthetic dataset with specified parameters.

2. Visualization:
   - Plot the generated dataset using matplotlib to visualize the distribution of classes.

3. Initialize Classifiers:
   - Logistic Regression with Cross-Validation:
     - Initialize LogisticRegressionCV with cross-validation and maximum iterations.
     - Wrap the logistic regression model with OneVsRestClassifier.
   - Stochastic Gradient Descent (SGD) Classifier:
     - Initialize SGDClassifier with hinge loss, L2 penalty, and maximum iterations.

## Second Question

Refer to the CWRU BEARING dataset, work with the fault diagnosis dataset. Perform the training and evaluation process using a Python linear classifier.

### Steps:

1. Data Normalization:
   - Normalize the training and test data using MinMaxScaler to scale the features between 0 and 1.

2. Logistic Regression Model:
   - Sigmoid Function:
     - Define the sigmoid function to map predicted values to probabilities.
   - Model:
     - Implement logistic regression to model the relationship between features and target variable.
   - Loss Function:
     - Use binary cross-entropy loss to measure the performance of the model.

3. Training and Evaluation:
   - Train the logistic regression model on the normalized training data.
   - Evaluate the model on the test data to assess its performance.

## Third Question

Consider a weather dataset called Weather in Szeged 2006-2016. Calculate the correlation matrix and scatter histogram of features. Perform LS (Least Squares) and RLS (Recursive Least Squares) estimations.

### Steps:

1. Correlation Matrix:
   - Calculate the correlation matrix to understand the relationships between features.
   - Visualize the correlation matrix using a heatmap.

2. Scatter Histogram:
   - Plot scatter histograms of features to visualize their distributions and relationships.

3. Linear Regression using Least Squares (LS):
   - Add a column of ones to the feature matrix to account for the intercept term.
   - Compute the coefficients using the least squares method.
   - Predict the target variable using the computed coefficients.

4. Recursive Least Squares (RLS):
   - RLS is an adaptive filter which updates the coefficients at each step based on new data points.
