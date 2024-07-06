# Cardiovascular Disease Prediction Model

## Overview
This project implements and compares two neural network models for predicting cardiovascular disease based on various health metrics. The first model is a simple neural network, while the second incorporates multiple optimization techniques to improve performance and generalization.

## Dataset
The dataset used in this project contains various health metrics and a binary indicator for the presence of cardiovascular disease. Key features include:

- Age (in years)
- Gender
- Height (in cm)
- Weight (in kg)
- Systolic blood pressure (ap_hi)
- Diastolic blood pressure (ap_lo)
- Cholesterol levels
- Glucose levels
- Smoking status
- Alcohol intake
- Physical activity level

The target variable is 'cardio', indicating the presence (1) or absence (0) of cardiovascular disease.

## Data Preprocessing
1. Age is converted from days to years.
2. Outliers in blood pressure readings are removed.
3. Features are standardized using StandardScaler.
4. Data is split into training (70%), validation (15%), and test (15%) sets.

## Models

### Model 1: Simple Neural Network
- Architecture: Input layer (based on feature count) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(1, Sigmoid)
- Optimizer: Default Adam optimizer
- Loss function: Binary crossentropy
- Metrics: Accuracy

### Model 2: Optimized Neural Network
- Architecture: Input layer → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Optimizer: Adam with custom learning rate
- Loss function: Binary crossentropy
- Metrics: Accuracy

Optimization techniques:
1. Dropout layers to prevent overfitting
2. Learning rate scheduling
3. Early stopping

## Training
Both models are trained for up to 100 epochs with a batch size of 32. Model 2 uses early stopping to prevent overfitting.

## Evaluation
Models are evaluated on the test set, and their performance is compared based on test accuracy.

## Visualization
Training and validation accuracy and loss are plotted for both models to visualize their learning progress and compare their performance.

## Dependencies
- pandas
- numpy
- scikit-learn
- tensorflow
- matplotlib

## Usage
1. Ensure all dependencies are installed.
2. Update the file path in the code to point to your dataset.
3. Run the script to train and evaluate both models.
4. Review the printed test accuracies and the generated plots to compare model performance.

## Future Improvements
- Experiment with different model architectures
- Implement k-fold cross-validation
- Try other optimization techniques like batch normalization or different optimizers
- Perform feature importance analysis
- Implement hyperparameter tuning

## Author
Joak Buoy Gai - b.joak@alustudent.com
