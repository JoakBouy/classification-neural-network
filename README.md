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

## Discussion on Optimization Techniques

### Overview
In this project, several optimization techniques were employed to enhance the performance and generalization ability of the neural network models. These techniques help mitigate overfitting, improve convergence, and achieve better accuracy on unseen data.

### Dropout Layers
**Principle:**
Dropout is a regularization technique that randomly sets a fraction of the input units to 0 at each update during training time, which helps prevent overfitting.

**Relevance to the Project:**
Incorporating dropout layers in the neural network (Model 2) helps in reducing overfitting by ensuring that the model does not rely too heavily on any individual neurons. This promotes the development of robust features that generalize well to new data.

**Parameters:**
- Dropout rate: 0.3

**Justification:**
A dropout rate of 0.3 was chosen based on empirical evidence from literature and experimentation. This value strikes a balance by providing sufficient regularization without overly hindering the learning capacity of the network.

### Learning Rate Scheduling
**Principle:**
Learning rate scheduling involves adjusting the learning rate during training. A common approach is to reduce the learning rate as training progresses, which allows for more precise adjustments to the model weights.

**Relevance to the Project:**
Using a learning rate schedule helps in fine-tuning the model during later stages of training, which can lead to better convergence and improved performance. It prevents the model from overshooting the optimal point and helps in settling into a minimum.

**Parameters:**
- Initial learning rate: Custom value (e.g., 0.001)
- Learning rate reduction: Applied based on validation performance.

**Justification:**
The initial learning rate was set to a typical value used in practice. The learning rate is reduced when the validation performance plateaus, ensuring that the model makes finer adjustments as it nears convergence.

### Early Stopping
**Principle:**
Early stopping monitors the model's performance on a validation set and stops training when performance stops improving, thereby preventing overfitting.

**Relevance to the Project:**
Early stopping is crucial for avoiding overfitting, especially when training deep neural networks. It helps in identifying the optimal point for halting training, ensuring that the model retains good generalization capability.

**Parameters:**
- Patience: Number of epochs to wait for an improvement before stopping (e.g., 10 epochs).

**Justification:**
A patience value of 10 epochs was chosen to allow the model sufficient opportunity to improve after a plateau. This value was selected based on experimentation and typical values used in the field.

### Summary of Parameter Selection
The parameter values for each optimization technique were selected through a combination of literature review, empirical testing, and cross-validation. The aim was to find a configuration that provides a good trade-off between model complexity, training time, and performance on the validation set.

By employing these optimization techniques, the project aims to develop neural network models that are robust, have good generalization capabilities, and perform well on the test data.

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
