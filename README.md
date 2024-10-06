# Cancer Diagnosis Using Artificial Neural Networks

## Overview
This project explores the potential of Artificial Neural Networks (ANN) in cancer diagnosis at **MD Anderson Cancer Institute**. The goal is to develop an ANN model using **TensorFlow** to classify cancer as benign or malignant based on radiological data. This model aims to assist radiologists in making more accurate and faster diagnoses.

## Features:
- **Preprocessed Dataset:** Radiological data was cleaned, normalized, and split for training and testing.
- **ANN Model:** Built using TensorFlow with multiple hidden layers, ReLU activation, and dropout for regularization.
- **Model Evaluation:** Performance metrics include accuracy, precision, recall, and F1-score.
- **Early Detection Aid:** Helps identify high-risk patients who may require immediate attention.

## Technologies Used:
- **TensorFlow/Keras** for building the ANN model.
- **Python** for data preprocessing and model development.
- **StandardScaler** for data normalization.
- **Performance Metrics:** Accuracy, Precision, Recall, F1-Score.

## Model Performance:
The optimized ANN model achieved:
- **Accuracy:** 98.24%
- **Precision:** 100%
- **Recall:** 95.35%
- **F1-Score:** 97.61%

## Potential Applications:
- **Early Detection:** Aids in early identification of malignant cases, improving patient outcomes.
- **Screening Tool:** Can be integrated into healthcare systems for efficient initial screenings.
- **Resource Allocation:** Assists healthcare providers in prioritizing high-risk cases.

## Future Improvements:
- Experiment with additional layers, neurons, or different activation functions.
- Explore data augmentation techniques to improve model generalization.
- Implement feature engineering to capture complex data patterns.

## Installation:
1. Clone this repository.
2. Install required libraries using `pip install -r requirements.txt`.
3. Run the `cancer.py` script to train and evaluate the model.
