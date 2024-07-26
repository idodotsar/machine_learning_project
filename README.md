# Diabetes Prediction Project

## Overview

This project aims to predict the likelihood of diabetes in patients using a variety of medical predictor variables. The analysis involves several steps, including data cleaning, exploratory data analysis (EDA), feature engineering, data preprocessing, and training multiple machine learning models.

## Dataset

The dataset used in this project contains the following features:
- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **Blood Pressure**: Diastolic blood pressure (mm Hg)
- **Skin Thickness**: Triceps skinfold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history
- **Age**: Age (years)
- **Outcome**: Class variable (0 or 1, where 1 indicates diabetes)

## Data Analysis

### Data Cleaning

Missing values and outliers were handled appropriately to ensure the quality of the data.

### Exploratory Data Analysis (EDA)

Visualizations and statistical analyses were performed to understand the distribution and relationships between variables. Key findings include:
- Distribution plots of features
- Correlation heatmaps
- Pair plots

![Distribution Plots]([https://github.com/yourusername/yourrepository/raw/main/Distribution%20plots.png](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/Distribution%20plots.png))
![Correlation Matrix]([https://github.com/yourusername/yourrepository/raw/main/corr_matrix.png](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/corr_matrix.png))

### Feature Engineering

New features were created, and existing features were transformed to enhance model performance.

### Data Preprocessing

The data was normalized and categorical variables were encoded to prepare for model training.

## Machine Learning Models

The following models were trained and evaluated:

1. **Logistic Regression**
   - Linear model for binary classification
2. **Decision Tree**
   - Non-linear model that splits data into subsets
3. **Random Forest**
   - Ensemble method of multiple decision trees
4. **Support Vector Machine (SVM)**
   - Linear model that finds the optimal separating hyperplane
5. **K-Nearest Neighbors (KNN)**
   - Non-parametric method based on majority class among k nearest neighbors
6. **Neural Networks**
   - Layers of interconnected neurons inspired by the human brain

## Process Overview

1. **Load the Data**: Imported the dataset into the environment.
2. **Data Cleaning**: Handled missing values and outliers.
3. **Exploratory Data Analysis (EDA)**: Visualized data distributions and relationships.
4. **Feature Engineering**: Created and transformed features.
5. **Data Preprocessing**: Normalized and encoded features.
6. **Model Training**: Trained multiple machine learning models.
7. **Model Evaluation**: Assessed model performance using accuracy, precision, recall, F1-score, and ROC-AUC.
8. **Model Tuning**: Optimized models through cross-validation and hyperparameter tuning.
9. **Model Deployment**: Deployed the best-performing model for predictions.

## Outputs

The outputs of the code include:
- Data visualizations (e.g., distribution plots, correlation heatmaps)
- Model performance metrics (e.g., accuracy, precision, recall, F1-score, ROC-AUC)
- Predictions on new data

![Data Split]([https://github.com/yourusername/yourrepository/raw/main/data_split.png](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/data_split.png))
![Final Results]([https://github.com/yourusername/yourrepository/raw/main/final%20results.png](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/final%20results.png))

## Usage

To run the project, execute the provided Jupyter notebook. Ensure all dependencies are installed as specified in the `requirements.txt` file.

## Conclusion

This project demonstrates the application of various machine learning models to predict diabetes, highlighting the importance of data preprocessing, feature engineering, and model evaluation in building accurate predictive models.
