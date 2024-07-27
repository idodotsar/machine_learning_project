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

#### Distribution of Features

The following plot shows the distribution of each feature, separated by the `Outcome` (0 for non-diabetic and 1 for diabetic). This helps in understanding how each feature varies between diabetic and non-diabetic patients.

![Distribution Plots](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/Distribution%20plots.png)

- **Pregnancies**: Diabetic patients tend to have more pregnancies than non-diabetic patients.
- **Glucose**: Higher glucose levels are strongly associated with diabetes.
- **Blood Pressure**: The distribution is quite similar for both diabetic and non-diabetic patients.
- **Skin Thickness**: Slightly higher values are observed in diabetic patients.
- **Insulin**: Diabetic patients show a wider range of insulin levels, including very high values.
- **BMI**: Higher BMI is associated with diabetes.
- **DiabetesPedigreeFunction**: Slightly higher values in diabetic patients, indicating a familial influence.
- **Age**: Older age is associated with a higher likelihood of diabetes.

#### Correlation Matrix

The correlation matrix below shows the pairwise correlations between features. The color intensity indicates the strength of the correlation, with positive correlations in red and negative correlations in blue. This helps in identifying features that are highly correlated with each other and with the outcome.

![Correlation Matrix](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/corr_matrix.png)

- **Glucose**: Shows the highest positive correlation with the `Outcome` (0.49), making it a significant predictor of diabetes.
- **BMI**: Also has a notable positive correlation with the `Outcome` (0.31).
- **Age**: Shows a moderate positive correlation with the `Outcome` (0.24).
- **Pregnancies**: Displays a moderate correlation with the `Outcome` (0.25).
- **Insulin**: Has a lower correlation with the `Outcome` (0.18), but still significant.
- **Other features**: Blood Pressure, Skin Thickness, and DiabetesPedigreeFunction have weaker correlations with the outcome.

### Feature Engineering and Data Preprocessing

The dataset was prepared for model training through the following steps:

1. **Scaling Features**: The features were standardized to ensure they are on the same scale, which helps in improving the performance of certain machine learning algorithms.
   ```python
   from sklearn.preprocessing import StandardScaler

   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```
2. **Splitting the Data**: The dataset was split into training and testing sets to evaluate the performance of the models.
   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
   ```
### Machine Learning Models

The following models were trained and evaluated:

1. **Logistic Regression**
   - Linear model for binary classification
2. **Decision Tree**
   - Non-linear model that splits data into subsets
3. **Random Forest**
   - Ensemble method of multiple decision trees

### Data Split

The following plot shows the distribution of diabetic and non-diabetic patients in the dataset.

![Data Split](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/data_split.png)

### Model Evaluation

The performance of the models was evaluated using various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. The following plot shows the precision scores of different models using cross-validation.

![Final Results](https://github.com/idodotsar/machine_learning_project/blob/616c71e97c1045d74ae054f8e66304f0e13ef267/images/final%20results.png)

- **Decision Tree (Default)**: Baseline model using default parameters.
- **Decision Tree (Hyperparameter Tuning)**: Improved performance by tuning hyperparameters.
- **Logistic Regression (Default)**: Baseline logistic regression model.
- **Logistic Regression (CV)**: Logistic regression model with cross-validation.
- **Logistic Regression (Hyperparameter Tuning)**: Further improved logistic regression model with hyperparameter tuning.
- **Random Forest (Default)**: Baseline random forest model.
- **Random Forest (Randomized Parameters)**: Improved random forest model by randomizing parameters.
- **Random Forest (Hyperparameter Tuning)**: Further improved random forest model with hyperparameter tuning.
- **Random Forest (Scaled)**: Final model evaluated on the test set, achieving a precision score of 0.801432.

### Best Performing Model

According to the precision scores from cross-validation, the best-performing model was the **Random Forest with Cross-Validation (CV)**, which achieved a precision score of 0.832881. This model was then evaluated on the test set as the **Random Forest (Scaled)**, which achieved a precision score of 0.801432, indicating its effectiveness in correctly identifying diabetic patients while minimizing false positives.

## Outputs

The outputs of the code include:
- Data visualizations (e.g., distribution plots, correlation heatmaps)
- Model performance metrics (e.g., accuracy, precision, recall, F1-score, ROC-AUC)
- Predictions on new data

## Usage

To run the project, execute the provided Jupyter notebook. Ensure all dependencies are installed as specified in the `requirements.txt` file.

## Conclusion

This project demonstrates the application of various machine learning models to predict diabetes, highlighting the importance of data preprocessing, feature engineering, and model evaluation in building accurate predictive models. The Random Forest model with Cross-Validation (CV) was the best performer during validation, achieving the highest precision score, and the Random Forest (Scaled) model performed effectively on the test set.
