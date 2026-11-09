# Day 16 Capstone - Customer Churn Prediction

## Project Overview

This capstone builds an end-to-end machine learning pipeline to predict customer churn. The project focuses on preventing data leakage, applying preprocessing correctly, comparing multiple classification models, tuning the selected model, evaluating it with suitable metrics, and translating the results into an actionable business recommendation.

The complete work is contained in `day16_capstone.ipynb`.

## Objective

Predict whether a customer will churn (`churn = 1`) using customer account and plan information available at prediction time.

Because missing a customer who is likely to churn can be costly, the project gives special attention to **recall for the churn class**. **ROC-AUC** is used for model comparison and selection.

## Dataset

The dataset is generated synthetically inside the notebook, so no external CSV file or internet connection is required.

It contains approximately 4,000 customer records with:

- Numeric features such as tenure, monthly charges, support tickets, latency, add-ons, days since last login, and total charges
- Categorical features such as contract type, payment method, internet service, and dependents
- Binary target: `churn`
- A deliberately introduced leakage feature: `exit_survey_sent`

## Project Workflow

### 1. Exploratory Data Analysis

- Examine churn class balance
- Compare numeric features between customers who stay and customers who churn
- Analyse churn rate across contract types
- Visualise useful feature relationships

### 2. Leakage Detection and Data Splitting

- Detect suspicious features associated too strongly with the target
- Remove `exit_survey_sent`
- Split the dataset using stratification into:
  - 70% training data
  - 15% validation data
  - 15% test data

### 3. Preprocessing Pipeline

A scikit-learn `ColumnTransformer` and `Pipeline` are used so preprocessing is learned from training data only.

Numeric features:

- `SimpleImputer`
- `StandardScaler`

Categorical features:

- `SimpleImputer`
- `OneHotEncoder(handle_unknown="ignore")`

This design helps prevent preprocessing leakage and applies the same transformations consistently during prediction.

### 4. Model Comparison

Four classification models are compared using 5-fold stratified cross-validation and ROC-AUC:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machine with RBF kernel

### 5. Hyperparameter Tuning

The Gradient Boosting pipeline is tuned using `GridSearchCV` with parameters including:

- Number of estimators
- Learning rate
- Maximum tree depth

### 6. Model Evaluation and Threshold Selection

The final model is evaluated using:

- ROC-AUC
- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix

Several decision thresholds are compared. A threshold around `0.30` is used to favour churn recall over the default `0.50` threshold.

### 7. Model Interpretation

The project uses:

- Permutation importance for the tuned Gradient Boosting model
- Logistic Regression coefficients for directional interpretation

These techniques help identify features that increase or decrease churn risk.

### 8. Business Recommendation and Reflection

The notebook concludes with:

- A short recommendation for a non-technical retention manager
- Reflection on data leakage
- Discussion of precision versus recall
- Model-selection reasoning
- Explanation of why the test set must remain untouched until final evaluation

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- Jupyter Notebook

## Main File

```text
day16_capstone.ipynb
```

## How to Run

1. Open `day16_capstone.ipynb` in Jupyter Notebook, JupyterLab, or VS Code with the Jupyter extension.
2. Select a Python environment containing NumPy, Pandas, Matplotlib, and scikit-learn.
3. Restart the kernel.
4. Run all cells from top to bottom.
5. Confirm that every cell executes without an error and review the generated metrics and plots.

If the required packages are missing, install them in the same Python environment used by the notebook:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Key Learning Outcomes

This capstone demonstrates how to:

- Frame churn prediction as a supervised binary-classification problem
- Detect and remove target leakage
- Maintain separate training, validation, and test datasets
- Build reusable preprocessing and modelling pipelines
- Handle numeric and categorical features correctly
- Compare models using cross-validation
- Tune models with `GridSearchCV`
- Evaluate an imbalanced classifier using business-relevant metrics
- Adjust the decision threshold based on precision-recall trade-offs
- Interpret model predictions and communicate results to business stakeholders

## Final Goal

The final result is a reproducible churn-prediction workflow in which preprocessing and modelling are combined into a single scikit-learn pipeline, reducing leakage risk and making the model safer to evaluate and deploy.
