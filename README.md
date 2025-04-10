<h1  align="center">Project 4: Cancer Prediction Using Machine Learning</h1>
<a name="readme-top"></a>

## Overview

This project applies machine learning techniques to predict cancer diagnoses using the [Cancer Prediction Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset/data) from Kaggle. The goal is to develop accurate classification models that distinguish between patients diagnosed with cancer and those without, demonstrating the potential of machine learning in healthcare analytics. 

Our objective was to build and evaluate classification models to determine whether a patient has cancer (`1`) or not (`0`), based on a range of clinical and lifestyle features.

## Dataset

**Source:** [Cancer Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset/data)

### Features

- **Age**: Integer (20–80)
- **Gender**: Binary (0 = Male, 1 = Female)
- **BMI**: Continuous (15–40)
- **Smoking**: Binary (0 = No, 1 = Yes)
- **GeneticRisk**: Categorical (0 = Low, 1 = Medium, 2 = High)
- **PhysicalActivity**: Continuous (hours per week, 0–10)
- **AlcoholIntake**: Continuous (units per week, 0–5)
- **CancerHistory**: Binary (0 = No, 1 = Yes)
- **Diagnosis**: Binary target (0 = No Cancer, 1 = Cancer)

### Target Variable

- **Diagnosis**: Indicates if a patient has cancer (`0` = No Cancer, `1` = Cancer)

### Data Distribution

The dataset is balanced and contains realistic variability across all features, supporting robust model training and evaluation.

---

## Exploratory Data Analysis (EDA)

- Checked for missing/null values
- Explored feature distributions
- Analyzed correlations with a heatmap
- Used count plots and box plots to examine relationships with the target

---

## Model Building

Multiple machine learning models were trained and evaluated to identify the best-performing classifier.

### Models Used

- Logistic Regression
- Random Forest Classifier

### Preprocessing

- Label encoding (e.g., for `Gender`)
- Feature scaling with `StandardScaler`
- Train-test split (80/20)

### Evaluation Metrics

- Accuracy
- Confusion Matrix
- Precision, Recall, F1-Score
- Feature Importance (for tree-based models)

---

## Results

- The **Random Forest** model achieved **94% accuracy**, showing strong overall performance with **high precision and recall** for both classes.
- **Macro F1-score**: 0.93  
- **Weighted F1-score**: 0.94  
- **Class 1 Recall (Cancer)**: 88%, indicating a slight opportunity to improve detection of positive cases.

### Top Predictive Features

<p align="center">
  <img src="https://github.com/clmj1727/Project4-MachineLearning/blob/main/Visualizations/Random%20Forest%20-%20Feature%20Importance.png" alt="Random Forest: Feature Importance" width="700">
</p>

- **Alcohol Intake**, **BMI**, and **Cancer History** were the most influential predictors.
- **Age** and **Physical Activity** also played notable roles, suggesting the impact of lifestyle and age-related factors.
- **Genetic Risk** remained important but was not the top predictor, possibly due to interactions with other features.
- **Smoking** and **Gender** showed lower importance, implying limited predictive power in this dataset.

**Key Takeaway**: Lifestyle factors—especially alcohol intake, BMI, and personal cancer history—are the most impactful in predicting cancer risk in this model, highlighting the role of health behaviors in prevention and early detection.

---

## Visualizations Included

- Correlation heatmap
- Confusion matrices for each model
- Feature importance plots
- Distribution plots of key features

---

## Deliverables

- `CancerNotebook.ipynb`: Jupyter notebook with full analysis
- `README.md`: Project documentation
- `Predicting Cancer Risk`: Slide deck

---

## Future Work

- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- Test additional models (e.g., Gradient Boosting)
- Deploy the model via a web app (e.g., Flask or Streamlit)
- Explore multiclass classification using additional labels (e.g., risk levels)

---

## References

- [Cancer Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset/data)
- Scikit-learn Documentation
- Matplotlib & Seaborn Libraries
<p  align="center">(<a  href="#readme-top">back to top</a>)</p>
