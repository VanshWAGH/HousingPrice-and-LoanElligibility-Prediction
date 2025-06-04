# 🏠 House Price & Loan Eligibility Prediction using Machine Learning

This project applies machine learning techniques to solve two practical problems in the real estate and financial domain:

1. **House Price Prediction**  
2. **Loan Eligibility Prediction**

Using real-world datasets, this project builds accurate and explainable models to support data-driven decision-making for buyers and financial institutions.

---

## 📌 Project Summary

This project involves:
- Predicting **house prices** using regression algorithms.
- Assessing **loan eligibility** using classification models.
- Applying data preprocessing, feature engineering, and model evaluation techniques.
- Deploying the final models with a user-friendly interface using **Flask/Streamlit**.

---

## 🏡 Dataset Descriptions

### 📁 House Price Prediction Dataset
- **Entries**: 21,613  
- **Target Variable**: `Sale Price`  
- **Key Features**:
  - Number of Bedrooms/Bathrooms
  - Flat Area, Lot Area
  - Condition, Grade, Latitude, Longitude, Zipcode
- **Missing Values**: Sale price (4), structural and location attributes.

### 💳 Loan Eligibility Dataset
- **Entries**: 614  
- **Target Variable**: `Loan Status` (Y/N)  
- **Key Features**:
  - ApplicantIncome, CoapplicantIncome, LoanAmount, Credit History
  - Property Area, Loan Purpose
- **Missing Values**: LoanAmount (22), Credit History (50), others.

---

## 🛠️ Methodology

### 🏡 House Price Prediction
#### Preprocessing:
- Imputed missing values (mean/mode)
- Feature scaling
- One-hot encoding for categorical data

#### Models Used:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **XGBoost Regressor**

#### Metrics:
- MAE, RMSE, R² Score

| Model               | R² Score |
|---------------------|----------|
| Linear Regression   | 0.78     |
| Decision Tree       | 0.85     |
| Random Forest       | **0.91** |
| XGBoost             | 0.89     |

---

### 💳 Loan Eligibility Prediction
#### Preprocessing:
- Imputed missing values (mode)
- Feature normalization
- Label and one-hot encoding

#### Models Used:
- **Random Forest Classifier**
- **XGBoost Classifier**
- **LightGBM Classifier**

#### Metrics:
- Accuracy, Precision, Recall, F1-Score

| Model               | Accuracy |
|---------------------|----------|
| Random Forest       | 87%      |
| XGBoost             | 85%      |
| LightGBM (LGBM)     | **88%**  |

---

## 🚀 Deployment

The trained models are deployed using **Flask** or **Streamlit**, allowing users to:
- Input house or applicant details
- Get real-time predictions for:
  - Estimated house price
  - Loan approval status

---

## ✅ Conclusion

- **Random Forest** was the best model for house price prediction.
- **LightGBM** outperformed others in loan eligibility classification.
- The models are reliable, interpretable, and accessible via an interactive frontend.

### 🔮 Future Work:
- Use real-time API data
- Hyperparameter tuning with GridSearchCV
- Add model explainability (SHAP, LIME)

---

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Kaggle Housing Dataset](https://www.kaggle.com/datasets)
- [Flask Framework](https://flask.palletsprojects.com/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)

## How to run this
- Genrate pkl model file by running both ipynb files.

