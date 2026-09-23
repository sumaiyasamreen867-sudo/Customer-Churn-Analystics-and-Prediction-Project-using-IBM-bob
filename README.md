# 📡 Telco Customer Churn Analytics & Prediction

A complete end-to-end machine learning project that analyses customer churn patterns for a telecommunications company and builds predictive models to identify customers at risk of churning. Implemented as an interactive Streamlit web application.

---

## 📋 Business Problem

Customer churn — when a customer stops doing business with a company — is a critical challenge in the telecommunications industry. Acquiring a new customer is significantly more expensive than retaining an existing one. This project aims to:

1. Understand the patterns and drivers associated with customer churn in this dataset.
2. Build predictive models to identify customers at risk.
3. Provide an interactive interface for real-time churn probability estimation.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | Telco Customer Churn dataset (Kaggle) |
| **File** | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| **Records** | 7,043 customers |
| **Features** | 21 (including target) |
| **Target** | `Churn` (Yes / No) |
| **Overall Churn Rate** | **26.54%** |

### Key Features
- **Demographics:** gender, SeniorCitizen, Partner, Dependents
- **Services:** PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Account:** tenure, Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

### Data Quality
- 11 rows had blank `TotalCharges` (all had `tenure = 0`); imputed with `MonthlyCharges`.
- `customerID` excluded from all modelling (identifier, not a predictive feature).
- No other missing values found.

---

## 🏗️ Methodology

```
Raw CSV → Data Validation → Cleaning & Imputation → EDA & Visualisation
    → Feature Engineering → Label Encoding → Train/Test Split (80/20, stratified)
    → StandardScaler (for scaled models) → Model Training → Evaluation
    → Feature Importance → Interactive Prediction Interface
```

### Preprocessing Pipeline
- Categorical features label-encoded using `LabelEncoder` fitted only on training data.
- Numeric features scaled with `StandardScaler` fitted only on training data.
- The same fitted encoders and scaler are applied to the prediction interface — no leakage.
- `customerID` dropped before any modelling step.

### Feature Engineering
| Feature | Description |
|---|---|
| `Tenure_Band` | tenure grouped: 0–12, 13–24, 25–48, 49–72 months |
| `Charge_Band` | MonthlyCharges grouped: Low, Mid, High, Very High |
| `Num_Services` | Count of add-on services subscribed |
| `Charge_Per_Month` | TotalCharges / (tenure + 1) |

> Note: Engineered band/count features are used for EDA visualisations only and **not** fed to the classifier, avoiding data leakage.

---

## 🤖 Models

Five classification algorithms were trained and evaluated:

| Model | Notes |
|---|---|
| Logistic Regression | Linear baseline; uses scaled features |
| Decision Tree | Interpretable tree model; max depth = 6 |
| Random Forest | Ensemble of 200 trees |
| Gradient Boosting | Sequential boosting ensemble, 200 estimators |
| K-Nearest Neighbors | k = 7; uses scaled features |

---

## 📈 Model Results (Test Set — 80/20 Stratified Split)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7984 | 0.6406 | 0.5481 | 0.5908 | 0.8404 |
| Decision Tree | 0.7842 | 0.6036 | 0.5455 | 0.5730 | 0.8246 |
| Random Forest | 0.7885 | 0.6301 | 0.4920 | 0.5526 | 0.8256 |
| **Gradient Boosting** | **0.7999** | **0.6554** | **0.5187** | **0.5791** | **0.8415** |
| K-Nearest Neighbors | 0.7530 | 0.5374 | 0.5000 | 0.5180 | 0.7745 |

🏆 **Best model:** Gradient Boosting (ROC-AUC = 0.8415)

---

## 💡 Business Insights

> All findings describe *associations* observed in this dataset. Correlation does not imply causation.

1. **Overall Churn Rate:** 26.54% — 1,869 out of 7,043 customers churned.

2. **Highest-Churn Segments:** Month-to-month contract holders (~43%), Fiber optic internet users (~42%), electronic check payers (~45%), and short-tenure customers (0–12 months).

3. **Tenure & Monthly Charges:** Churned customers have shorter average tenure (18.0 vs 37.6 months) and higher average monthly charges ($74.44 vs $61.27).

4. **Contract Type:** Month-to-month: 42.7% churn | One year: 11.3% | Two year: 2.8% — the sharpest differentiator observed.

5. **Internet & Payment:** Fiber optic associated with 41.9% churn vs 19.0% for DSL and 7.4% for no internet service. Electronic check payment associated with ~45% churn vs ~15% for automatic methods.

6. **Add-on Services:** Customers without OnlineSecurity, TechSupport, DeviceProtection, or OnlineBackup show considerably higher observed churn rates.

7. **Feature Importance:** `tenure`, `TotalCharges`, `MonthlyCharges`, and `Contract` are the most important predictive features across tree-based models.

---

## ⚠️ Limitations

- This is a single observational snapshot — temporal trends cannot be determined.
- Class imbalance (~73% No / ~27% Yes) means accuracy alone is misleading; ROC-AUC and F1 are more informative.
- Associations observed in this dataset do not establish causal relationships.
- No hyperparameter optimisation was performed (fixed seeds used for reproducibility).
- The dataset is a publicly available Kaggle sample and may not fully reflect real-world telecom distributions.

---

## 🚀 Installation & Usage

### Prerequisites
Python 3.9+

### Setup
```bash
# Clone or download the project
cd telco-churn

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

### The CSV file
Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the **same directory** as `app.py`.

---

## 🗂️ Project Structure

```
telco-churn/
├── app.py                              # Complete Streamlit application (all Python code)
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── project_report.docx                 # Detailed professional project report
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (required)
```

---

## 🖥️ Application Pages

| Page | Description |
|---|---|
| 🏠 Overview & KPI Dashboard | Key metrics, churn distribution, high-level charts |
| 🔍 Data Explorer | Raw data, validation stats, filtering, descriptive stats |
| 📊 Exploratory Data Analysis | Demographics, services, charges, correlation matrix |
| 💡 Business Insights | Answers to 6 business questions with interactive charts |
| ⚙️ Feature Engineering | Engineered features, visualisations, model feature set |
| 🤖 Model Training & Evaluation | Per-model metrics, confusion matrix, ROC curve, feature importance |
| 📈 Model Comparison | Side-by-side metrics, radar chart, combined ROC curves |
| 🔮 Customer Prediction | Interactive form for real-time churn prediction with probability gauge |

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| streamlit | Web application framework |
| pandas | Data manipulation |
| numpy | Numerical computation |
| scikit-learn | Machine learning models and preprocessing |
| plotly | Interactive visualisations |
| matplotlib | Static plotting support |
| seaborn | Statistical visualisations |
| python-docx | Word report generation |

---

*Dataset: Telco Customer Churn — sourced from Kaggle (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)*
