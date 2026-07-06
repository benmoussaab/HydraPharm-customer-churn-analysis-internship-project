# Customer Churn Analytics Platform

An end-to-end customer churn analytics platform built using **Python**, **LightGBM**, **XGBoost**, and **Streamlit**. The application transforms raw customer transaction data into actionable business insights and predicts customers at risk of churn.

---

# Features

## Data Import

- Upload CSV or Excel files
- Automatic data validation
- Sample dataset download
- Supports customer transaction datasets

---

## Data Preprocessing

- Automatic date parsing
- Monthly customer aggregation
- Removal of invalid transactions
- Customer-level feature generation

---

## Feature Engineering

The application automatically generates analytical features including:

### Spending Features

- Total spending
- Spending by product category
- Cumulative spending
- Rolling spending (2–6 months)
- Average monthly spending
- Spending standard deviation
- Month-over-month spending change
- Spending trend (Linear Regression slope)

### Purchase Features

- Total quantity purchased
- Quantity by category
- Cumulative quantities
- Rolling quantity features

### Customer Activity Features

- Invoice count
- Average invoices
- Invoice standard deviation
- Days between purchases
- Customer inactivity period
- Churn label generation

---

## Machine Learning

- LightGBM Classifier
- Configurable feature selection
- Train/Test evaluation mode
- Future churn prediction mode

### Class Imbalance Handling

- Random Under Sampling
- SMOTE Oversampling

---

## Model Evaluation

The application provides:

- Accuracy
- ROC-AUC Score
- Classification Report
- Confusion Matrix
- Feature Importance Ranking

---

## Interactive Dashboard

The Streamlit interface allows users to:

- Upload datasets
- Select prediction month
- Choose training features
- Select sampling strategy
- Train the model
- Predict customer churn
- Visualize feature importance
- Display confusion matrix
- Filter customers by churn probability
- Export predictions to CSV
- Export predictions to Excel

---

## Business Insights

The platform identifies:

- Customers likely to churn
- High-risk customer segments
- Spending behavior trends
- Customer purchase patterns
- Key churn drivers

---

# Project Results

Using one year of customer transactional data, the final model achieved:

- **92.7% Accuracy**
- **71.2% Recall** for churn detection
- Interactive customer-level churn prediction
- Automated feature importance analysis

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- Streamlit
- Matplotlib
- Seaborn
- OpenPyXL

---

# Application Workflow

```
Customer Transactions
        │
        ▼
Data Cleaning
        │
        ▼
Monthly Aggregation
        │
        ▼
Feature Engineering
        │
        ▼
Sampling (SMOTE / UnderSampling)
        │
        ▼
LightGBM Training
        │
        ▼
Model Evaluation
        │
        ▼
Customer Churn Prediction
        │
        ▼
Business Reports & Export
```

---

# Output

The application generates:

- Customer churn probability
- Predicted churn status
- Feature importance analysis
- Model performance metrics
- Downloadable CSV reports
- Downloadable Excel reports

---

# Future Improvements

- SHAP Explainability
- Hyperparameter Optimization
- Power BI Integration
- Customer Segmentation Dashboard
- Automated Monthly Retraining
- Cloud Deployment
