import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pandas.api.types import is_period_dtype
import uuid
import io

# Set page config
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
if 'total_features' not in st.session_state:
    st.session_state.total_features = 0
if 'preprocessed_data' not in st.session_state:
    st.session_state.preprocessed_data = None
if 'available_features' not in st.session_state:
    st.session_state.available_features = []
if 'selected_month' not in st.session_state:
    st.session_state.selected_month = None
if 'selected_latest_month' not in st.session_state:
    st.session_state.selected_latest_month = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = {}  # Store predictions by month
if 'available_months' not in st.session_state:
    st.session_state.available_months = []

# Title and description
st.title("Customer Churn Prediction App")
st.write("Choose a mode: Test Mode to evaluate predictions for a specific month, or Prediction Mode to predict churn for the next month.")

# Add File Format Information
st.subheader("Required File Format (CSV or Excel)")
st.markdown("""
To use this app, please upload a CSV or Excel file with the following columns:

| Column Name | Description | Data Type | Example |
|-------------|-------------|-----------|---------|
| UUID        | Unique customer identifier | String | ABC123 |
| DATE TRANS  | Transaction date | Date (YYYY-MM-DD) | 2024-03-15 |
| CA C        | Revenue from category C | Float | 100.50 |
| CA G        | Revenue from category G | Float | 200.75 |
| CA P        | Revenue from category P | Float | 150.25 |
| Q C         | Quantity from category C | Float | 10.0 |
| Q G         | Quantity from category G | Float | 5.0 |
| Q P         | Quantity from category P | Float | 8.0 |
| CA QUOTA    | Quota revenue | Float | 50.00 |
| QTE QUOTA   | Quota quantity | Float | 2.0 |

You can download a sample CSV or Excel file below to see the expected format.
""")

# Create a sample DataFrame for download
sample_data = pd.DataFrame({
    'UUID': ['ABC123', 'DEF456', 'GHI789'],
    'DATE TRANS': ['2024-03-15', '2024-03-16', '2024-04-01'],
    'CA C': [100.50, 0.00, 300.25],
    'CA G': [200.75, 150.00, 0.00],
    'CA P': [150.25, 200.50, 100.00],
    'Q C': [10.0, 0.0, 15.0],
    'Q G': [5.0, 8.0, 0.0],
    'Q P': [8.0, 12.0, 6.0],
    'CA QUOTA': [50.00, 75.00, 25.00],
    'QTE QUOTA': [2.0, 3.0, 1.0]
})

# Offer sample CSV and Excel for download
col1, col2 = st.columns(2)
with col1:
    sample_csv = sample_data.to_csv(index=False)
    st.download_button(
        label="Download Sample CSV",
        data=sample_csv,
        file_name="sample_customer_data.csv",
        mime="text/csv"
    )
with col2:
    excel_buffer = io.BytesIO()
    sample_data.to_excel(excel_buffer, index=False, engine='openpyxl')
    excel_buffer.seek(0)
    st.download_button(
        label="Download Sample Excel",
        data=excel_buffer,
        file_name="sample_customer_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# Function to validate YYYY-MM format
def is_valid_month(month_str):
    pattern = r'^\d{4}-(0[1-9]|1[0-2])$'
    return bool(re.match(pattern, month_str))

# Function to compute slope for rolling features
def compute_slope(series):
    if series.isnull().sum() > 0 or len(series) < len(series.dropna()):
        return np.nan
    X = np.arange(len(series)).reshape(-1, 1)
    y = series.values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    return model.coef_[0][0]

# Data preprocessing function
def process_data(df):
    df = df.drop(columns='Unnamed: 0', errors='ignore')
    df["DATE TRANS"] = pd.to_datetime(df["DATE TRANS"])
    df["month"] = df["DATE TRANS"].dt.to_period("M")
    
    cols = ['CA C', 'CA G', 'CA P', 'Q C', 'Q G', 'Q P']
    df_filtered = df[~(df[cols] <= 0).all(axis=1)].copy()
    
    # Monthly aggregation
    monthly_summary = df_filtered.groupby(["UUID", "month"]).agg(
        total_spend_c=("CA C", "sum"),
        total_spend_p=("CA P", "sum"),
        total_spend_g=("CA G", "sum"),
        total_spend_cota=("CA QUOTA", "sum"),
        total_quantity_cota=("QTE QUOTA", "sum"),
        total_quantity_c=("Q C", "sum"),
        total_quantity_p=("Q P", "sum"),
        total_quantity_g=("Q G", "sum"),
        invoice_count=("UUID", "count"),
        first_invoice_date=("DATE TRANS", "min"),
        last_invoice_date=("DATE TRANS", "max")
    ).reset_index()
    
    ds = monthly_summary.sort_values(['UUID', 'month'])
    ds['next_month_first_date'] = ds.groupby('UUID')['first_invoice_date'].shift(-1)
    ds['gap_days'] = (ds['next_month_first_date'] - ds['last_invoice_date']).dt.days - 1
    monthly_summary = ds.copy()
    
    # Cumulative sums
    monthly_summary["cum_spend_c"] = monthly_summary.groupby("UUID")["total_spend_c"].cumsum()
    monthly_summary["cum_spend_p"] = monthly_summary.groupby("UUID")["total_spend_p"].cumsum()
    monthly_summary["cum_spend_g"] = monthly_summary.groupby("UUID")["total_spend_g"].cumsum()
    monthly_summary["cum_quantity_c"] = monthly_summary.groupby("UUID")["total_quantity_c"].cumsum()
    monthly_summary["cum_quantity_p"] = monthly_summary.groupby("UUID")["total_quantity_p"].cumsum()
    monthly_summary["cum_quantity_g"] = monthly_summary.groupby("UUID")["total_quantity_g"].cumsum()
    monthly_summary["cum_invoice_count"] = monthly_summary.groupby("UUID")["invoice_count"].cumsum()
    
    # Rolling features
    monthly_summary = monthly_summary.sort_values(by=["UUID", "month"])
    monthly_summary.set_index("month", inplace=True)
    windows = [2, 3, 4, 5, 6]
    spend_cols = ["total_spend_c", "total_spend_p", "total_spend_g", "total_spend_cota"]
    quantity_cols = ["total_quantity_c", "total_quantity_p", "total_quantity_g", "total_quantity_cota"]
    invoice_col = "invoice_count"
    monthly_summary["total_spend"] = monthly_summary[spend_cols].sum(axis=1)
    
    for w in windows:
        rolled_spend_sum = (
            monthly_summary.groupby("UUID")[spend_cols]
            .rolling(window=w, min_periods=w)
            .sum()
            .rename(columns=lambda x: f"{x}_{w}m_sum")
        )
        rolled_quantity_sum = (
            monthly_summary.groupby("UUID")[quantity_cols]
            .rolling(window=w, min_periods=w)
            .sum()
            .rename(columns=lambda x: f"{x}_{w}m_sum")
        )
        rolled_total_spend_mean = (
            monthly_summary["total_spend"].groupby(monthly_summary["UUID"])
            .rolling(window=w, min_periods=w)
            .mean()
            .rename(f"avg_spend_{w}m")
        )
        rolled_total_spend_std = (
            monthly_summary["total_spend"].groupby(monthly_summary["UUID"])
            .rolling(window=w, min_periods=w)
            .std()
            .rename(f"std_spend_{w}m")
        )
        rolled_invoice_mean = (
            monthly_summary.groupby("UUID")[invoice_col]
            .rolling(window=w, min_periods=w)
            .mean()
            .rename(f"avg_invoices_{w}m")
        )
        rolled_invoice_std = (
            monthly_summary.groupby("UUID")[invoice_col]
            .rolling(window=w, min_periods=w)
            .std()
            .rename(f"std_invoices_{w}m")
        )
        rolled_features = pd.concat([
            rolled_spend_sum, rolled_quantity_sum, rolled_total_spend_mean,
            rolled_total_spend_std, rolled_invoice_mean, rolled_invoice_std
        ], axis=1).reset_index()
        monthly_summary = monthly_summary.reset_index().merge(
            rolled_features, on=["UUID", "month"], how="left"
        ).set_index("month")
    
    monthly_summary.reset_index(inplace=True)
    
    for w in windows:
        monthly_summary[f"total_spend_{w}m"] = (
            monthly_summary[[f"total_spend_c_{w}m_sum", f"total_spend_p_{w}m_sum", f"total_spend_g_{w}m_sum"]].sum(axis=1)
        )
        monthly_summary[f"total_quantity_{w}m"] = (
            monthly_summary[[f"total_quantity_c_{w}m_sum", f"total_quantity_p_{w}m_sum", f"total_quantity_g_{w}m_sum"]].sum(axis=1)
        )
    
    # Month-over-month percent change
    for k in range(2, 6):
        monthly_summary[f"spend_mom_pct_change_{k}"] = (
            monthly_summary.groupby("UUID")["total_spend"].transform(lambda x: x.pct_change(periods=k))
        )
    
    # Rolling slope features
    for w in windows:
        monthly_summary[f"spend_{w}m_slope"] = (
            monthly_summary.groupby("UUID")["total_spend"]
            .rolling(window=w, min_periods=w)
            .apply(compute_slope, raw=False)
            .reset_index(level=0, drop=True)
        )
    
    monthly_summary2 = monthly_summary.dropna(subset=["total_spend_c_6m_sum"])
    monthly_summary3 = monthly_summary2.copy()
    monthly_summary3["churn"] = (
        (monthly_summary3["gap_days"] > 29) |
        (monthly_summary3["next_month_first_date"].isna() & (monthly_summary3["month"] != monthly_summary3["month"].max()))
    ).astype(int)
    
    def drop_after_first_churn(df):
        def filter_client(group):
            if (group["churn"] == 1).any():
                first_churn_idx = group.index[group["churn"] == 1][0]
                return group.loc[:first_churn_idx]
            else:
                return group
        return df.groupby("UUID", group_keys=False).apply(filter_client).reset_index(drop=True)
    
    data = drop_after_first_churn(monthly_summary3)
    data['total_spend'] = data['total_spend_c'] + data['total_spend_p'] + data['total_spend_g']
    data['total_quantity'] = data['total_quantity_c'] + data['total_quantity_p'] + data['total_quantity_g']
    
    # Convert 'month' from Period to Timestamp
    if is_period_dtype(data['month']):
        data['month'] = data['month'].apply(lambda x: x.to_timestamp())
    
    return data

# Model evaluation function for Test Mode
def evaluate_model(X_train, y_train, X_test, y_test, uuids_test, selected_month, sampling_method, sampling_ratio, selected_features):
    # Validate selected features
    missing_features = [f for f in selected_features if f not in X_train.columns or f not in X_test.columns]
    if missing_features:
        raise ValueError(f"Selected features not found in data: {missing_features}")
    
    # Filter selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    model = lgb.LGBMClassifier(
        random_state=42, num_leaves=50, max_depth=6, min_child_samples=30,
        learning_rate=0.01, n_estimators=500, bagging_fraction=0.8,
        feature_fraction=0.8, bagging_freq=5
    )
    
    # Apply sampling based on user selection
    if sampling_method == "Undersampling":
        sampler = RandomUnderSampler(random_state=42, sampling_strategy=sampling_ratio)
    else:  # SMOTE
        sampler = SMOTE(random_state=42, sampling_strategy=sampling_ratio)
    
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
    
    # Train model
    model.fit(X_train_res, y_train_res)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Results DataFrame
    results_df = pd.DataFrame({
        'UUID': uuids_test['UUID'].values,
        'Month': selected_month,
        'Churn_Probability': y_pred_proba,
        'Predicted_Churn': y_pred
    })
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return model, results_df, accuracy, roc_auc, cm, report, feature_importance

# Prediction function for Prediction Mode
def predict_next_month(X_train, y_train, X_test, uuids_test, next_month, sampling_method, sampling_ratio, selected_features):
    # Validate selected features
    missing_features = [f for f in selected_features if f not in X_train.columns or f not in X_test.columns]
    if missing_features:
        raise ValueError(f"Selected features not found in data: {missing_features}")
    
    # Filter selected features
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    model = lgb.LGBMClassifier(
        random_state=42, num_leaves=50, max_depth=6, min_child_samples=30,
        learning_rate=0.01, n_estimators=500, bagging_fraction=0.8,
        feature_fraction=0.8, bagging_freq=5
    )
    
    # Apply sampling based on user selection
    if sampling_method == "Undersampling":
        sampler = RandomUnderSampler(random_state=42, sampling_strategy=sampling_ratio)
    else:  # SMOTE
        sampler = SMOTE(random_state=42, sampling_strategy=sampling_ratio)
    
    X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
    
    # Train model
    model.fit(X_train_res, y_train_res)
    
    # Predict probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    
    # Results DataFrame
    results_df = pd.DataFrame({
        'UUID': uuids_test['UUID'].values,
        'Month': next_month,
        'Churn_Probability': y_pred_proba,
        'Predicted_Churn': y_pred
    })
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    return results_df, feature_importance

# Main app logic
mode = st.radio("Select Mode", ["Test Mode", "Prediction Mode"])

# File uploader and preprocessing
st.subheader("Upload Data")
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"], key=f"{mode.lower().replace(' ', '_')}_uploader")

if uploaded_file is not None:
    # Read CSV or Excel
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file, engine='openpyxl')
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
    else:
        if 'DATE TRANS' not in df.columns:
            st.error("File must contain 'DATE TRANS' column.")
        else:
            df["DATE TRANS"] = pd.to_datetime(df["DATE TRANS"], errors='coerce')
            df["month"] = df["DATE TRANS"].dt.to_period("M")
            available_months = df["month"].dropna().dt.strftime('%Y-%m').unique()
            st.session_state.available_months = sorted(available_months)
            
            if len(available_months) == 0:
                st.error("No valid months found in 'DATE TRANS' column.")
            else:
                # Preprocess data only if new file or explicitly requested
                if (st.session_state.preprocessed_data is None or 
                    st.button("Preprocess Data")):
                    with st.spinner("Preprocessing data..."):
                        data = process_data(df)
                        st.session_state.preprocessed_data = data
                        st.session_state.results = None  # Reset results
                        st.session_state.selected_month = None
                        st.session_state.selected_latest_month = None
                        st.session_state.selected_features = None
                        st.success("Data preprocessed successfully!")
                
                data = st.session_state.preprocessed_data
                exclude_cols = [
                    'month', 'churn', 'last_invoice_date', 'next_month_first_date',
                    'first_invoice_date', 'UUID', 'gap_days'
                ]
                available_features = [col for col in data.columns if col not in exclude_cols]
                st.session_state.available_features = available_features

                # Sampling options (after preprocessing)
                st.subheader("Sampling Options")
                sampling_method = st.selectbox("Select Sampling Method", ["Undersampling", "SMOTE"], key="sampling_method")
                if sampling_method == "Undersampling":
                    sampling_ratio = st.number_input(
                        "Enter Undersampling Ratio (0.1 to 1.0, e.g., 0.4 for 40% minority class)",
                        min_value=0.1, max_value=1.0, value=0.4, step=0.05, key="sampling_ratio"
                    )
                else:
                    sampling_ratio = st.number_input(
                        "Enter SMOTE Oversampling Ratio (0.1 to 2.0, e.g., 0.5 for 50% minority class)",
                        min_value=0.1, max_value=2.0, value=0.5, step=0.05, key="sampling_ratio"
                    )

                # Feature selection
                st.subheader("Select Features for Training")
                st.info("Select or deselect features below, then click 'Train Model' to apply changes.")
                selected_features = st.multiselect(
                    "Choose features to use for training (select at least one)",
                    options=available_features,
                    default=available_features if st.session_state.selected_features is None else st.session_state.selected_features,
                    key=f"{mode.lower().replace(' ', '_')}_features"
                )

                if not selected_features:
                    st.error("Please select at least one feature for training.")
                else:
                    st.session_state.selected_features = selected_features

                    if mode == "Test Mode":
                        st.subheader("Test Mode: Evaluate Model for a Specific Month")
                        selected_month = st.selectbox(
                            "Select month to predict (YYYY-MM)",
                            options=st.session_state.available_months,
                            index=len(st.session_state.available_months)-1 if st.session_state.available_months else 0,
                            key="test_month"
                        )

                        if selected_month:
                            if not is_valid_month(selected_month):
                                st.error("Invalid month format. Use YYYY-MM (e.g., 2024-03).")
                            elif selected_month not in st.session_state.available_months:
                                st.error(f"Month {selected_month} not found in data.")
                            else:
                                st.session_state.selected_month = selected_month
                                
                                # Split data
                                selected_month_dt = pd.to_datetime(selected_month)
                                train = data[data['month'] < selected_month_dt]
                                test = data[data['month'] == selected_month_dt]
                                
                                if train.empty or test.empty:
                                    st.error("Insufficient data for training or testing.")
                                else:
                                    # Train and evaluate only if explicitly triggered
                                    if st.button("Train Model", key="test_train"):
                                        with st.spinner("Training and evaluating model..."):
                                            X_train = train.drop(columns=exclude_cols)
                                            y_train = train['churn']
                                            X_test = test.drop(columns=exclude_cols)
                                            y_test = test['churn']
                                            uuids_test = test[['UUID', 'month']]
                                            
                                            try:
                                                model, results_df, accuracy, roc_auc, cm, report, feature_importance = evaluate_model(
                                                    X_train, y_train, X_test, y_test, uuids_test, selected_month,
                                                    sampling_method, sampling_ratio, selected_features
                                                )
                                                
                                                # Store predictions in history
                                                st.session_state.prediction_history[selected_month] = results_df
                                                
                                                # Store results in session state
                                                st.session_state.results = {
                                                    'results_df': results_df,
                                                    'accuracy': accuracy,
                                                    'roc_auc': roc_auc,
                                                    'cm': cm,
                                                    'report': report,
                                                    'feature_importance': feature_importance,
                                                    'month': selected_month
                                                }
                                                st.session_state.total_features = len(feature_importance)
                                                st.success("Model trained successfully!")
                                            except ValueError as e:
                                                st.error(str(e))
                                
                                # Display results
                                if st.session_state.results is not None and st.session_state.results.get('month') == selected_month:
                                    results = st.session_state.results
                                    results_df = results['results_df']
                                    accuracy = results['accuracy']
                                    roc_auc = results['roc_auc']
                                    cm = results['cm']
                                    report = results['report']
                                    feature_importance = results['feature_importance']
                                    
                                    # Filter options
                                    st.subheader(f"Churn Predictions for {selected_month}")
                                    st.write("Filter predictions:")
                                    filter_option = st.radio(
                                        "Select filter",
                                        ["Show all clients", "Show churned clients", "Show not churned clients", "Show clients above probability threshold"],
                                        key="test_filter_option"
                                    )
                                    filtered_results_df = results_df
                                    
                                    if filter_option == "Show churned clients":
                                        filtered_results_df = results_df[results_df['Predicted_Churn'] == 1]
                                        st.write(f"Clients predicted to churn in {selected_month}:")
                                    elif filter_option == "Show not churned clients":
                                        filtered_results_df = results_df[results_df['Predicted_Churn'] == 0]
                                        st.write(f"Clients predicted not to churn in {selected_month}:")
                                    elif filter_option == "Show clients above probability threshold":
                                        prob_threshold = st.slider(
                                            "Select minimum churn probability threshold",
                                            min_value=0.0, max_value=1.0, value=0.9, step=0.01,
                                            key="test_prob_threshold"
                                        )
                                        filtered_results_df = results_df[results_df['Churn_Probability'] >= prob_threshold]
                                        st.write(f"Clients with churn probability >= {prob_threshold} in {selected_month}:")
                                    else:
                                        st.write("All clients sorted by churn probability:")
                                    
                                    if filtered_results_df.empty:
                                        st.warning("No clients match the selected filter criteria.")
                                    else:
                                        st.dataframe(
                                            filtered_results_df.sort_values(by='Churn_Probability', ascending=False)
                                            .style.format({'Churn_Probability': '{:.2f}'})
                                        )
                                    
                                    # Download filtered results as CSV and Excel
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        csv = filtered_results_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Filtered Predictions as CSV",
                                            data=csv,
                                            file_name=f"churn_predictions_{selected_month}_filtered.csv",
                                            mime="text/csv"
                                        )
                                    with col2:
                                        excel_buffer = io.BytesIO()
                                        filtered_results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                                        excel_buffer.seek(0)
                                        st.download_button(
                                            label="Download Filtered Predictions as Excel",
                                            data=excel_buffer,
                                            file_name=f"churn_predictions_{selected_month}_filtered.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    
                                    # Model performance
                                    st.subheader("Model Performance")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write(f"**Accuracy**: {accuracy:.4f}")
                                        st.write(f"**ROC AUC**: {roc_auc:.4f}")
                                        st.write("**Classification Report**:")
                                        report_df = pd.DataFrame(report).transpose()
                                        st.dataframe(report_df)
                                    
                                    with col2:
                                        st.write("**Confusion Matrix**:")
                                        fig, ax = plt.subplots(figsize=(5, 4))
                                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
                                        ax.set_xlabel('Predicted')
                                        ax.set_ylabel('Actual')
                                        st.pyplot(fig)
                                    
                                    # Feature importance display
                                    st.subheader("Feature Importance")
                                    st.info("Adjusting the number of top features only updates the display below, without re-running the process.")
                                    top_n = st.number_input(
                                        f"Enter number of top features to display (1-{st.session_state.total_features})",
                                        min_value=1, max_value=st.session_state.total_features, value=10, step=1, key="test_top_n"
                                    )
                                    st.write(f"Top {top_n} most important features:")
                                    st.dataframe(feature_importance.head(top_n))
                                    
                                    # Plot feature importance
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.barplot(data=feature_importance.head(top_n), x='Importance', y='Feature')
                                    plt.title(f'Top {top_n} Feature Importance')
                                    st.pyplot(fig)
                                else:
                                    st.info("Click 'Train Model' to generate results with the selected features.")
                    
                    elif mode == "Prediction Mode":
                        st.subheader("Prediction Mode: Predict Churn for Next Month")
                        latest_month = max(st.session_state.available_months)
                        next_month = (pd.to_datetime(latest_month) + pd.offsets.MonthEnd(1)).strftime('%Y-%m')
                        
                        st.markdown(f"Latest month in dataset: **{latest_month}**. Predictions will be made for: **{next_month}**.")
                        
                        selected_month = st.selectbox(
                            f"Select the latest month in your data (YYYY-MM, default: {latest_month})",
                            options=st.session_state.available_months,
                            index=len(st.session_state.available_months)-1 if st.session_state.available_months else 0,
                            key="pred_month"
                        )
                        
                        if selected_month:
                            if not is_valid_month(selected_month):
                                st.error("Invalid month format. Use YYYY-MM (e.g., 2024-03).")
                            elif selected_month not in st.session_state.available_months:
                                st.error(f"Month {selected_month} not found in data.")
                            else:
                                st.session_state.selected_latest_month = selected_month
                                
                                # Split data
                                selected_month_dt = pd.to_datetime(selected_month)
                                train = data[data['month'] <= selected_month_dt]
                                test = data[data['month'] == selected_month_dt]
                                
                                if train.empty or test.empty:
                                    st.error("Insufficient data for training or prediction.")
                                else:
                                    # Train and predict only if triggered
                                    if st.button("Train Model", key="pred_train"):
                                        with st.spinner("Training and predicting..."):
                                            X_train = train.drop(columns=exclude_cols)
                                            y_train = train['churn']
                                            X_test = test.drop(columns=exclude_cols)
                                            uuids_test = test[['UUID', 'month']]
                                            
                                            try:
                                                results_df, feature_importance = predict_next_month(
                                                    X_train, y_train, X_test, uuids_test, next_month,
                                                    sampling_method, sampling_ratio, selected_features
                                                )
                                                
                                                # Store predictions in history
                                                st.session_state.prediction_history[next_month] = results_df
                                                
                                                # Store results
                                                st.session_state.results = {
                                                    'results_df': results_df,
                                                    'feature_importance': feature_importance,
                                                    'month': next_month
                                                }
                                                st.session_state.total_features = len(feature_importance)
                                                st.success("Predictions generated successfully!")
                                            except ValueError as e:
                                                st.error(str(e))
                                
                                # Display results
                                if st.session_state.results is not None and st.session_state.results.get('month') == next_month:
                                    results = st.session_state.results
                                    results_df = results['results_df']
                                    feature_importance = results['feature_importance']
                                    
                                    # Filter options
                                    st.subheader(f"Churn Predictions for {next_month}")
                                    st.write("Filter predictions:")
                                    filter_option = st.radio(
                                        "Select filter",
                                        ["Show all clients", "Show churned clients", "Show not churned clients", "Show clients above probability threshold"],
                                        key="pred_filter_option"
                                    )
                                    filtered_results_df = results_df
                                    
                                    if filter_option == "Show churned clients":
                                        filtered_results_df = results_df[results_df['Predicted_Churn'] == 1]
                                        st.write(f"Clients predicted to churn in {next_month}:")
                                    elif filter_option == "Show not churned clients":
                                        filtered_results_df = results_df[results_df['Predicted_Churn'] == 0]
                                        st.write(f"Clients predicted not to churn in {next_month}:")
                                    elif filter_option == "Show clients above probability threshold":
                                        prob_threshold = st.slider(
                                            "Select minimum churn probability threshold",
                                            min_value=0.0, max_value=1.0, value=0.9, step=0.01,
                                            key="pred_prob_threshold"
                                        )
                                        filtered_results_df = results_df[results_df['Churn_Probability'] >= prob_threshold]
                                        st.write(f"Clients with churn probability >= {prob_threshold} in {next_month}:")
                                    else:
                                        st.write("All clients sorted by churn probability:")
                                    
                                    if filtered_results_df.empty:
                                        st.warning("No clients match the selected filter criteria.")
                                    else:
                                        st.dataframe(
                                            filtered_results_df.sort_values(by='Churn_Probability', ascending=False)
                                            .style.format({'Churn_Probability': '{:.2f}'})
                                        )
                                    
                                    # Download filtered results as CSV and Excel
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        csv = filtered_results_df.to_csv(index=False)
                                        st.download_button(
                                            label="Download Filtered Predictions as CSV",
                                            data=csv,
                                            file_name=f"churn_predictions_{next_month}_filtered.csv",
                                            mime="text/csv"
                                        )
                                    with col2:
                                        excel_buffer = io.BytesIO()
                                        filtered_results_df.to_excel(excel_buffer, index=False, engine='openpyxl')
                                        excel_buffer.seek(0)
                                        st.download_button(
                                            label="Download Filtered Predictions as Excel",
                                            data=excel_buffer,
                                            file_name=f"churn_predictions_{next_month}_filtered.xlsx",
                                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                        )
                                    
                                    # Feature importance display
                                    st.subheader("Feature Importance")
                                    st.info("Adjusting the number of top features only updates the display below, without re-running the process.")
                                    top_n = st.number_input(
                                        f"Enter number of top features to display (1-{st.session_state.total_features})",
                                        min_value=1, max_value=st.session_state.total_features, value=10, step=1, key="pred_top_n"
                                    )
                                    st.write(f"Top {top_n} most important features:")
                                    st.dataframe(feature_importance.head(top_n))
                                    
                                    # Plot feature importance
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    sns.barplot(data=feature_importance.head(top_n), x='Importance', y='Feature')
                                    plt.title(f'Top {top_n} Feature Importance')
                                    st.pyplot(fig)
                                else:
                                    st.info("Click 'Train Model' to generate predictions with the selected features.")
else:
    st.info("Please upload a CSV or Excel file to start the analysis.")