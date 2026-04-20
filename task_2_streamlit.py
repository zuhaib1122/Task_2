import streamlit as st
import datetime
import requests
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier

# --- Page Configuration ---
st.set_page_config(page_title="Athlete Performance Dashboard", layout="wide")

# --- App Title & Sidebar ---
st.title("🏀 Youth Basketball Performance Analysis")
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to:", ["Overview", "Data Cleaning & EDA", "Hypothesis Testing", "Machine Learning", "Live Weather"])

# --- 1. Reusable Cleaner Function (Cached) ---
@st.cache_data
def dataset_cleaner(df, numerical_strategy='median', categorical_strategy='mode', outlier_method='iqr'):
    clean_df = df.copy()
    log = []
    
    # Handle Missing Values
    for col in clean_df.columns:
        null_count = clean_df[col].isnull().sum()
        if null_count > 0:
            if clean_df[col].dtype in ['int64', 'float64']:
                fill_val = clean_df[col].median() if numerical_strategy == 'median' else clean_df[col].mean()
                clean_df[col] = clean_df[col].fillna(fill_val)
            else:
                if not clean_df[col].mode().empty:
                    mode_val = clean_df[col].mode()[0]
                    clean_df[col] = clean_df[col].fillna(mode_val)
    
    # Outliers
    if outlier_method == 'iqr':
        numerical_cols = clean_df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_cols:
            Q1 = clean_df[col].quantile(0.25)
            Q3 = clean_df[col].quantile(0.75)
            IQR = Q3 - Q1
            clean_df[col] = np.where(clean_df[col] < (Q1 - 1.5 * IQR), Q1 - 1.5 * IQR, clean_df[col])
            clean_df[col] = np.where(clean_df[col] > (Q3 + 1.5 * IQR), Q3 + 1.5 * IQR, clean_df[col])
            
    return clean_df

# --- Load Data ---
@st.cache_data
def load_local_data():
    # Replace with your actual path or a file uploader
    # df = pd.read_csv("C:/Users/Hafiz Zuhaib Idrees/Desktop/youth_basketball_training_dataset.csv")
    # For demo, creating a dummy check or using file uploader is better
    try:
        df = pd.read_csv("youth_basketball_training_dataset.csv")
    except:
        st.error("Please ensure 'youth_basketball_training_dataset.csv' is in the same folder.")
        return pd.DataFrame()
    return df

mydf = load_local_data()

if not mydf.empty:
    if page == "Overview":
        st.header("📋 Problem Statement")
        st.info("The objective of this analysis was to determine if Gender is a significant predictor of athlete strength and Game_Performance_Index. We wanted to see if training programs should be specialized by gender or if a unified approach is more effective.")
        st.subheader("Dataset Preview")
        st.dataframe(mydf.head(10))

    elif page == "Data Cleaning & EDA":
        st.header(" Data Cleaning & Exploratory Analysis")
        clean_df = dataset_cleaner(mydf)
        st.success("Data cleaned and Outliers treated (Winsorization).")
        
        # Correlations
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Pearson Correlation")
            pearson_corr = mydf[['Age', 'Game_Performance_Index', 'Height_cm', 'Weight_kg', 'Jump_Height_cm']].corr()
            fig1, ax1 = plt.subplots()
            sns.heatmap(pearson_corr, annot=True, cmap='coolwarm', ax=ax1)
            st.pyplot(fig1)

        with col2:
            st.write("### Spearman Correlation")
            spearman_corr = mydf[['Age', 'Strength', 'Endurance', 'Recovery_Time_hours']].corr(method='spearman')
            fig2, ax2 = plt.subplots()
            sns.heatmap(spearman_corr, annot=True, cmap='viridis', ax=ax2)
            st.pyplot(fig2)

        st.write("### Distribution & Skewness")
        skew_cols = ['Age', 'Height_cm', 'Weight_kg', 'Game_Performance_Index']
        fig3, axes = plt.subplots(1, 4, figsize=(20, 5))
        for i, col in enumerate(skew_cols):
            sns.kdeplot(mydf[col], fill=True, ax=axes[i])
            axes[i].set_title(f"{col}\nSkew: {mydf[col].skew():.2f}")
        st.pyplot(fig3)

        # Performance vs BMI
        st.write("### Performance vs BMI Analysis")
        mydf['BMI'] = mydf['Weight_kg']/(mydf['Height_cm']/100)**2
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.scatterplot(data=mydf, x='BMI', y='Game_Performance_Index', hue='Strength', palette='magma', ax=ax4)
        plt.axvline(28, color='red', linestyle='--', label='High BMI Threshold')
        st.pyplot(fig4)

    elif page == "Hypothesis Testing":
        st.header("🧪 Statistical Hypothesis Testing")
        
        # Test 1
        man_p = mydf[mydf['Gender']=='Male']['Game_Performance_Index']
        woman_p = mydf[mydf['Gender'] == 'Female']['Game_Performance_Index']
        t_stat, p_val = stats.ttest_ind(man_p, woman_p, equal_var=False)
        
        st.subheader("Performance Difference by Gender")
        st.write(f"**T-Statistic:** {t_stat:.4f} | **P-Value:** {p_val:.4f}")
        if p_val < 0.05:
            st.error("Reject Null Hypothesis: Significant difference exists.")
        else:
            st.success("Fail to Reject Null Hypothesis: No significant difference in performance between Male and Female.")

    elif page == "Machine Learning":
        st.header("🤖 Model Performance (Decision Tree)")
        
        features = ['Weekly_Training_Hours', 'Development_Score', 'Strength', 'Agility_sec', 'Recovery_Time_hours', 'Decision_Making_Ability', 'Dribbling_Speed_sec', 'Focus_Level']
        X = mydf[features]
        y = mydf['Game_Performance_Index']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = DecisionTreeRegressor(max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("R-squared Score", f"{r2:.4f}")
        m_col2.metric("Mean Absolute Error", f"{mae:.2f}")
        
        st.subheader("Actual vs Predicted")
        comparison = pd.DataFrame({'Actual': y_test.values.flatten(), 'Predicted': preds.flatten()})
        st.dataframe(comparison.head(10))

        st.subheader(" Recommendations")
        st.success("**Unified Training:** Since gender isn't a top predictor, use a unified curriculum to save resources.")
        st.warning("**Talent-Based Recruitment:** Focus on 'Decision Making' and 'Strength' scores rather than categorical labels.")

    elif page == "Live Weather":
        st.header(" Live Weather Fetch (Lahore)")
        url = "https://api.open-meteo.com/v1/forecast?latitude=31.5204&longitude=74.3587&hourly=temperature_2m,relative_humidity_2m&past_days=2"
        res = requests.get(url).json()
        weather_df = pd.DataFrame(res['hourly'])
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        
        st.write("### Recent Hourly Data")
        st.dataframe(weather_df.head())
        
        avg_temp = weather_df['temperature_2m'].mean()
        st.metric("Avg Temperature (Past 2 Days)", f"{avg_temp:.2f} °C")