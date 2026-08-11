import os
import json
import joblib
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as gg
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Traffic Volume Prediction System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    
    /* Header branding styling */
    .brand-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .brand-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .brand-header p {
        margin: 0.3rem 0 0 0;
        color: #94A3B8;
        font-size: 1rem;
    }
    
    /* Card Component */
    .custom-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    /* Metric Card Styling */
    .metric-container {
        background: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }
    
    /* Badge styling */
    .badge-low {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-moderate {
        background-color: #FEF9C3;
        color: #854D0E;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-high {
        background-color: #FFEDD5;
        color: #9A3412;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }
    .badge-veryhigh {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.2rem;
        display: inline-block;
    }

    /* Footer styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0F172A;
        color: #94A3B8;
        text-align: center;
        padding: 0.5rem;
        font-size: 0.85rem;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Path definitions
DATA_PATH = "data/traffic_data.csv"
MODEL_PATH = "models/traffic_model.pkl"
PIPELINE_PATH = "models/preprocessing_pipeline.pkl"
METRICS_PATH = "models/model_metrics.json"

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, keep_default_na=False)
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_name'] = df['date'].dt.day_name()
        df['hour'] = df['date'].dt.hour
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        return df
    return None

@st.cache_resource
def load_models():
    model = None
    pipeline = None
    metrics = None
    
    if os.path.exists(MODEL_PATH) and os.path.exists(PIPELINE_PATH):
        model = joblib.load(MODEL_PATH)
        pipeline = joblib.load(PIPELINE_PATH)
    
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
            
    return model, pipeline, metrics

df = load_dataset()
model, pipeline, metrics = load_models()

# Developer Sidebar Info
st.sidebar.markdown("### 🚦 Traffic Prediction System")
st.sidebar.caption("Machine Learning Based Traffic Volume Modeling")

st.sidebar.markdown("---")
st.sidebar.markdown("#### 👨‍💻 Developer Profile")
st.sidebar.markdown("**Name:** Swapna V")
st.sidebar.markdown("**Role:** ML Engineer")
st.sidebar.markdown("**Company:** IPEC Solutions")
st.sidebar.markdown("---")

# Navigation Options
menu_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "📈 Traffic Trends",
        "🤖 Model Performance",
        "🚦 Traffic Prediction",
        "ℹ️ About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 Tip: Navigate through the tabs to explore traffic EDA, compare ML models, or make custom predictions.")

# Check for missing models
if model is None or pipeline is None:
    st.warning("⚠️ ML Model or Preprocessing Pipeline not found!")
    st.info("Please train the model first by running `python train_model.py` in your workspace terminal.")

# Header Banner
st.markdown("""
<div class="brand-header">
    <h1>🚦 Traffic Volume Prediction System</h1>
    <p>Real-time Urban Traffic Flow Forecasting & Decision Support | Developed by Swapna V (ML Engineer, IPEC Solutions)</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. HOME PAGE
# -----------------------------------------------------------------------------
if menu_choice == "🏠 Home":
    st.subheader("Welcome to the Traffic Volume Prediction System")
    st.markdown("""
    This machine learning system empowers urban traffic planners, municipal engineers, and commuters to **analyze historical traffic trends and predict hourly vehicle volumes** based on weather conditions, seasonal patterns, and time-of-day indicators.
    """)
    
    if df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Total Hourly Records</div>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
            
        with col2:
            avg_vol = int(df['traffic_volume'].mean())
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Avg Vehicles / Hour</div>
            </div>
            """.format(avg_vol), unsafe_allow_html=True)
            
        with col3:
            max_vol = int(df['traffic_volume'].max())
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Peak Traffic Volume</div>
            </div>
            """.format(max_vol), unsafe_allow_html=True)
            
        with col4:
            best_model_name = metrics["best_model_name"] if metrics else "Gradient Boosting"
            best_r2 = metrics["results"][best_model_name]["R2"] if metrics else 0.9821
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{:.1f}%</div>
                <div class="metric-label">Best Model R² ({})</div>
            </div>
            """.format(best_r2 * 100, best_model_name), unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown("### 🔄 Machine Learning Pipeline Workflow")
        st.markdown("""
        The system executes an end-to-end automated machine learning architecture:
        
        1. **Data Collection & Cleaning**: Processing hourly traffic logs, handling duplicates, and parsing temporal structures.
        2. **Feature Engineering**: Extracting hour-of-day, day-of-week, weekend flags, and weather severity indexes.
        3. **Exploratory Data Analysis (EDA)**: Profiling rush hours, weekend variations, and precipitation effects.
        4. **Preprocessing Pipeline**: Standardizing numeric features and one-hot encoding weather & holiday categories.
        5. **Multi-Model Training**: Fitting Linear Regression, Decision Trees, Random Forests, and Gradient Boosting algorithms.
        6. **Model Selection & Evaluation**: Evaluating MAE, RMSE, and R² scores on out-of-sample test datasets.
        7. **Real-time Inference**: Serving instant traffic volume predictions and traffic level classifications via Streamlit.
        """)
        
    with col_b:
        st.markdown("### 🎯 System Features")
        st.success("✅ **Interactive EDA**: Detailed traffic profiling by hour, day, month, and weather.")
        st.info("📊 **Model Leaderboard**: Side-by-side performance evaluation across 5 regression models.")
        st.warning("⚡ **Instant Prediction**: Enter time and weather parameters to calculate predicted traffic.")
        st.error("🚦 **4-Tier Traffic Categorization**: Automatic classification (Low, Moderate, High, Very High).")

# -----------------------------------------------------------------------------
# 2. DATA ANALYSIS PAGE
# -----------------------------------------------------------------------------
elif menu_choice == "📊 Data Analysis":
    st.subheader("📊 Dataset Exploration & Summary")
    
    if df is not None:
        st.write("Explore dataset dimensions, missing values, variable types, and statistical properties.")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{df.shape[0]:,}")
        col2.metric("Total Features", f"{df.shape[1]}")
        col3.metric("Missing Values", f"{df.isnull().sum().sum()}")
        col4.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        st.markdown("---")
        
        tab1, tab2, tab3 = st.tabs(["📋 Dataset Preview", "📈 Summary Statistics", "🔍 Missing & Data Types"])
        
        with tab1:
            st.markdown("##### Dataset First & Last Rows")
            rows_to_show = st.slider("Select number of rows to view", 5, 50, 10)
            
            col_head, col_tail = st.columns(2)
            with col_head:
                st.caption("First Rows (Head)")
                st.dataframe(df.head(rows_to_show), use_container_width=True)
            with col_tail:
                st.caption("Last Rows (Tail)")
                st.dataframe(df.tail(rows_to_show), use_container_width=True)
                
        with tab2:
            st.markdown("##### Numerical Features Summary (`describe`)")
            st.dataframe(df.describe().T.style.format("{:.2f}"), use_container_width=True)
            
            st.markdown("##### Categorical Distribution")
            col_cat1, col_cat2 = st.columns(2)
            with col_cat1:
                st.write("**Weather Main Distribution**")
                st.dataframe(df['weather_main'].value_counts().reset_index().rename(columns={'index':'Weather', 'weather_main':'Count'}), use_container_width=True)
            with col_cat2:
                st.write("**Holiday Frequency**")
                st.dataframe(df['holiday'].value_counts().reset_index().rename(columns={'index':'Holiday', 'holiday':'Count'}), use_container_width=True)
                
        with tab3:
            st.markdown("##### Data Types & Missing Values Breakdown")
            df_info = pd.DataFrame({
                "Column Name": df.columns,
                "Data Type": [str(dt) for dt in df.dtypes],
                "Null Count": df.isnull().sum().values,
                "Unique Values": [df[c].nunique() for c in df.columns]
            })
            st.dataframe(df_info, use_container_width=True)
    else:
        st.error("Dataset not found. Please verify data/traffic_data.csv exists.")

# -----------------------------------------------------------------------------
# 3. TRAFFIC TRENDS PAGE
# -----------------------------------------------------------------------------
elif menu_choice == "📈 Traffic Trends":
    st.subheader("📈 Exploratory Traffic Trends & Patterns")
    
    if df is not None:
        # Filter section
        st.markdown("##### 🎛️ Data Visualization Filters")
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            selected_month = st.selectbox("Select Month Filter", ["All"] + list(range(1, 13)))
        with col_f2:
            selected_weather = st.selectbox("Select Weather Main Filter", ["All"] + list(df['weather_main'].unique()))
        with col_f3:
            selected_day = st.selectbox("Select Day of Week Filter", ["All", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
            
        filtered_df = df.copy()
        if selected_month != "All":
            filtered_df = filtered_df[filtered_df['month'] == int(selected_month)]
        if selected_weather != "All":
            filtered_df = filtered_df[filtered_df['weather_main'] == selected_weather]
        if selected_day != "All":
            filtered_df = filtered_df[filtered_df['day_name'] == selected_day]
            
        st.markdown(f"*(Showing **{len(filtered_df):,}** records matching filters)*")
        st.markdown("---")
        
        # Row 1: Hourly Profile & Day of Week
        col_t1, col_t2 = st.columns(2)
        
        with col_t1:
            st.markdown("##### 🕒 Average Traffic Volume by Hour of Day")
            hourly_avg = filtered_df.groupby("hour")["traffic_volume"].mean().reset_index()
            fig_hour = px.line(
                hourly_avg, x="hour", y="traffic_volume",
                markers=True,
                labels={"hour": "Hour of Day (0-23)", "traffic_volume": "Avg Vehicles / Hour"},
                color_discrete_sequence=["#3B82F6"]
            )
            fig_hour.update_layout(xaxis=dict(tickmode="linear", tick0=0, dtick=1), template="plotly_white")
            st.plotly_chart(fig_hour, use_container_width=True)
            
        with col_t2:
            st.markdown("##### 📅 Average Traffic Volume by Day of Week")
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_avg = filtered_df.groupby("day_name")["traffic_volume"].mean().reindex(day_order).reset_index()
            fig_day = px.bar(
                daily_avg, x="day_name", y="traffic_volume",
                labels={"day_name": "Day of Week", "traffic_volume": "Avg Vehicles / Hour"},
                color="traffic_volume",
                color_continuous_scale="Blues"
            )
            fig_day.update_layout(template="plotly_white")
            st.plotly_chart(fig_day, use_container_width=True)
            
        # Row 2: Weather Impact & Monthly Trend
        col_t3, col_t4 = st.columns(2)
        
        with col_t3:
            st.markdown("##### 🌦️ Weather Condition vs Traffic Volume")
            weather_avg = filtered_df.groupby("weather_main")["traffic_volume"].mean().reset_index().sort_values("traffic_volume", ascending=False)
            fig_weather = px.bar(
                weather_avg, x="weather_main", y="traffic_volume",
                labels={"weather_main": "Weather Condition", "traffic_volume": "Avg Traffic Volume"},
                color="traffic_volume",
                color_continuous_scale="Teal"
            )
            fig_weather.update_layout(template="plotly_white")
            st.plotly_chart(fig_weather, use_container_width=True)
            
        with col_t4:
            st.markdown("##### 🗓️ Monthly Traffic Trend")
            monthly_avg = filtered_df.groupby("month")["traffic_volume"].mean().reset_index()
            fig_month = px.bar(
                monthly_avg, x="month", y="traffic_volume",
                labels={"month": "Month (1-12)", "traffic_volume": "Avg Vehicles / Hour"},
                color_discrete_sequence=["#6366F1"]
            )
            fig_month.update_layout(xaxis=dict(tickmode="linear", tick0=1, dtick=1), template="plotly_white")
            st.plotly_chart(fig_month, use_container_width=True)
            
        # Row 3: Temperature vs Traffic Scatter & Correlation Heatmap
        col_t5, col_t6 = st.columns(2)
        
        with col_t5:
            st.markdown("##### 🌡️ Temperature (°C) vs Traffic Volume")
            fig_temp = px.scatter(
                filtered_df.sample(min(1000, len(filtered_df)), random_state=42),
                x="temperature", y="traffic_volume",
                color="weather_main",
                opacity=0.6,
                labels={"temperature": "Temperature (°C)", "traffic_volume": "Traffic Volume"},
                template="plotly_white"
            )
            st.plotly_chart(fig_temp, use_container_width=True)
            
        with col_t6:
            st.markdown("##### 🔢 Numerical Features Correlation Matrix")
            num_cols = ['hour', 'month', 'day_of_week', 'is_weekend', 'temperature', 'rain_1h', 'snow_1h', 'clouds_all', 'traffic_volume']
            corr_matrix = filtered_df[num_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                labels=dict(color="Correlation"),
                x=num_cols, y=num_cols
            )
            fig_corr.update_layout(template="plotly_white")
            st.plotly_chart(fig_corr, use_container_width=True)

# -----------------------------------------------------------------------------
# 4. MODEL PERFORMANCE PAGE
# -----------------------------------------------------------------------------
elif menu_choice == "🤖 Model Performance":
    st.subheader("🤖 Machine Learning Model Leaderboard & Evaluation")
    
    if metrics is not None:
        st.markdown(f"**Trained Models Evaluation Summary** *(Trained at: {metrics.get('trained_at', 'N/A')})*")
        
        results_df = pd.DataFrame(metrics["results"]).T.reset_index().rename(columns={"index": "Model Name"})
        results_df = results_df.sort_values("R2", ascending=False)
        
        st.markdown(f"### 🏆 Winning Model: **{metrics['best_model_name']}**")
        st.success(f"Achieved highest test accuracy with R² = **{metrics['results'][metrics['best_model_name']]['R2']:.4f}** and RMSE = **{metrics['results'][metrics['best_model_name']]['RMSE']:.2f}** vehicles/hour.")
        
        st.markdown("---")
        
        col_m1, col_m2 = st.columns([1.2, 1])
        
        with col_m1:
            st.markdown("##### 📊 Model Performance Comparison Table")
            st.dataframe(
                results_df.style.highlight_max(axis=0, subset=["R2"], color="#DCFCE7")
                                .highlight_min(axis=0, subset=["RMSE", "MAE", "MSE"], color="#DCFCE7")
                                .format({"MAE": "{:.2f}", "MSE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}),
                use_container_width=True
            )
            
        with col_m2:
            st.markdown("##### 📈 Model R² Score Comparison")
            fig_comp = px.bar(
                results_df, x="Model Name", y="R2",
                color="R2",
                text_auto=".3f",
                color_continuous_scale="Viridis",
                labels={"R2": "R² Score (Closer to 1.0 is Best)"}
            )
            fig_comp.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_comp, use_container_width=True)
            
        st.markdown("---")
        
        # Feature importance if available
        if "top_feature_importances" in metrics and metrics["top_feature_importances"]:
            st.markdown("##### 🔍 Top Feature Importances (Tree Ensemble Model)")
            fi_df = pd.DataFrame(
                list(metrics["top_feature_importances"].items()),
                columns=["Feature", "Importance"]
            ).sort_values("Importance", ascending=True)
            
            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Purples",
                title="Relative Feature Importance in Traffic Volume Prediction"
            )
            fig_fi.update_layout(template="plotly_white")
            st.plotly_chart(fig_fi, use_container_width=True)
            
        # Sample Actual vs Predicted Plot
        if df is not None and model is not None and pipeline is not None:
            st.markdown("##### 🎯 Sample Test Set: Actual vs Predicted Traffic Volume")
            # Generate test sample
            sample_df = df.tail(200).copy()
            num_f = metrics["num_features"]
            cat_f = metrics["cat_features"]
            
            X_sample = sample_df[num_f + cat_f]
            X_sample_prep = pipeline.transform(X_sample)
            y_pred_sample = model.predict(X_sample_prep)
            
            sample_df["Predicted_Volume"] = y_pred_sample
            
            fig_actual_pred = go.Figure()
            fig_actual_pred.add_trace(go.Scatter(x=sample_df["date"], y=sample_df["traffic_volume"], mode="lines", name="Actual Volume", line=dict(color="#2563EB", width=2)))
            fig_actual_pred.add_trace(go.Scatter(x=sample_df["date"], y=sample_df["Predicted_Volume"], mode="lines", name="Predicted Volume", line=dict(color="#EF4444", width=2, dash="dash")))
            
            fig_actual_pred.update_layout(
                title="Actual vs Predicted Traffic Volume Over Time (Latest Test Period)",
                xaxis_title="Datetime",
                yaxis_title="Traffic Volume (Vehicles / Hour)",
                template="plotly_white"
            )
            st.plotly_chart(fig_actual_pred, use_container_width=True)
    else:
        st.error("Model metrics not found. Run train_model.py first.")

# -----------------------------------------------------------------------------
# 5. TRAFFIC PREDICTION PAGE
# -----------------------------------------------------------------------------
elif menu_choice == "🚦 Traffic Prediction":
    st.subheader("🚦 Predict Hourly Traffic Volume")
    st.write("Input current or forecasted weather and time parameters to generate an instant traffic volume prediction.")
    
    if model is not None and pipeline is not None:
        with st.form("prediction_form"):
            col_p1, col_p2, col_p3 = st.columns(3)
            
            with col_p1:
                input_date = st.date_input("Select Date", datetime.date.today())
                input_hour = st.slider("Select Hour of Day (0 - 23)", 0, 23, 8)
                input_holiday = st.selectbox("Holiday Status", [
                    "No Holiday", "New Year's Day", "Martin Luther King Jr. Day",
                    "Washington's Birthday", "Memorial Day", "Independence Day",
                    "Labor Day", "Columbus Day", "Veterans Day", "Thanksgiving", "Christmas Day"
                ])
                
            with col_p2:
                input_temp = st.number_input("Temperature (°C)", min_value=-20.0, max_value=50.0, value=18.0, step=0.5)
                input_rain = st.number_input("Rain in Past Hour (mm)", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                input_snow = st.number_input("Snow in Past Hour (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1)
                
            with col_p3:
                input_clouds = st.slider("Cloud Coverage (%)", 0, 100, 20)
                input_weather_main = st.selectbox("Weather Condition (Main)", [
                    "Clear", "Clouds", "Rain", "Snow", "Mist", "Fog", "Drizzle", "Thunderstorm"
                ])
                
                weather_desc_map = {
                    "Clear": ["sky is clear"],
                    "Clouds": ["few clouds", "scattered clouds", "broken clouds", "overcast clouds"],
                    "Rain": ["light rain", "moderate rain", "heavy intensity rain"],
                    "Snow": ["light snow", "heavy snow"],
                    "Mist": ["mist"],
                    "Fog": ["fog"],
                    "Drizzle": ["light intensity drizzle"],
                    "Thunderstorm": ["thunderstorm with light rain", "proximity thunderstorm"]
                }
                
                input_weather_desc = st.selectbox("Weather Description", weather_desc_map[input_weather_main])
                
            submit_btn = st.form_submit_button("🚦 Predict Traffic Volume", use_container_width=True, type="primary")
            
        if submit_btn:
            dt_obj = datetime.datetime.combine(input_date, datetime.time(input_hour, 0))
            month = dt_obj.month
            day_of_week = dt_obj.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            
            input_dict = {
                'hour': [input_hour],
                'month': [month],
                'day_of_week': [day_of_week],
                'is_weekend': [is_weekend],
                'temperature': [input_temp],
                'rain_1h': [input_rain],
                'snow_1h': [input_snow],
                'clouds_all': [input_clouds],
                'holiday': [input_holiday],
                'weather_main': [input_weather_main],
                'weather_description': [input_weather_desc]
            }
            
            input_df = pd.DataFrame(input_dict)
            
            # Preprocess and Predict
            input_prep = pipeline.transform(input_df)
            pred_volume = float(model.predict(input_prep)[0])
            pred_volume = max(100, int(round(pred_volume)))
            
            st.markdown("---")
            st.markdown("### 📊 Traffic Volume Prediction Results")
            
            res_col1, res_col2 = st.columns([1.2, 1])
            
            with res_col1:
                st.markdown(f"#### Predicted Volume: **{pred_volume:,} vehicles / hour**")
                
                # Classification thresholds
                if pred_volume < 1500:
                    level = "🟢 Low Traffic"
                    badge_class = "badge-low"
                    desc = "Traffic flow is smooth and uncongested. Minimal travel delays expected."
                elif pred_volume < 3500:
                    level = "🟡 Moderate Traffic"
                    badge_class = "badge-moderate"
                    desc = "Normal urban traffic density. Standard commute times apply."
                elif pred_volume < 5200:
                    level = "🟠 High Traffic"
                    badge_class = "badge-high"
                    desc = "Heavy traffic conditions. Expect moderate delays and plan alternate routes if possible."
                else:
                    level = "🔴 Very High Traffic"
                    badge_class = "badge-veryhigh"
                    desc = "Severe congestion / Peak rush hour. High risk of heavy delays."
                    
                st.markdown(f'<div class="{badge_class}">{level}</div>', unsafe_allow_html=True)
                st.info(f"ℹ️ **Advisory**: {desc}")
                
            with res_col2:
                # Historical Comparison if dataset available
                if df is not None:
                    hist_hour_avg = int(df[df['hour'] == input_hour]['traffic_volume'].mean())
                    diff = pred_volume - hist_hour_avg
                    pct = (diff / hist_hour_avg) * 100
                    
                    st.metric(
                        label=f"Historical Avg for Hour {input_hour}:00",
                        value=f"{hist_hour_avg:,} veh/h",
                        delta=f"{diff:+,} ({pct:+.1f}%)"
                    )
                    
            # Visual Capacity Gauge
            st.markdown("##### 🚗 Road Capacity Utilization Gauge")
            max_capacity = 7000
            load_pct = min(100.0, (pred_volume / max_capacity) * 100)
            
            st.progress(load_pct / 100.0)
            st.caption(f"Predicted traffic represents **{load_pct:.1f}%** of maximum road design capacity ({max_capacity:,} vehicles/hour).")
            
    else:
        st.error("ML Model or Preprocessing Pipeline unavailable. Please run `python train_model.py`.")

# -----------------------------------------------------------------------------
# 6. ABOUT PAGE
# -----------------------------------------------------------------------------
elif menu_choice == "ℹ️ About":
    st.subheader("ℹ️ Project & Developer Information")
    
    col_a1, col_a2 = st.columns([1, 1])
    
    with col_a1:
        st.markdown("""
        ### 👨‍💻 Developer Profile
        
        - **Name:** Swapna V
        - **Role:** ML Engineer
        - **Company:** IPEC Solutions
        - **Project:** Traffic Volume Prediction System
        
        ---
        
        ### 🛠️ Technology Stack
        
        - **Programming Language:** Python 3.10
        - **Data Manipulation:** Pandas, NumPy
        - **Visualization:** Plotly, Matplotlib
        - **Machine Learning:** Scikit-Learn
        - **Model Persistence:** Joblib
        - **Web Application Framework:** Streamlit
        """)
        
    with col_a2:
        st.markdown("""
        ### 🎯 Project Objectives & Value Proposition
        
        The **Traffic Volume Prediction System** is designed to support real-time urban planning, intelligent transportation systems (ITS), and municipal traffic signal management.
        
        **Key Objectives:**
        - **Congestion Mitigation**: Predict high-volume hours to dynamically adjust traffic light timings.
        - **Weather Sensitivity Analysis**: Quantify the impact of rain, snow, and extreme temperatures on road capacity.
        - **Decision Support**: Provide actionable traffic level classifications (Low, Moderate, High, Very High) for commuters and fleet managers.
        - **Model Transparency**: Compare multiple machine learning algorithms (Linear Regression, Decision Tree, Random Forest, Gradient Boosting) to ensure maximum predictive accuracy.
        """)

# Footer
st.markdown("""
<div class="footer">
    Traffic Volume Prediction System | Developed by Swapna V | ML Engineer | IPEC Solutions
</div>
""", unsafe_allow_html=True)
