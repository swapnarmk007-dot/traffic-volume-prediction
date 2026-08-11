import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)

DATA_PATH = "data/traffic_data.csv"

def train_and_evaluate():
    print("="*60)
    print("🚦 TRAFFIC VOLUME PREDICTION SYSTEM - MODEL TRAINING PIPELINE")
    print("   Developer: Swapna V | ML Engineer | IPEC Solutions")
    print("="*60)

    # 1. Load Dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run data/generate_dataset.py first.")
        
    print(f"\n[1/8] Loading dataset from '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH, keep_default_na=False)
    print(f"      Loaded {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Data Cleaning
    print("\n[2/8] Performing Data Cleaning...")
    initial_rows = len(df)
    df = df.drop_duplicates()
    duplicates_removed = initial_rows - len(df)
    print(f"      Duplicates removed: {duplicates_removed}")
    
    # Handle missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"      Missing values found: {missing_count}. Imputing/dropping...")
        df = df.ffill().bfill()
    else:
        print("      No missing values found.")

    # 3. Feature Engineering
    print("\n[3/8] Extracting Datetime and Domain Features...")
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d %H:%M:%S")
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek # 0=Mon, 6=Sun
    df['hour'] = df['date'].dt.hour
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Feature column definitions
    num_features = ['hour', 'month', 'day_of_week', 'is_weekend', 'temperature', 'rain_1h', 'snow_1h', 'clouds_all']
    cat_features = ['holiday', 'weather_main', 'weather_description']
    target_col = 'traffic_volume'

    X = df[num_features + cat_features]
    y = df[target_col]

    print(f"      Numerical features ({len(num_features)}): {num_features}")
    print(f"      Categorical features ({len(cat_features)}): {cat_features}")

    # 4. Train/Test Split
    print("\n[4/8] Splitting Data into Train (80%) and Test (20%) Sets...")
    # Using chronological split for time-series integrity
    split_index = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    print(f"      Train set size: {len(X_train)} samples")
    print(f"      Test set size:  {len(X_test)} samples")

    # 5. Preprocessing Pipeline
    print("\n[5/8] Building & Fitting Preprocessing Pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ]
    )

    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # Get feature names after one-hot encoding
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()
    all_feature_names = num_features + encoded_cat_names

    # 6. Model Training & Comparison
    print("\n[6/8] Training Multiple Machine Learning Models...")
    
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=15),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42, learning_rate=0.1, max_depth=6),
        "Tuned Random Forest": RandomForestRegressor(n_estimators=150, random_state=42, max_depth=20, min_samples_split=4, n_jobs=-1)
    }

    results = {}
    best_r2 = -float('inf')
    best_model_name = None
    best_model_obj = None

    for name, model in models.items():
        print(f"      Training {name}...")
        model.fit(X_train_prep, y_train)
        y_pred = model.predict(X_test_prep)

        mae = float(mean_absolute_error(y_test, y_pred))
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        results[name] = {
            "MAE": round(mae, 2),
            "MSE": round(mse, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }

        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
            best_model_obj = model

    # 7. Print Model Comparison Table
    print("\n[7/8] Model Evaluation Summary:")
    print("="*65)
    print(f"{'Model Name':<30} | {'MAE':<8} | {'RMSE':<8} | {'R² Score':<8}")
    print("="*65)
    for name, metrics in results.items():
        print(f"{name:<30} | {metrics['MAE']:<8.2f} | {metrics['RMSE']:<8.2f} | {metrics['R2']:<8.4f}")
    print("="*65)
    print(f"🏆 Best Performing Model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")

    # Feature Importance for best model if supported
    feature_importances = {}
    if hasattr(best_model_obj, 'feature_importances_'):
        importances = best_model_obj.feature_importances_
        importance_pairs = sorted(zip(all_feature_names, importances), key=lambda x: x[1], reverse=True)
        top_importances = dict(importance_pairs[:15]) # Top 15 features
        feature_importances = {k: round(float(v), 4) for k, v in top_importances.items()}

    # 8. Save Best Model, Pipeline, and Metrics
    print("\n[8/8] Saving Model and Preprocessing Artifacts...")
    joblib.dump(best_model_obj, "models/traffic_model.pkl")
    joblib.dump(preprocessor, "models/preprocessing_pipeline.pkl")
    
    # Save metrics metadata for Streamlit UI
    metrics_summary = {
        "best_model_name": best_model_name,
        "results": results,
        "feature_names": all_feature_names,
        "top_feature_importances": feature_importances,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "num_features": num_features,
        "cat_features": cat_features
    }
    
    with open("models/model_metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=4)

    print("      ✅ Saved 'models/traffic_model.pkl'")
    print("      ✅ Saved 'models/preprocessing_pipeline.pkl'")
    print("      ✅ Saved 'models/model_metrics.json'")
    print("\nPipeline execution complete! Ready for Streamlit web deployment.\n")

if __name__ == "__main__":
    train_and_evaluate()
