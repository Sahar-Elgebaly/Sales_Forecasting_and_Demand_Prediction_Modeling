import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
import os
import tempfile
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1) Category → CSV Mapping
# ===============================
DATASETS = {
    "Furniture": "data/df_furniture_weekly_features2.csv",
    "Office Supplies": "data/df_office_weekly_features2.csv",
    "Technology": "data/df_technology_weekly_features2.csv",
}

# ===============================
# 2) Feature columns - FOR SALES MODELS ONLY
# ===============================
numeric_features = [
    "Promotion_Flag", "Discount",
    "Year", "Week_of_Year", "Month", "Quarter",
    "week_sin", "week_cos",
    "lag_1", "lag_2", "lag_4", "lag_12",
    "roll_mean_4", "roll_mean_12"
]

# ===============================
# 3) Models → category + target
# ===============================
MODELS = {
    "best_furniture_sales_model":    ("Furniture", "Sales"),
    "best_furniture_quantity_model": ("Furniture", "Quantity"),

    "best_office_sales_model":       ("Office Supplies", "Sales"),
    "best_office_quantity_model":    ("Office Supplies", "Quantity"),

    "best_technology_sales_model":   ("Technology", "Sales"),
    "best_technology_quantity_model":("Technology", "Quantity"),
}

# folder for joblib models:
MODEL_DIR = "Model_joblib"

# ===============================
# 4) MLflow Setup
# ===============================
mlflow.set_tracking_uri("mlruns")

def load_dataset(category):
    csv_path = DATASETS[category]
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # Convert date column if exists
    if 'week_date' in df.columns:
        df['week_date'] = pd.to_datetime(df['week_date'])
    return df

def detect_model_type(model):
    """Detect if model is Holt-Winters or sklearn"""
    model_class = str(type(model))
    
    if 'HoltWinters' in model_class or 'ExponentialSmoothing' in model_class:
        return "holt_winters"
    elif 'sklearn' in model_class.lower() or 'Pipeline' in model_class:
        return "sklearn"
    else:
        return "unknown"

def prepare_data_for_holt_winters(df, target):
    """
    For Holt-Winters: We need time series data only
    Returns: (None, y) because Holt-Winters doesn't use X features
    """
    y = df[target].values
    
    # Holt-Winters expects a proper time series
    # We'll create a time index if available
    if 'week_date' in df.columns:
        dates = pd.to_datetime(df['week_date'])
        # Create a time series with dates as index
        ts_series = pd.Series(y, index=dates)
        return None, ts_series
    else:
        # If no dates, just use index as time
        return None, pd.Series(y)

def evaluate_holt_winters_model(model, y_true_series):
    """
    Evaluate Holt-Winters model properly
    Holt-Winters models are trained on the entire series
    We need to do proper time series evaluation
    """
    try:
        # Get the length of the training data
        train_length = len(y_true_series)
        
        # For in-sample predictions (fitted values)
        if hasattr(model, 'fittedvalues'):
            y_pred = model.fittedvalues.values
        elif hasattr(model, 'predict'):
            # Predict on the training period
            y_pred = model.predict(start=0, end=train_length-1)
        else:
            # If we can't get fitted values, use forecast
            y_pred = model.forecast(steps=train_length)
        
        # Ensure same length
        if len(y_pred) != len(y_true_series):
            y_pred = y_pred[:len(y_true_series)]
        
        return y_pred
        
    except Exception as e:
        print(f"Holt-Winters evaluation error: {e}")
        # Return simple forecast as fallback
        return model.forecast(steps=len(y_true_series))

# ===============================
# 5) Main Loop - FIXED VERSION
# ===============================

for model_name, (category, target) in MODELS.items():

    print(f"\n{'='*60}")
    print(f"Processing: {model_name}")
    print(f"Category: {category}, Target: {target}")
    print('='*60)

    # Load dataset
    try:
        df = load_dataset(category)
        print(f"Dataset loaded: {df.shape[0]} rows")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        continue

    # Load model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        continue

    try:
        model = joblib.load(model_path)
        model_type = detect_model_type(model)
        print(f"Model type: {model_type}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        continue

    # Prepare data and evaluate BASED ON MODEL TYPE
    if model_type == "holt_winters":
        print("⚠ NOTE: This is a Holt-Winters model (for time series)")
        print("   - Does NOT use feature columns")
        print("   - Uses only historical patterns")
        print("   - R² may be negative if not properly evaluated")
        
        # For Holt-Winters: use time series data only
        X = None
        y_true_series = prepare_data_for_holt_winters(df, target)[1]
        y_true = y_true_series.values
        
        # Get predictions (fitted values)
        y_pred = evaluate_holt_winters_model(model, y_true_series)
        
        # For Holt-Winters, we should do proper time series split
        # But for now, use a simple train-test split
        split_idx = int(len(y_true) * 0.8)
        
        y_train = y_true[:split_idx]
        y_test = y_true[split_idx:]
        
        # Forecast for test period
        try:
            forecast_steps = len(y_test)
            y_test_pred = model.forecast(steps=forecast_steps)
            
            # Calculate metrics on TEST SET only
            if len(y_test) > 0 and len(y_test_pred) > 0:
                mse = mean_squared_error(y_test, y_test_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_test_pred)
                r2 = r2_score(y_test, y_test_pred)
                
                print(f"Test Metrics (forecast) - MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
                
                # Also calculate in-sample metrics for reference
                y_train_pred = y_pred[:split_idx]
                r2_train = r2_score(y_train, y_train_pred) if len(y_train) > 0 else 0
                print(f"Train R2 (in-sample): {r2_train:.4f}")
            else:
                mse = rmse = mae = r2 = 0.0
                
        except Exception as e:
            print(f"Forecast evaluation failed: {e}")
            # Fallback to in-sample metrics
            if len(y_true) > 0 and len(y_pred) > 0:
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                print(f"In-sample Metrics - MSE: {mse:.2f}, R2: {r2:.4f}")
            else:
                mse = rmse = mae = r2 = 0.0
        
        # For plotting
        plot_y_true = y_true
        plot_y_pred = y_pred
        
    else:
        # For sklearn models
        print("✓ This is a scikit-learn model (uses features)")
        X = df[numeric_features]
        y_true = df[target].values
        
        # Check if all features exist
        missing_features = [f for f in numeric_features if f not in df.columns]
        if missing_features:
            print(f"⚠ Missing features: {missing_features}")
            # Fill missing with 0
            for feature in missing_features:
                df[feature] = 0
        
        X = df[numeric_features]
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        print(f"Metrics - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        plot_y_true = y_true
        plot_y_pred = y_pred

    # Create MLflow experiment
    experiment_name = f"{category}-{target}"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=model_name):
        
        # Log basic info
        mlflow.log_param("category", category)
        mlflow.log_param("target", target)
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("model_file", model_name)
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("model_class", str(type(model)))
        
        # Log metrics
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        
        # Add warning for negative R2
        if r2 < 0:
            mlflow.log_param("warning", "Negative R2 - model may not be suitable")
            print("⚠ WARNING: Negative R2 score logged")
        
        # Log model parameters
        if model_type == "sklearn" and hasattr(model, "get_params"):
            params = model.get_params()
            for k, v in params.items():
                if v is not None and isinstance(v, (int, float, str, bool)):
                    try:
                        mlflow.log_param(f"sk_{k}", v)
                    except:
                        pass
        elif model_type == "holt_winters":
            # Log Holt-Winters parameters
            if hasattr(model, 'params'):
                for k, v in model.params.items():
                    if isinstance(v, (int, float, str, bool)):
                        try:
                            mlflow.log_param(f"hw_{k}", v)
                        except:
                            pass
        
        # Plot Actual vs Predicted
        try:
            plt.figure(figsize=(14, 6))
            
            # Plot 1: Scatter plot
            plt.subplot(1, 2, 1)
            plt.scatter(plot_y_true, plot_y_pred, alpha=0.6, s=30)
            
            # Add perfect prediction line
            max_val = max(np.max(plot_y_true), np.max(plot_y_pred))
            min_val = min(np.min(plot_y_true), np.min(plot_y_pred))
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.7)
            
            plt.xlabel(f"Actual {target}")
            plt.ylabel(f"Predicted {target}")
            plt.title(f"{model_name}\nR² = {r2:.4f}")
            plt.grid(True, alpha=0.3)
            
            # Add stats text box
            stats_text = f"MSE: {mse:.2f}\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}"
            plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # Plot 2: Time series comparison
            plt.subplot(1, 2, 2)
            if 'week_date' in df.columns and len(plot_y_true) == len(df):
                dates = pd.to_datetime(df['week_date'])
                plt.plot(dates, plot_y_true, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
                plt.plot(dates, plot_y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
                plt.xlabel('Date')
                plt.ylabel(target)
                plt.title('Time Series Comparison')
                plt.legend()
                plt.xticks(rotation=45)
            else:
                # Plot by index
                indices = np.arange(len(plot_y_true))
                plt.plot(indices, plot_y_true, 'b-', label='Actual', alpha=0.7, linewidth=1.5)
                plt.plot(indices, plot_y_pred, 'r-', label='Predicted', alpha=0.7, linewidth=1.5)
                plt.xlabel('Index')
                plt.ylabel(target)
                plt.title('Actual vs Predicted')
                plt.legend()
            
            plt.tight_layout()
            
            # Save and log plot
            with tempfile.TemporaryDirectory() as tmpdir:
                plot_path = os.path.join(tmpdir, f"{model_name}_evaluation.png")
                plt.savefig(plot_path, bbox_inches="tight", dpi=150)
                mlflow.log_artifact(plot_path, "plots")
            plt.close()
            
            print("✓ Evaluation plot created")
            
        except Exception as e:
            print(f"[WARN] Plot creation failed: {e}")
        
        # Log model to MLflow
        try:
            if model_type == "sklearn":
                # For sklearn models
                signature = infer_signature(X, plot_y_pred) if X is not None else None
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=model_name,
                    signature=signature,
                    input_example=X.iloc[:3] if X is not None else None
                )
                print("✓ Model logged with mlflow.sklearn")
                
            elif model_type == "holt_winters":
                # For Holt-Winters - log as pyfunc
                class HoltWintersMLflowWrapper(mlflow.pyfunc.PythonModel):
                    def __init__(self, hw_model):
                        self.hw_model = hw_model
                        
                    def predict(self, context, model_input):
                        """
                        Holt-Winters wrapper for MLflow
                        model_input should contain 'steps' column or we use length
                        """
                        # If input is empty or has no steps info, forecast 1 step
                        if model_input is None or len(model_input) == 0:
                            steps = 1
                        else:
                            steps = len(model_input)
                        
                        forecast = self.hw_model.forecast(steps=steps)
                        return forecast.values if hasattr(forecast, 'values') else forecast
                
                wrapper = HoltWintersMLflowWrapper(model)
                
                # Log with pyfunc
                mlflow.pyfunc.log_model(
                    python_model=wrapper,
                    artifact_path="model",
                    registered_model_name=model_name,
                    conda_env={
                        'channels': ['conda-forge'],
                        'dependencies': [
                            'python>=3.8',
                            'statsmodels>=0.14.0',
                            'numpy>=1.21.0',
                            'pandas>=1.3.0'
                        ]
                    }
                )
                print("✓ Holt-Winters logged with pyfunc")
                
        except Exception as e:
            print(f"[ERROR] Failed to log model: {e}")
            # Fallback: save model file
            try:
                mlflow.log_artifact(model_path, "model_files")
                print("✓ Model file logged as artifact")
            except Exception as e2:
                print(f"[ERROR] Artifact logging failed: {e2}")
        
        print(f"✓ MLflow run completed for {model_name}\n")

print("\n" + "="*60)
print("MLFLOW LOGGING COMPLETED")
print("="*60)

# Summary with R2 analysis
print("\n📊 R² SCORE ANALYSIS:")
print("-" * 50)

summary_data = []
for model_name, (category, target) in MODELS.items():
    model_path = os.path.join(MODEL_DIR, f"{model_name}.joblib")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            model_type = detect_model_type(model)
            
            # Load data
            df = load_dataset(category)
            
            # Simple evaluation
            if model_type == "holt_winters":
                # For Holt-Winters, do simple forecast
                y_true = df[target].values
                split_idx = int(len(y_true) * 0.8)
                y_test = y_true[split_idx:]
                
                try:
                    forecast = model.forecast(steps=len(y_test))
                    r2 = r2_score(y_test, forecast) if len(y_test) > 0 else -999
                except:
                    r2 = -999
            else:
                # For sklearn
                X = df[numeric_features]
                # Check missing features
                for feature in numeric_features:
                    if feature not in df.columns:
                        df[feature] = 0
                X = df[numeric_features]
                y_true = df[target].values
                y_pred = model.predict(X)
                r2 = r2_score(y_true, y_pred)
            
            status = "✅" if r2 > 0 else "⚠" if r2 > -1 else "❌"
            summary_data.append([model_name, category, target, model_type, f"{r2:.4f}", status])
            
        except Exception as e:
            summary_data.append([model_name, category, target, "ERROR", str(e)[:30], "❌"])
    else:
        summary_data.append([model_name, category, target, "NOT FOUND", "N/A", "❌"])

# Print summary table
print("\n{:<35} {:<20} {:<10} {:<15} {:<10} {:<5}".format(
    "Model", "Category", "Target", "Type", "R²", "Status"))
print("-" * 100)

for row in summary_data:
    print("{:<35} {:<20} {:<10} {:<15} {:<10} {:<5}".format(*row))

print("\n🔍 INTERPRETATION:")
print("• ✅ R² > 0: Model explains variance (good)")
print("• ⚠ -1 < R² < 0: Model worse than mean (needs improvement)")
print("• ❌ R² < -1: Serious issues with model or evaluation")

print("\nTo view results:")
print("  mlflow ui")
print("  Then open: http://localhost:5000")