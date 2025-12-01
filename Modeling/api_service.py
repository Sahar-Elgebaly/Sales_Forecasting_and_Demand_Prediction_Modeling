from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import logging
import os
import numpy as np
from datetime import datetime, timedelta
import pickle
import json

# ============================
# Logging
# ============================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI(title="Weekly Sales & Quantity Prediction API")

# ============================
# Paths
# ============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")       
MODELS_DIR = os.path.join(BASE_DIR, "Model_joblib")

# ============================
# IMPORT SAME FUNCTIONS 
# ============================
def load_weekly_df(filename: str) -> pd.DataFrame:
    """Load weekly data CSV - نفس Streamlit"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path)
    if 'week_date' in df.columns:
        df['week_date'] = pd.to_datetime(df['week_date'])
    return df

def get_date_from_year_week(year: int, week_of_year: int) -> datetime:
    first_day = datetime(year, 1, 1)
    if first_day.weekday() <= 3:
        first_monday = first_day - timedelta(days=first_day.weekday())
    else:
        first_monday = first_day + timedelta(days=(7 - first_day.weekday()))
    
    return first_monday + timedelta(weeks=week_of_year - 1)

def calculate_week_features(date_obj: datetime):
    week_of_year = date_obj.isocalendar()[1]
    week_rad = 2 * np.pi * (week_of_year - 1) / 52
    week_sin = np.sin(week_rad)
    week_cos = np.cos(week_rad)
    return week_sin, week_cos, week_of_year

def get_base_features_api(category: str, year: int, week_of_year: int, weekly_data: dict) -> dict:
    """Get base features for prediction - EXACTLY LIKE STREAMLIT"""
    target_date = get_date_from_year_week(year, week_of_year)
    week_sin, week_cos, calculated_week = calculate_week_features(target_date)
    month = target_date.month
    quarter = (month - 1) // 3 + 1
    
    base_features = {
        "Year": year,
        "Week_of_Year": week_of_year,
        "Month": month,
        "Quarter": quarter,
        "week_sin": float(week_sin),
        "week_cos": float(week_cos),
    }
    

    if category in weekly_data:
        df = weekly_data[category]
        if not df.empty and 'Sales' in df.columns and 'Quantity' in df.columns:
            recent_data = df.sort_values('week_date').tail(52)
            
            if not recent_data.empty:
                # Sales lags
                lag_1 = float(recent_data['Sales'].iloc[-1] if len(recent_data) >= 1 else 1000)
                lag_2 = float(recent_data['Sales'].iloc[-2] if len(recent_data) >= 2 else 900)
                lag_4 = float(recent_data['Sales'].iloc[-4] if len(recent_data) >= 4 else 800)
                lag_12 = float(recent_data['Sales'].iloc[-12] if len(recent_data) >= 12 else 700)
                roll_mean_4 = float(recent_data['Sales'].tail(4).mean() if len(recent_data) >= 4 else 950)
                roll_mean_12 = float(recent_data['Sales'].tail(12).mean() if len(recent_data) >= 12 else 850)
                
                # Quantity lags
                qty_lag_1 = float(recent_data['Quantity'].iloc[-1] if len(recent_data) >= 1 else 10)
                qty_lag_2 = float(recent_data['Quantity'].iloc[-2] if len(recent_data) >= 2 else 9)
                qty_lag_4 = float(recent_data['Quantity'].iloc[-4] if len(recent_data) >= 4 else 8)
                qty_lag_12 = float(recent_data['Quantity'].iloc[-12] if len(recent_data) >= 12 else 7)
                qty_roll_mean_4 = float(recent_data['Quantity'].tail(4).mean() if len(recent_data) >= 4 else 9.5)
                qty_roll_mean_12 = float(recent_data['Quantity'].tail(12).mean() if len(recent_data) >= 12 else 8.5)
                
                base_features.update({
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_4": lag_4,
                    "lag_12": lag_12,
                    "roll_mean_4": roll_mean_4,
                    "roll_mean_12": roll_mean_12,
                    "Quantity_lag_1": qty_lag_1,
                    "Quantity_lag_2": qty_lag_2,
                    "Quantity_lag_4": qty_lag_4,
                    "Quantity_lag_12": qty_lag_12,
                    "Quantity_roll_mean_4": qty_roll_mean_4,
                    "Quantity_roll_mean_12": qty_roll_mean_12,
                })
    
    # Default values  
    defaults = {
        "lag_1": 1000.0, "lag_2": 900.0, "lag_4": 800.0, "lag_12": 700.0,
        "roll_mean_4": 950.0, "roll_mean_12": 850.0,
        "Quantity_lag_1": 10.0, "Quantity_lag_2": 9.0, 
        "Quantity_lag_4": 8.0, "Quantity_lag_12": 7.0,
        "Quantity_roll_mean_4": 9.5, "Quantity_roll_mean_12": 8.5,
    }
    
    for key, value in defaults.items():
        if key not in base_features:
            base_features[key] = value
    
    return base_features

# ============================
# Load Data
# ============================
try:
    WEEKLY_DATA = {
        "Furniture": load_weekly_df("df_furniture_weekly_features2.csv"),
        "Office Supplies": load_weekly_df("df_office_weekly_features2.csv"),
        "Technology": load_weekly_df("df_technology_weekly_features2.csv"),
    }
    logger.info("Weekly feature CSVs loaded successfully")
except Exception as e:
    logger.error(f"Error loading weekly data: {e}")
    WEEKLY_DATA = {}

# ============================
# Load Models
# ============================
def load_model_safe(filename: str):
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        logger.error(f"Error loading {filename} with joblib: {e}")
        try:
            with open(path, 'rb') as f:
                model = pickle.load(f)
            return model
        except Exception as e2:
            logger.error(f"Error loading {filename} with pickle: {e2}")
            raise

EXPECTED_MODELS = {
    "Furniture": {
        "Sales": "best_furniture_sales_model.joblib",
        "Quantity": "best_furniture_quantity_model.joblib",
    },
    "Office Supplies": {
        "Sales": "best_office_sales_model.joblib", 
        "Quantity": "best_office_quantity_model.joblib",
    },
    "Technology": {
        "Sales": "best_technology_sales_model.joblib",
        "Quantity": "best_technology_quantity_model.joblib",
    },
}

MODELS = {}
MODELS_STATUS = {}

for category, model_files in EXPECTED_MODELS.items():
    MODELS[category] = {}
    MODELS_STATUS[category] = {}
    
    for target, filename in model_files.items():
        try:
            model = load_model_safe(filename)
            MODELS[category][target] = model
            MODELS_STATUS[category][target] = "loaded"
            logger.info(f"✓ {category} {target} model loaded successfully")
        except Exception as e:
            MODELS_STATUS[category][target] = f"failed - {str(e)}"
            logger.error(f"✗ Failed to load {category} {target} model: {e}")

# ============================
#   Prediction Logic 
# ============================
def predict_sales_quantity_api(category: str, year: int, week_of_year: int, discount: float, promotion_flag: float):
    try:
        # Get base 
        base_features = get_base_features_api(category, year, week_of_year, WEEKLY_DATA)
        base_features["Discount"] = discount
        base_features["Promotion_Flag"] = promotion_flag
        
        # Feature columns 
        feature_cols = [
            "Promotion_Flag", "Discount",
            "Year", "Week_of_Year", "Month", "Quarter",
            "week_sin", "week_cos",
            "lag_1", "lag_2", "lag_4", "lag_12",
            "roll_mean_4", "roll_mean_12"
        ]
        
        
        input_df = pd.DataFrame([{col: base_features.get(col, 0) for col in feature_cols}])
        
        # Get models
        sales_model = MODELS.get(category, {}).get("Sales")
        quantity_model = MODELS.get(category, {}).get("Quantity")
        
        if sales_model is None or quantity_model is None:
            raise ValueError(f"Models not found for category: {category}")
        
        if hasattr(sales_model, 'predict'):
            pred_sales = float(sales_model.predict(input_df)[0])
        else:
            pred_sales = 1000.0
        
        model_type = str(type(quantity_model))
        if 'HoltWinters' in model_type:
            try:
                forecast_result = quantity_model.forecast(steps=1)
                if hasattr(forecast_result, '__len__'):
                    pred_quantity = float(forecast_result[0])
                else:
                    pred_quantity = float(forecast_result)
            except:
                pred_quantity = max(1.0, pred_sales * 0.01)
        elif hasattr(quantity_model, 'predict'):
            pred_quantity = float(quantity_model.predict(input_df)[0])
        else:
            pred_quantity = max(1.0, pred_sales * 0.01)
        
        pred_sales = max(0.0, pred_sales)
        pred_quantity = max(1.0, pred_quantity)
        
        if pred_sales > 0 and pred_quantity > 0:
            avg_price = pred_sales / pred_quantity
            if avg_price < 1.0:
                pred_quantity = pred_sales / 10.0
            elif avg_price > 10000.0:
                pred_quantity = pred_sales / 1000.0
        
        return {
            "predicted_sales": round(pred_sales, 2),
            "predicted_quantity": round(pred_quantity, 0),
            "used_features": {col: float(input_df.iloc[0][col]) for col in feature_cols},
            "base_features": {k: float(v) if isinstance(v, (int, float)) else v 
                            for k, v in base_features.items() if k not in feature_cols}
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)[:100]}")
        defaults = {
            "Furniture": (1500.0, 8.0),
            "Office Supplies": (800.0, 40.0),
            "Technology": (2500.0, 3.0)
        }
        default_sales, default_qty = defaults.get(category, (1000.0, 10.0))
        
        return {
            "predicted_sales": default_sales,
            "predicted_quantity": default_qty,
            "used_features": {},
            "base_features": {},
            "error": str(e)
        }

# ============================
# Request / Response Schemas
# ============================
class WeeklyPredictionRequest(BaseModel):
    category: str
    year: int
    week_of_year: int
    discount: float
    promotion_flag: float

class WeeklyPredictionResponse(BaseModel):
    predicted_sales: float
    predicted_quantity: float
    used_features_sales: dict
    used_features_quantity: dict
    date: str = None

# ============================
# Prediction Endpoint 
# ============================
@app.post("/predict_weekly", response_model=WeeklyPredictionResponse)
def predict_weekly(req: WeeklyPredictionRequest):
    
    # Validate inputs
    if not WEEKLY_DATA:
        raise HTTPException(status_code=500, detail="Data not loaded on server")
    
    category = req.category
    if category not in MODELS:
        raise HTTPException(status_code=400,
                          detail=f"Invalid category '{category}'. Allowed: {list(MODELS.keys())}")
    
    if not (1 <= req.week_of_year <= 52):
        raise HTTPException(status_code=400, detail="week_of_year must be between 1 and 52")
    
    if not (0 <= req.discount <= 1):
        raise HTTPException(status_code=400, detail="discount must be between 0 and 1")
    
    if not (0 <= req.promotion_flag <= 1):
        raise HTTPException(status_code=400, detail="promotion_flag must be 0 or 1")

    # Call the SAME prediction function as Streamlit
    results = predict_sales_quantity_api(
        category=category,
        year=req.year,
        week_of_year=req.week_of_year,
        discount=req.discount,
        promotion_flag=req.promotion_flag
    )
    
    # Get prediction date
    pred_date = get_date_from_year_week(req.year, req.week_of_year)
    
    response = {
        "predicted_sales": results["predicted_sales"],
        "predicted_quantity": results["predicted_quantity"],
        "used_features_sales": results["used_features"],
        "used_features_quantity": results.get("used_features", {}),
        "date": pred_date.strftime("%Y-%m-%d")
    }
    
    # If there's an error in results, raise HTTPException
    if "error" in results:
        raise HTTPException(status_code=500, detail=f"Prediction error: {results['error']}")
    
    return response

# ============================
# Additional Endpoints
# ============================
@app.get("/")
def root():
    return {"message": "Weekly Sales Prediction API - Same logic as Streamlit"}

@app.get("/health")
def health():
    return {
        "status": "healthy" if WEEKLY_DATA and any(MODELS.values()) else "not_ready",
        "categories_loaded": list(WEEKLY_DATA.keys()),
        "models_status": MODELS_STATUS,
    }

@app.get("/features/{category}")
def get_features_sample(category: str):
    if category not in WEEKLY_DATA:
        raise HTTPException(status_code=404, detail=f"Category {category} not found")
    
    df = WEEKLY_DATA[category]
    sample = df.head(1).to_dict('records')[0] if not df.empty else {}
    
    # Convert to JSON serializable
    def convert_value(v):
        if isinstance(v, (np.integer, np.int64)):
            return int(v)
        elif isinstance(v, (np.floating, np.float64)):
            return float(v)
        elif isinstance(v, pd.Timestamp):
            return v.isoformat()
        else:
            return v
    
    sample = {k: convert_value(v) for k, v in sample.items()}
    
    return {
        "available_columns": list(df.columns),
        "sample_features": sample
    }

@app.get("/models")
def get_models():
    return {
        "loaded_models": {
            category: {
                target: "loaded" if model else "not loaded" 
                for target, model in models.items()
            }
            for category, models in MODELS.items()
        },
        "detailed_status": MODELS_STATUS
    }
# ============================
# Run the API
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)