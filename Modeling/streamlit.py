import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import os
import json
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Weekly Sales Forecasting",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #a7b7ff 0%, #c58bff 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .prediction-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .feature-card {
        background: rgba(255,255,255,0.95);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #1a2980, #26d0ce);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 20px rgba(0,0,0,0.3);
    }
    
    .big-number {
        font-size: 3rem;
        font-weight: bold;
        color: #1a2980;
        text-align: center;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-top: -10px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    model_dir = "Model_joblib"
    
    model_files = {
        "Furniture": {
            "Sales": "best_furniture_sales_model.joblib",
            "Quantity": "best_furniture_quantity_model.joblib"
        },
        "Office Supplies": {
            "Sales": "best_office_sales_model.joblib",
            "Quantity": "best_office_quantity_model.joblib"
        },
        "Technology": {
            "Sales": "best_technology_sales_model.joblib",
            "Quantity": "best_technology_quantity_model.joblib"
        }
    }
    
    loaded_count = 0
    for category, files in model_files.items():
        models[category] = {}
        for target, filename in files.items():
            path = os.path.join(model_dir, filename)
            if os.path.exists(path):
                try:
                    model = None
                    loading_error = None
                    
                    try:
                        model = joblib.load(path)
                    except Exception as e1:
                        loading_error = f"Joblib failed: {str(e1)[:100]}"
                        try:
                            with open(path, 'rb') as f:
                                model = pickle.load(f)
                        except Exception as e2:
                            loading_error = f"Pickle failed: {str(e2)[:100]}"
                    
                    if model is not None:
                        models[category][target] = model
                        loaded_count += 1
                    else:
                        st.warning(f"⚠ Could not load {category} {target}: {loading_error}")
                        
                except Exception as e:
                    st.error(f"❌ Error loading {category} {target}: {str(e)[:100]}")
            else:
                st.warning(f"⚠ Model file not found: {path}")
    
    if loaded_count == 0:
        st.error("❌ No models could be loaded. Using default predictions.")
        for category in ["Furniture", "Office Supplies", "Technology"]:
            models[category] = {
                "Sales": "default_model",
                "Quantity": "default_model"
            }
    
    return models

@st.cache_data
def load_weekly_data():
    data = {}
    data_files = {
        "Furniture": "data/df_furniture_weekly_features2.csv",
        "Office Supplies": "data/df_office_weekly_features2.csv",
        "Technology": "data/df_technology_weekly_features2.csv"
    }
    
    for category, filepath in data_files.items():
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                if 'week_date' in df.columns:
                    df['week_date'] = pd.to_datetime(df['week_date'])
                data[category] = df
            except Exception as e:
                st.warning(f"⚠ Could not load {category} data: {str(e)[:50]}")
        else:
            st.warning(f"⚠ Data file not found: {filepath}")
    
    return data

def get_date_from_year_week(year, week_of_year):
    try:
        first_day = datetime(year, 1, 1)
        if first_day.weekday() <= 3:
            first_monday = first_day - timedelta(days=first_day.weekday())
        else:
            first_monday = first_day + timedelta(days=(7 - first_day.weekday()))
        
        target_date = first_monday + timedelta(weeks=week_of_year - 1)
        return target_date
    except:
        return datetime(year, 1, 1)

def calculate_week_features(date_str):
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    week_of_year = date_obj.isocalendar()[1]
    
    week_rad = 2 * np.pi * (week_of_year - 1) / 52
    week_sin = np.sin(week_rad)
    week_cos = np.cos(week_rad)
    
    return week_sin, week_cos, week_of_year

def get_base_features(category, year, week_of_year, weekly_data):
    target_date = get_date_from_year_week(year, week_of_year)
    target_date_str = target_date.strftime("%Y-%m-%d")
    
    week_sin, week_cos, calculated_week = calculate_week_features(target_date_str)
    month = target_date.month
    quarter = (month - 1) // 3 + 1
    
    base_features = {
        "Year": year,
        "Week_of_Year": week_of_year,
        "Month": month,
        "Quarter": quarter,
        "week_sin": week_sin,
        "week_cos": week_cos,
    }
    
    if category in weekly_data:
        df = weekly_data[category]
        if not df.empty and 'Sales' in df.columns and 'Quantity' in df.columns:
            recent_data = df.sort_values('week_date').tail(52)
            
            if not recent_data.empty:
                lag_1 = recent_data['Sales'].iloc[-1] if len(recent_data) >= 1 else 1000
                lag_2 = recent_data['Sales'].iloc[-2] if len(recent_data) >= 2 else 900
                lag_4 = recent_data['Sales'].iloc[-4] if len(recent_data) >= 4 else 800
                lag_12 = recent_data['Sales'].iloc[-12] if len(recent_data) >= 12 else 700
                roll_mean_4 = recent_data['Sales'].tail(4).mean() if len(recent_data) >= 4 else 950
                roll_mean_12 = recent_data['Sales'].tail(12).mean() if len(recent_data) >= 12 else 850
                
                qty_lag_1 = recent_data['Quantity'].iloc[-1] if len(recent_data) >= 1 else 10
                qty_lag_2 = recent_data['Quantity'].iloc[-2] if len(recent_data) >= 2 else 9
                qty_lag_4 = recent_data['Quantity'].iloc[-4] if len(recent_data) >= 4 else 8
                qty_lag_12 = recent_data['Quantity'].iloc[-12] if len(recent_data) >= 12 else 7
                qty_roll_mean_4 = recent_data['Quantity'].tail(4).mean() if len(recent_data) >= 4 else 9.5
                qty_roll_mean_12 = recent_data['Quantity'].tail(12).mean() if len(recent_data) >= 12 else 8.5
                
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
    
    defaults = {
        "lag_1": 1000, "lag_2": 900, "lag_4": 800, "lag_12": 700,
        "roll_mean_4": 950, "roll_mean_12": 850,
        "Quantity_lag_1": 10, "Quantity_lag_2": 9, 
        "Quantity_lag_4": 8, "Quantity_lag_12": 7,
        "Quantity_roll_mean_4": 9.5, "Quantity_roll_mean_12": 8.5,
    }
    
    for key, value in defaults.items():
        if key not in base_features:
            base_features[key] = value
    
    return base_features

def predict_sales_quantity(category, year, week_of_year, discount, promotion_flag, models, weekly_data):
    try:
        base_features = get_base_features(category, year, week_of_year, weekly_data)
        base_features["Discount"] = discount
        base_features["Promotion_Flag"] = promotion_flag
        
        feature_cols = [
            "Promotion_Flag", "Discount",
            "Year", "Week_of_Year", "Month", "Quarter",
            "week_sin", "week_cos",
            "lag_1", "lag_2", "lag_4", "lag_12",
            "roll_mean_4", "roll_mean_12"
        ]
        
        input_df = pd.DataFrame([{col: base_features.get(col, 0) for col in feature_cols}])
        
        sales_model = models.get(category, {}).get("Sales")
        quantity_model = models.get(category, {}).get("Quantity")
        
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
        st.error(f"❌ Prediction error: {str(e)}")
        
        defaults = {
            "Furniture": (1500.0, 8.0),
            "Office Supplies": (800.0, 40.0),
            "Technology": (2500.0, 3.0)
        }
        default_sales, default_qty = defaults.get(category, (1000.0, 10.0))
        
        print(f"Error in prediction: {str(e)}")
        
        return {
            "predicted_sales": default_sales,
            "predicted_quantity": default_qty,
            "used_features": {},
            "base_features": {},
            "error": str(e),
            "is_fallback": True
        }

with st.sidebar:
    st.markdown("""
    <div style='background: linear-gradient(90deg, #1a2980, #26d0ce); 
                padding: 1rem; border-radius: 10px; margin-bottom: 2rem;'>
        <h3 style='color: white; text-align: center;'>🔧 System Setup</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Load Models & Data")
    
    if st.button("🔄 Load All Models", type="primary", use_container_width=True):
        with st.spinner("Loading models and data..."):
            models = load_models()
            weekly_data = load_weekly_data()
            st.session_state.models = models
            st.session_state.weekly_data = weekly_data
            st.success("✅ Models loaded successfully!")
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Presets")
    
    presets = {
        "📈 Peak Season (Week 51)": {"week": 51, "discount": 20, "promotion": 1},
        "📉 Off Season (Week 5)": {"week": 5, "discount": 0, "promotion": 0},
        "🎯 Clearance Sale": {"week": 30, "discount": 40, "promotion": 1},
        "🏷️ Regular Week": {"week": 25, "discount": 10, "promotion": 1}
    }
    
    for preset_name, preset_values in presets.items():
        if st.button(preset_name, use_container_width=True):
            st.session_state.week = preset_values["week"]
            st.session_state.discount = preset_values["discount"]
            st.session_state.promotion_flag = preset_values["promotion"]
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("### ℹ️ System Info")
    st.info("""
    **Direct Model Prediction**
    - Runs locally
    - Fast predictions
    
    **Categories Available:**
    - Furniture
    - Office Supplies  
    - Technology
    """)

st.markdown("""
<div class="main-header">
    <h1 style='color: white; margin: 0;'>📊 Weekly Sales & Quantity Forecasting</h1>
    <p style='color: white; opacity: 0.9;'>Direct Model Prediction - No API Required</p>
</div>
""", unsafe_allow_html=True)

if 'models' not in st.session_state:
    st.session_state.models = {}
if 'weekly_data' not in st.session_state:
    st.session_state.weekly_data = {}
if 'week' not in st.session_state:
    st.session_state.week = datetime.now().isocalendar()[1]
if 'discount' not in st.session_state:
    st.session_state.discount = 0.0
if 'promotion_flag' not in st.session_state:
    st.session_state.promotion_flag = 0

st.markdown("""
<div style='background: white; padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
    <h2 style='color: #1a2980;'>🎯 Prediction Inputs</h2>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    categories = ["Furniture", "Office Supplies", "Technology"]
    selected_category = st.selectbox(
        "📦 Product Category",
        categories,
        help="Select the product category for prediction"
    )
    
    category_info = {
        "Furniture": "Average price: $150-2000",
        "Office Supplies": "Average price: $10-200", 
        "Technology": "Average price: $500-5000"
    }
    st.caption(category_info.get(selected_category, ""))

with col2:
    current_year = datetime.now().year
    
    selected_year = st.number_input(
        "📅 Year",
        min_value=2014,
        max_value=current_year + 5,
        value=2014,
        step=1,
        help="Select prediction year"
    )
    
    selected_week = st.slider(
        "🗓️ Week of Year",
        min_value=1,
        max_value=52,
        value=st.session_state.week,
        help="Select week number (1-52)"
    )
    
    pred_date = get_date_from_year_week(selected_year, selected_week)
    st.caption(f"📌 Date: {pred_date.strftime('%B %d, %Y')}")

with col3:
    discount = st.slider(
        "💰 Discount (%)",
        min_value=0.0,
        max_value=100.0,
        value=0.0,
        step=0.5,           
        format="%.1f%%",
        help="Set discount percentage"
    )
    
    discount_color = "🔴" if discount == 0 else "🟡" if discount < 30 else "🟢"
    st.metric("Discount Level", f"{discount_color} {discount:.1f}%")
    
    discount_value = discount / 100.0

with col4:
    promotion_options = {
        "No Promotion": 0,
        "Active Promotion": 1
    }
    
    selected_promotion = st.radio(
        "🏷️ Promotion Status",
        list(promotion_options.keys()),
        index=st.session_state.promotion_flag,
        horizontal=True
    )
    
    promotion_flag = promotion_options[selected_promotion]
    
    if promotion_flag == 1:
        st.success("✅ Promotion Active")
    else:
        st.info("ℹ️ No Promotion")

st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_button = st.button(
        "🚀 PREDICT SALES & QUANTITY", 
        type="primary", 
        use_container_width=True,
        disabled=len(st.session_state.models) == 0
    )

if predict_button:
    if not st.session_state.models:
        st.error("❌ Please load models first using the sidebar button!")
        st.info("💡 Click '🔄 Load All Models' in the sidebar to load the required models.")
    else:
        try:
            with st.spinner("🔮 Making predictions..."):
                results = predict_sales_quantity(
                    category=selected_category,
                    year=selected_year,
                    week_of_year=selected_week,
                    discount=discount_value,
                    promotion_flag=promotion_flag,
                    models=st.session_state.models,
                    weekly_data=st.session_state.weekly_data
                )
                
                st.session_state.prediction_results = results
                
                if results.get('is_fallback'):
                    st.warning("⚠ Using fallback values due to prediction error. Please check model files.")
                
                st.markdown("---")
                st.markdown("""
                <div style='background: linear-gradient(90deg, #1a2980, #26d0ce); 
                            padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
                    <h2 style='color: white; text-align: center;'>📊 Prediction Results</h2>
                </div>
                """, unsafe_allow_html=True)
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <h3 style='color: #1a2980;'>💰 Predicted Sales</h3>
                        <div class='big-number'>${results['predicted_sales']:,.2f}</div>
                        <p class='metric-label'>Expected Revenue</p>
                        <div style='background: #e3f2fd; padding: 10px; border-radius: 8px; margin-top: 15px;'>
                            <small>💡 Average: ${results['predicted_sales']/results['predicted_quantity']:.2f} per unit</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"""
                    <div class='prediction-card'>
                        <h3 style='color: #1a2980;'>📦 Predicted Quantity</h3>
                        <div class='big-number'>{results['predicted_quantity']:,.0f}</div>
                        <p class='metric-label'>Units Expected</p>
                        <div style='background: #e3f2fd; padding: 10px; border-radius: 8px; margin-top: 15px;'>
                            <small>📈 Based on week {selected_week} trends</small>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📈 Prediction Visualization")
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=['Sales ($)', 'Quantity'],
                    y=[results['predicted_sales'], results['predicted_quantity']],
                    name='Prediction',
                    marker_color=['#667eea', '#764ba2'],
                    text=[f"${results['predicted_sales']:,.0f}", f"{results['predicted_quantity']:,.0f}"],
                    textposition='auto',
                ))
                
                fig.update_layout(
                    title=f"{selected_category} - Week {selected_week}, {selected_year}",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color="#1a2980"),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 🔍 Features Used in Prediction")
                
                if results['used_features']:
                    features_df = pd.DataFrame.from_dict(
                        results['used_features'], 
                        orient='index', 
                        columns=['Value']
                    )
                    features_df = features_df.sort_values('Value', ascending=False)
                    
                    tab1, tab2 = st.tabs(["📋 Feature Values", "📊 Feature Distribution"])
                    
                    with tab1:
                        st.dataframe(features_df, use_container_width=True)
                    
                    with tab2:
                        fig2 = px.bar(
                            features_df.reset_index(),
                            x='index',
                            y='Value',
                            title="Feature Importance (Relative)",
                            labels={'index': 'Feature', 'Value': 'Value'},
                            color='Value',
                            color_continuous_scale='Viridis'
                        )
                        fig2.update_layout(xaxis_tickangle=-45)
                        st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("### 💾 Download Results")
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                
                with col_dl1:
                    results_df = pd.DataFrame({
                        'Metric': ['Predicted Sales', 'Predicted Quantity'],
                        'Value': [results['predicted_sales'], results['predicted_quantity']],
                        'Category': [selected_category, selected_category],
                        'Year': [selected_year, selected_year],
                        'Week': [selected_week, selected_week],
                        'Discount': [f"{discount}%", f"{discount}%"],
                        'Promotion': [selected_promotion, selected_promotion],
                        'Date': [pred_date.strftime('%Y-%m-%d'), pred_date.strftime('%Y-%m-%d')]
                    })
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"prediction_{selected_category}_{selected_year}_week{selected_week}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col_dl2:
                    json_data = json.dumps(results, indent=2, default=str)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_data,
                        file_name=f"prediction_{selected_category}_{selected_year}_week{selected_week}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col_dl3:
                    if st.button("💾 Save to Session", use_container_width=True):
                        st.session_state.last_prediction = {
                            'results': results,
                            'inputs': {
                                'category': selected_category,
                                'year': selected_year,
                                'week': selected_week,
                                'discount': discount,
                                'promotion': selected_promotion
                            }
                        }
                        st.success("✅ Saved to session!")
                
                if selected_category in st.session_state.weekly_data:
                    st.markdown("### 📊 Historical Comparison")
                    
                    df_category = st.session_state.weekly_data[selected_category]
                    if not df_category.empty and 'Sales' in df_category.columns:
                        avg_sales = df_category['Sales'].mean()
                        avg_qty = df_category['Quantity'].mean() if 'Quantity' in df_category.columns else 0
                        
                        col_hist1, col_hist2 = st.columns(2)
                        
                        with col_hist1:
                            sales_diff = ((results['predicted_sales'] - avg_sales) / avg_sales * 100)
                            st.metric(
                                "Sales vs Historical Average",
                                f"${results['predicted_sales']:,.0f}",
                                f"{sales_diff:+.1f}%"
                            )
                        
                        with col_hist2:
                            if avg_qty > 0:
                                qty_diff = ((results['predicted_quantity'] - avg_qty) / avg_qty * 100)
                                st.metric(
                                    "Quantity vs Historical Average",
                                    f"{results['predicted_quantity']:,.0f}",
                                    f"{qty_diff:+.1f}%"
                                )
        
        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            st.info("Please try again or reload the models from the sidebar.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <p>📊 <strong>Direct Model Prediction System</strong></p>
    <small>Models loaded from: Model_joblib/ • Data from: data/</small>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Current Status")
    
    if st.session_state.models:
        loaded_count = sum(len(models) for models in st.session_state.models.values())
        st.success(f"✅ {loaded_count}/6 models loaded")
        
        loaded_categories = []
        for category in ["Furniture", "Office Supplies", "Technology"]:
            if category in st.session_state.models:
                loaded_categories.append(f"✓ {category}")
        
        if loaded_categories:
            st.info("Loaded Categories:")
            for cat in loaded_categories:
                st.write(cat)
    else:
        st.warning("⚠ No models loaded")
    
    if st.session_state.weekly_data:
        st.success(f"✅ {len(st.session_state.weekly_data)}/3 datasets loaded")
# في نهاية الsidebar
with st.sidebar:
    st.markdown("---")
    
    # عرض الرابط العام
    public_url = "https://sales-forecast-app.streamlit.app/"
    
    st.markdown("### 📱 Open on Other Devices")
    st.markdown(f"""
    **Public URL:**
    ```
    {public_url}
    ```
    """)
    
    # زر لفتح الرابط مباشرة
    if st.button("🌍 Open Public Link", use_container_width=True):
        st.markdown(f'<meta http-equiv="refresh" content="0;url={public_url}">', unsafe_allow_html=True)
        st.success("Opening public link...")
    
    # QR Code للجوال
    try:
        import qrcode
        from PIL import Image
        import io
        
        with st.expander("📲 Scan QR Code for Mobile"):
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(public_url)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white")
            
            # تحويل الصورة
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            img_bytes.seek(0)
            
            st.image(img_bytes, width=200)
            st.caption("Scan with phone camera")
    except:
        pass