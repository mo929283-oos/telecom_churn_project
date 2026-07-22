"""
===========================================================
TelcoGuard AI - All-in-One Application
===========================================================
Author : Mostafa Ahmed
Project: TelcoGuard AI
===========================================================
"""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# 1. PROJECT PATHS & CONFIG
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "xgb_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
MODEL_COLUMNS_PATH = MODEL_DIR / "model_columns.pkl"

PROJECT_NAME = "TelcoGuard AI"
PROJECT_VERSION = "2.0"
AUTHOR = "Mostafa Ahmed"
COMPANY = "TelcoGuard Analytics"
MODEL_NAME = "XGBoost Classifier"
MODEL_TYPE = "Binary Classification"
TARGET = "Customer Churn"

# Theme Colors
PRIMARY = "#4F8BF9"
SECONDARY = "#1E293B"
SUCCESS = "#16A34A"
WARNING = "#F59E0B"
DANGER = "#DC2626"
INFO = "#38BDF8"
BACKGROUND = "#0F172A"
CARD = "#1E293B"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"

LOW_RISK = 0.40
MEDIUM_RISK = 0.70
HIGH_RISK = 1.00

DEFAULT_CUSTOMER_VALUE = 1200
DEFAULT_RETENTION_COST = 120
DEFAULT_PROFIT_MARGIN = 0.35

NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

YES_NO_COLUMNS = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"
]

CATEGORICAL_COLUMNS = {
    "gender": ["Male", "Female"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaymentMethod": [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ]
}

RISK_LABELS = {
    "Low": "#22C55E",
    "Medium": "#F59E0B",
    "High": "#EF4444"
}

HERO_TITLE = "TelcoGuard AI"
HERO_SUBTITLE = "Enterprise Customer Churn Intelligence Platform Powered by XGBoost & Explainable AI"
FOOTER = "© 2026 TelcoGuard AI | Built with Streamlit + Plotly"

YES_NO_MAPPING = {"Yes": 1, "No": 0}
GENDER_MAPPING = {"Male": 1, "Female": 0}


# =============================================================================
# 2. STYLES (CSS)
# =============================================================================

def load_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"]{
    font-family:'Poppins',sans-serif;
}

.stApp{
    background: linear-gradient(135deg, #07111f 0%, #0f172a 30%, #172554 100%);
    color:white;
}

header, footer, #MainMenu {
    visibility:hidden;
}

section[data-testid="stSidebar"]{
    background: linear-gradient(180deg, rgba(20,30,48,.95), rgba(36,59,85,.96));
    border-right:1px solid rgba(255,255,255,.08);
    backdrop-filter:blur(20px);
}

.hero{
    padding:45px;
    border-radius:25px;
    background: linear-gradient(135deg, rgba(79,139,249,.22), rgba(30,41,59,.92));
    border:1px solid rgba(255,255,255,.10);
    backdrop-filter:blur(16px);
    box-shadow: 0 25px 60px rgba(0,0,0,.45);
    margin-bottom:30px;
}

.hero h1{
    font-size:52px;
    font-weight:800;
    color:white;
    margin-bottom:10px;
}

.hero p{
    font-size:18px;
    color:#d8e3ff;
}

.glass{
    background: rgba(255,255,255,.05);
    border-radius:22px;
    padding:22px;
    backdrop-filter:blur(18px);
    border: 1px solid rgba(255,255,255,.08);
    box-shadow: 0 15px 35px rgba(0,0,0,.35);
    transition:.35s;
}

.metric-card{
    background: linear-gradient(135deg, rgba(79,139,249,.18), rgba(255,255,255,.04));
    padding:20px;
    border-radius:18px;
    border:1px solid rgba(255,255,255,.08);
    text-align:center;
    transition:.35s;
}

.metric-card:hover{
    transform:scale(1.04);
}

.metric-title{
    font-size:15px;
    color:#CBD5E1;
}

.metric-value{
    font-size:34px;
    font-weight:700;
    color:white;
}

.stButton>button{
    width:100%;
    height:55px;
    border-radius:15px;
    border:none;
    font-weight:700;
    font-size:16px;
    background: linear-gradient(90deg, #4F8BF9, #2563EB);
    color:white;
    transition:.3s;
}

.stButton>button:hover{
    transform:translateY(-3px);
    box-shadow: 0 15px 30px rgba(79,139,249,.35);
}

div[data-baseweb="select"]>div{
    background:rgba(255,255,255,.05);
    border-radius:12px;
    border:1px solid rgba(255,255,255,.08);
}

input{
    background:rgba(255,255,255,.05)!important;
    color:white!important;
}

.footer{
    text-align:center;
    padding:20px;
    color:#94A3B8;
    font-size:14px;
}

hr{
    border:none;
    height:1px;
    background: rgba(255,255,255,.08);
    margin-top:25px;
    margin-bottom:25px;
}

.ai-card{
    background: linear-gradient(135deg, rgba(59,130,246,.12), rgba(255,255,255,.04));
    padding:28px;
    border-radius:20px;
    border-left:6px solid #3B82F6;
    box-shadow:0 10px 25px rgba(0,0,0,.30);
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# 3. MODEL LOADERS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_all():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(MODEL_COLUMNS_PATH)
    return model, scaler, columns

def validate():
    errors = []
    try: joblib.load(MODEL_PATH)
    except Exception as e: errors.append(f"Model Error: {e}")
    try: joblib.load(SCALER_PATH)
    except Exception as e: errors.append(f"Scaler Error: {e}")
    try: joblib.load(MODEL_COLUMNS_PATH)
    except Exception as e: errors.append(f"Columns Error: {e}")
    return errors


# =============================================================================
# 4. PREPROCESSING & PREDICTION ENGINE
# =============================================================================

def preprocess(customer_data, scaler, model_columns):
    df = pd.DataFrame([customer_data])

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in YES_NO_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(YES_NO_MAPPING)

    if "gender" in df.columns:
        df["gender"] = df["gender"].map(GENDER_MAPPING)

    df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], dtype=int)

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    df[NUMERIC_COLUMNS] = scaler.transform(df[NUMERIC_COLUMNS])
    return df

def get_risk_level(prob):
    if prob < 0.40: return "Low"
    elif prob < 0.70: return "Medium"
    return "High"

def recommendation(prob):
    if prob >= 0.85:
        return {"priority": "Critical", "action": "Immediate retention campaign, assign senior support, offer premium discount and contact within 24 hours."}
    elif prob >= 0.70:
        return {"priority": "High", "action": "Provide loyalty offers, personalized discounts and technical support follow-up."}
    elif prob >= 0.40:
        return {"priority": "Medium", "action": "Monitor customer behavior and send engagement campaigns."}
    return {"priority": "Low", "action": "Customer is stable. Continue standard engagement strategy."}

def predict_customer(customer_data, model, scaler, model_columns):
    X = preprocess(customer_data, scaler, model_columns)
    prediction = int(model.predict(X)[0])
    probability = float(model.predict_proba(X)[0][1])

    return {
        "prediction": prediction,
        "probability": probability,
        "probability_percent": round(probability * 100, 2),
        "risk": get_risk_level(probability),
        "confidence": round(max(probability, 1 - probability) * 100, 2),
        "status": "Likely to Churn" if prediction == 1 else "Likely to Stay",
        "recommendation": recommendation(probability),
        "financial_loss": round(probability * customer_data["MonthlyCharges"] * 12, 2)
    }


# =============================================================================
# 5. CHARTS & COMPONENTS
# =============================================================================

def update_layout(fig, title):
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=20)),
        template="plotly_dark",
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Poppins", color="white")
    )
    return fig

def gauge_chart(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=probability * 100, number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": PRIMARY},
            "steps": [
                {"range": [0, 40], "color": SUCCESS},
                {"range": [40, 70], "color": WARNING},
                {"range": [70, 100], "color": DANGER}
            ]
        }
    ))
    return update_layout(fig, "Churn Risk Gauge")

def probability_chart(probability):
    fig = px.pie(
        names=["Stay", "Churn"],
        values=[round((1 - probability) * 100, 2), round(probability * 100, 2)],
        hole=.65,
        color_discrete_sequence=[SUCCESS, DANGER]
    )
    return update_layout(fig, "Customer Probability")

def render_hero():
    st.markdown(f"""
    <div class="hero">
        <h1>🚀 {HERO_TITLE}</h1>
        <p>{HERO_SUBTITLE}</p>
        <br>
        <span style="background:#2563EB; padding:8px 18px; border-radius:20px; font-weight:600;">
            XGBoost Powered Intelligence
        </span>
    </div>
    """, unsafe_allow_html=True)

def render_kpis(res):
    col1, col2, col3, col4 = st.columns(4)
    items = [
        ("Churn Probability", f"{res['probability_percent']}%", "⚠️"),
        ("Risk Level", res["risk"], "🔥"),
        ("Confidence", f"{res['confidence']}%", "🎯"),
        ("Financial Risk", f"${res['financial_loss']}", "💰")
    ]
    for col, (title, val, icon) in zip([col1, col2, col3, col4], items):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:30px;">{icon}</div>
                <div class="metric-title">{title}</div>
                <div class="metric-value">{val}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================================
# 6. MAIN APPLICATION ENTRY POINT
# =============================================================================

st.set_page_config(page_title="TelcoGuard AI", page_icon="🚀", layout="wide")

load_css()

# Validate files availability
errors = validate()
if errors:
    st.error("⚠️ Model files not found! Make sure `models/` directory has `xgb_model.pkl`, `scaler.pkl`, and `model_columns.pkl`.")
    for err in errors: st.write(err)
    st.stop()

model, scaler, model_columns = load_all()

render_hero()

st.markdown("## 📋 Customer Profile Input")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Numeric Features")
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)
        gender = st.selectbox("Gender", CATEGORICAL_COLUMNS["gender"])

    with col2:
        st.subheader("Services & Subscriptions")
        internet_service = st.selectbox("Internet Service", CATEGORICAL_COLUMNS["InternetService"])
        contract = st.selectbox("Contract Type", CATEGORICAL_COLUMNS["Contract"])
        payment_method = st.selectbox("Payment Method", CATEGORICAL_COLUMNS["PaymentMethod"])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col3:
        st.subheader("Additional Services")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    submit_btn = st.form_submit_button("🔍 Predict Churn Risk")

if submit_btn:
    customer_data = {
        "tenure": tenure, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
        "gender": gender, "InternetService": internet_service, "Contract": contract,
        "PaymentMethod": payment_method, "Partner": partner, "Dependents": dependents,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "PaperlessBilling": paperless_billing
    }

    res = predict_customer(customer_data, model, scaler, model_columns)

    st.markdown("---")
    st.markdown("## 📊 Analysis & Insights")

    render_kpis(res)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(gauge_chart(res["probability"]), use_container_width=True)
    with c2:
        st.plotly_chart(probability_chart(res["probability"]), use_container_width=True)

    rec = res["recommendation"]
    st.markdown(f"""
    <div class="ai-card">
        <h3>🤖 AI Recommendation ({rec['priority']} Priority)</h3>
        <p style="font-size: 16px; margin-top: 10px;">{rec['action']}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"<div class='footer'>{FOOTER}</div>", unsafe_allow_html=True)
DEFAULT_RETENTION_COST = 20

DEFAULT_PROFIT_MARGIN = 0.35

