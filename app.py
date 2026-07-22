"""
===========================================================
TelcoGuard AI
Configuration File
===========================================================
Author : Mostafa Ahmed
Project: TelcoGuard AI
===========================================================
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

ASSETS_DIR = BASE_DIR / "assets"

REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


# =============================================================================
# MODEL FILES
# =============================================================================

MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

MODEL_COLUMNS_PATH = MODEL_DIR / "model_columns.pkl"


# =============================================================================
# PROJECT INFO
# =============================================================================

PROJECT_NAME = "TelcoGuard AI"

PROJECT_VERSION = "2.0"

AUTHOR = "Mostafa Ahmed"

COMPANY = "TelcoGuard Analytics"

MODEL_NAME = "XGBoost Classifier"

MODEL_TYPE = "Binary Classification"

TARGET = "Customer Churn"


# =============================================================================
# THEME COLORS
# =============================================================================

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


# =============================================================================
# RISK LEVELS
# =============================================================================

LOW_RISK = 0.40

MEDIUM_RISK = 0.70

HIGH_RISK = 1.00


# =============================================================================
# FINANCIAL SETTINGS
# =============================================================================

DEFAULT_CUSTOMER_VALUE = 1200

DEFAULT_RETENTION_COST = 120

DEFAULT_PROFIT_MARGIN = 0.35


# =============================================================================
# NUMERIC FEATURES
# =============================================================================

NUMERIC_COLUMNS = [

    "tenure",

    "MonthlyCharges",

    "TotalCharges"

]


# =============================================================================
# YES / NO FEATURES
# =============================================================================

YES_NO_COLUMNS = [

    "Partner",

    "Dependents",

    "PhoneService",

    "MultipleLines",

    "OnlineSecurity",

    "OnlineBackup",

    "DeviceProtection",

    "TechSupport",

    "StreamingTV",

    "StreamingMovies",

    "PaperlessBilling"

]


# =============================================================================
# CATEGORICAL FEATURES
# =============================================================================

CATEGORICAL_COLUMNS = {

    "gender": [

        "Male",

        "Female"

    ],

    "InternetService": [

        "DSL",

        "Fiber optic",

        "No"

    ],

    "Contract": [

        "Month-to-month",

        "One year",

        "Two year"

    ],

    "PaymentMethod": [

        "Electronic check",

        "Mailed check",

        "Bank transfer (automatic)",

        "Credit card (automatic)"

    ]

}


# =============================================================================
# KPI DEFAULT VALUES
# =============================================================================

DEFAULT_ACCURACY = 0.84

DEFAULT_PRECISION = 0.77

DEFAULT_RECALL = 0.73

DEFAULT_F1 = 0.75

DEFAULT_AUC = 0.88


# =============================================================================
# RISK LABELS
# =============================================================================

RISK_LABELS = {

    "Low": "#22C55E",

    "Medium": "#F59E0B",

    "High": "#EF4444"

}


# =============================================================================
# DASHBOARD TEXT
# =============================================================================

HERO_TITLE = "TelcoGuard AI"

HERO_SUBTITLE = (
    "Enterprise Customer Churn Intelligence Platform "
    "Powered by XGBoost & Explainable AI"
)


FOOTER = (
    "© 2026 TelcoGuard AI | Built with Streamlit + Plotly"
)
import streamlit as st


def load_css():

    st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

html,
body,
[class*="css"]{
    font-family:'Poppins',sans-serif;
}


/*=========================================================
MAIN
=========================================================*/

.stApp{

background:
linear-gradient(
135deg,
#07111f 0%,
#0f172a 30%,
#172554 100%
);

color:white;

}


/*=========================================================
HEADER
=========================================================*/

header{

visibility:hidden;

}

footer{

visibility:hidden;

}

#MainMenu{

visibility:hidden;

}


/*=========================================================
SIDEBAR
=========================================================*/

section[data-testid="stSidebar"]{

background:

linear-gradient(

180deg,

rgba(20,30,48,.95),

rgba(36,59,85,.96)

);

border-right:1px solid rgba(255,255,255,.08);

backdrop-filter:blur(20px);

}


/*=========================================================
HERO
=========================================================*/

.hero{

padding:45px;

border-radius:25px;

background:

linear-gradient(

135deg,

rgba(79,139,249,.22),

rgba(30,41,59,.92)

);

border:1px solid rgba(255,255,255,.10);

backdrop-filter:blur(16px);

box-shadow:

0 25px 60px rgba(0,0,0,.45);

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


/*=========================================================
GLASS CARD
=========================================================*/

.glass{

background:

rgba(255,255,255,.05);

border-radius:22px;

padding:22px;

backdrop-filter:blur(18px);

border:

1px solid rgba(255,255,255,.08);

box-shadow:

0 15px 35px rgba(0,0,0,.35);

transition:.35s;

}

.glass:hover{

transform:translateY(-6px);

box-shadow:

0 25px 45px rgba(0,0,0,.45);

}


/*=========================================================
METRIC CARD
=========================================================*/

.metric-card{

background:

linear-gradient(

135deg,

rgba(79,139,249,.18),

rgba(255,255,255,.04)

);

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


/*=========================================================
BUTTON
=========================================================*/

.stButton>button{

width:100%;

height:55px;

border-radius:15px;

border:none;

font-weight:700;

font-size:16px;

background:

linear-gradient(

90deg,

#4F8BF9,

#2563EB

);

color:white;

transition:.3s;

}

.stButton>button:hover{

transform:translateY(-3px);

box-shadow:

0 15px 30px rgba(79,139,249,.35);

}


/*=========================================================
INPUTS
=========================================================*/

.stSelectbox,

.stNumberInput,

.stSlider,

.stTextInput{

background:transparent;

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


/*=========================================================
CUSTOM TITLE
=========================================================*/

.title{

font-size:32px;

font-weight:700;

margin-bottom:10px;

}

.subtitle{

font-size:18px;

color:#94A3B8;

}


/*=========================================================
FOOTER
=========================================================*/

.footer{

text-align:center;

padding:20px;

color:#94A3B8;

font-size:14px;

}


/*=========================================================
SCROLLBAR
=========================================================*/

::-webkit-scrollbar{

width:10px;

}

::-webkit-scrollbar-track{

background:#0f172a;

}

::-webkit-scrollbar-thumb{

background:#2563EB;

border-radius:20px;

}


/*=========================================================
DIVIDER
=========================================================*/

hr{

border:none;

height:1px;

background:

rgba(255,255,255,.08);

margin-top:25px;

margin-bottom:25px;

}

</style>

""",unsafe_allow_html=True)
# =========================================================
# TABS
# =========================================================

.stTabs [data-baseweb="tab-list"]{

gap:12px;

background:rgba(255,255,255,.03);

padding:10px;

border-radius:16px;

}

.stTabs [data-baseweb="tab"]{

height:55px;

padding-left:22px;

padding-right:22px;

background:rgba(255,255,255,.04);

border-radius:12px;

color:white;

font-weight:600;

transition:.3s;

}

.stTabs [aria-selected="true"]{

background:

linear-gradient(
90deg,
#2563EB,
#4F8BF9
);

}


/*=========================================================
EXPANDER
=========================================================*/

.streamlit-expanderHeader{

font-size:17px;

font-weight:600;

color:white;

background:rgba(255,255,255,.04);

border-radius:12px;

}


/*=========================================================
TABLES
=========================================================*/

thead tr th{

background:#1E3A8A!important;

color:white!important;

font-size:15px;

}

tbody{

background:rgba(255,255,255,.03);

}


/*=========================================================
DATAFRAME
=========================================================*/

[data-testid="stDataFrame"]{

border-radius:18px;

overflow:hidden;

border:1px solid rgba(255,255,255,.08);

}


/*=========================================================
GAUGE CARD
=========================================================*/

.gauge-card{

padding:25px;

border-radius:20px;

background:

linear-gradient(
135deg,
rgba(37,99,235,.15),
rgba(255,255,255,.04)
);

border:1px solid rgba(255,255,255,.08);

}


/*=========================================================
RISK BADGE
=========================================================*/

.badge-low{

padding:8px 18px;

border-radius:25px;

background:#16A34A;

color:white;

font-weight:600;

display:inline-block;

}

.badge-medium{

padding:8px 18px;

border-radius:25px;

background:#F59E0B;

color:white;

font-weight:600;

display:inline-block;

}

.badge-high{

padding:8px 18px;

border-radius:25px;

background:#DC2626;

color:white;

font-weight:600;

display:inline-block;

}


/*=========================================================
AI CARD
=========================================================*/

.ai-card{

background:

linear-gradient(
135deg,
rgba(59,130,246,.12),
rgba(255,255,255,.04)
);

padding:28px;

border-radius:20px;

border-left:6px solid #3B82F6;

box-shadow:0 10px 25px rgba(0,0,0,.30);

}


/*=========================================================
TIMELINE
=========================================================*/

.timeline{

border-left:3px solid #3B82F6;

margin-left:18px;

padding-left:22px;

}

.timeline-item{

margin-bottom:24px;

position:relative;

}

.timeline-item::before{

content:"";

position:absolute;

left:-31px;

top:4px;

width:14px;

height:14px;

border-radius:50%;

background:#4F8BF9;

}


/*=========================================================
DOWNLOAD BUTTON
=========================================================*/

.download-btn{

display:inline-block;

padding:14px 26px;

border-radius:12px;

background:

linear-gradient(
90deg,
#2563EB,
#3B82F6
);

color:white;

font-weight:700;

text-decoration:none;

transition:.3s;

}

.download-btn:hover{

transform:translateY(-4px);

}


/*=========================================================
PROBABILITY CARD
=========================================================*/

.probability-card{

padding:24px;

border-radius:20px;

background:

linear-gradient(
135deg,
rgba(99,102,241,.15),
rgba(255,255,255,.05)
);

text-align:center;

}

.probability-value{

font-size:54px;

font-weight:800;

color:white;

}


/*=========================================================
CHART CONTAINER
=========================================================*/

.chart-card{

padding:20px;

background:rgba(255,255,255,.04);

border-radius:20px;

border:1px solid rgba(255,255,255,.08);

box-shadow:0 10px 25px rgba(0,0,0,.25);

}
"""
===========================================================
TelcoGuard AI
Model Loader
===========================================================
"""

import joblib
import streamlit as st

from config import (
    MODEL_PATH,
    SCALER_PATH,
    MODEL_COLUMNS_PATH
)


# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_model():
    """
    Load trained XGBoost model.
    """

    model = joblib.load(MODEL_PATH)

    return model


# ==========================================================
# LOAD SCALER
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_scaler():
    """
    Load StandardScaler.
    """

    scaler = joblib.load(SCALER_PATH)

    return scaler


# ==========================================================
# LOAD MODEL COLUMNS
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_columns():
    """
    Load training columns.
    """

    columns = joblib.load(MODEL_COLUMNS_PATH)

    return columns


# ==========================================================
# LOAD EVERYTHING
# ==========================================================

@st.cache_resource(show_spinner=False)
def load_all():

    model = load_model()

    scaler = load_scaler()

    columns = load_columns()

    return model, scaler, columns


# ==========================================================
# CHECK FILES
# ==========================================================

def validate():

    errors = []

    try:
        load_model()
    except Exception as e:
        errors.append(
            f"Model Error : {e}"
        )

    try:
        load_scaler()
    except Exception as e:
        errors.append(
            f"Scaler Error : {e}"
        )

    try:
        load_columns()
    except Exception as e:
        errors.append(
            f"Columns Error : {e}"
        )

    return errors
"""
===========================================================
TelcoGuard AI
Data Preprocessing
===========================================================
"""

import pandas as pd

from config import NUMERIC_COLUMNS


# ==========================================================
# YES / NO ENCODING
# ==========================================================

YES_NO_MAPPING = {
    "Yes": 1,
    "No": 0
}


# ==========================================================
# GENDER
# ==========================================================

GENDER_MAPPING = {
    "Male": 1,
    "Female": 0
}


# ==========================================================
# CLEAN INPUT
# ==========================================================

def clean_input(data: dict):

    df = pd.DataFrame([data])

    # -----------------------------------------

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = (
            pd.to_numeric(
                df["TotalCharges"],
                errors="coerce"
            )
            .fillna(0)
        )

    # -----------------------------------------

    for col in NUMERIC_COLUMNS:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    return df


# ==========================================================
# YES / NO
# ==========================================================

def encode_binary(df):

    binary_columns = [

        "Partner",

        "Dependents",

        "PhoneService",

        "MultipleLines",

        "OnlineSecurity",

        "OnlineBackup",

        "DeviceProtection",

        "TechSupport",

        "StreamingTV",

        "StreamingMovies",

        "PaperlessBilling"

    ]

    for col in binary_columns:

        if col in df.columns:

            df[col] = df[col].map(
                YES_NO_MAPPING
            )

    return df


# ==========================================================
# GENDER
# ==========================================================

def encode_gender(df):

    if "gender" in df.columns:

        df["gender"] = df["gender"].map(
            GENDER_MAPPING
        )

    return df


# ==========================================================
# ONE HOT
# ==========================================================

def one_hot_encode(df):

    categorical = [

        "InternetService",

        "Contract",

        "PaymentMethod"

    ]

    df = pd.get_dummies(

        df,

        columns=categorical,

        dtype=int

    )

    return df


# ==========================================================
# MATCH TRAINING COLUMNS
# ==========================================================

def align_columns(

        df,

        model_columns

):

    for col in model_columns:

        if col not in df.columns:

            df[col] = 0

    df = df[model_columns]

    return df


# ==========================================================
# SCALE
# ==========================================================

def scale_numeric(

        df,

        scaler

):

    df[NUMERIC_COLUMNS] = scaler.transform(

        df[NUMERIC_COLUMNS]

    )

    return df


# ==========================================================
# COMPLETE PIPELINE
# ==========================================================

def preprocess(

        customer_data,

        scaler,

        model_columns

):

    df = clean_input(

        customer_data

    )

    df = encode_binary(df)

    df = encode_gender(df)

    df = one_hot_encode(df)

    df = align_columns(

        df,

        model_columns

    )

    df = scale_numeric(

        df,

        scaler

    )

    return df
"""
===========================================================
TelcoGuard AI
Prediction Engine
===========================================================
"""

import numpy as np
import pandas as pd

from preprocessing import preprocess


# ==========================================================
# RISK LEVEL
# ==========================================================

def get_risk_level(probability):

    if probability < 0.40:
        return "Low"

    elif probability < 0.70:
        return "Medium"

    return "High"


# ==========================================================
# CONFIDENCE
# ==========================================================

def confidence_score(probability):

    probability = float(probability)

    confidence = max(
        probability,
        1 - probability
    )

    return round(confidence * 100, 2)


# ==========================================================
# CUSTOMER STATUS
# ==========================================================

def customer_status(prediction):

    if prediction == 1:
        return "Likely to Churn"

    return "Likely to Stay"


# ==========================================================
# AI RECOMMENDATION
# ==========================================================

def recommendation(probability):

    if probability >= 0.85:

        return {
            "priority": "Critical",
            "action":
            "Immediate retention campaign, assign senior support, "
            "offer premium discount and contact within 24 hours."
        }

    elif probability >= 0.70:

        return {
            "priority": "High",
            "action":
            "Provide loyalty offers, personalized discounts and "
            "technical support follow-up."
        }

    elif probability >= 0.40:

        return {
            "priority": "Medium",
            "action":
            "Monitor customer behavior and send engagement campaigns."
        }

    return {
        "priority": "Low",
        "action":
        "Customer is stable. Continue standard engagement strategy."
    }


# ==========================================================
# BUSINESS IMPACT
# ==========================================================

def business_impact(
        probability,
        monthly_charge):

    expected_loss = probability * monthly_charge * 12

    return round(expected_loss, 2)


# ==========================================================
# MAIN PREDICTION
# ==========================================================

def predict_customer(

        customer_data,

        model,

        scaler,

        model_columns

):

    X = preprocess(

        customer_data,

        scaler,

        model_columns

    )

    prediction = int(

        model.predict(X)[0]

    )

    probability = float(

        model.predict_proba(X)[0][1]

    )

    result = {

        "prediction": prediction,

        "probability": probability,

        "probability_percent":

            round(probability * 100, 2),

        "risk":

            get_risk_level(probability),

        "confidence":

            confidence_score(probability),

        "status":

            customer_status(prediction),

        "recommendation":

            recommendation(probability),

        "financial_loss":

            business_impact(

                probability,

                customer_data["MonthlyCharges"]

            )

    }

    return result


# ==========================================================
# BULK PREDICTION
# ==========================================================

def batch_predict(

        dataframe,

        model,

        scaler,

        model_columns

):

    results = []

    for _, row in dataframe.iterrows():

        prediction = predict_customer(

            row.to_dict(),

            model,

            scaler,

            model_columns

        )

        results.append(prediction)

    return pd.DataFrame(results)
"""
===========================================================
TelcoGuard AI
Professional Plotly Charts
===========================================================
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ==========================================================
# COLOR PALETTE
# ==========================================================

PRIMARY = "#4F8BF9"
SUCCESS = "#22C55E"
WARNING = "#F59E0B"
DANGER = "#EF4444"
CARD = "#1E293B"
BACKGROUND = "#0F172A"


# ==========================================================
# COMMON LAYOUT
# ==========================================================

def update_layout(fig, title):

    fig.update_layout(

        title=dict(
            text=title,
            x=0.5,
            font=dict(size=20)
        ),

        template="plotly_dark",

        paper_bgcolor=BACKGROUND,

        plot_bgcolor=BACKGROUND,

        margin=dict(
            l=20,
            r=20,
            t=50,
            b=20
        ),

        font=dict(
            family="Poppins",
            color="white"
        )

    )

    return fig


# ==========================================================
# GAUGE CHART
# ==========================================================

def gauge_chart(probability):

    value = probability * 100

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=value,

            number={
                "suffix": "%"
            },

            gauge={

                "axis": {

                    "range": [0, 100]

                },

                "bar": {

                    "color": PRIMARY

                },

                "steps": [

                    {

                        "range": [0, 40],

                        "color": SUCCESS

                    },

                    {

                        "range": [40, 70],

                        "color": WARNING

                    },

                    {

                        "range": [70, 100],

                        "color": DANGER

                    }

                ]

            }

        )

    )

    return update_layout(

        fig,

        "Churn Risk Gauge"

    )


# ==========================================================
# DONUT CHART
# ==========================================================

def probability_chart(probability):

    stay = round(

        (1 - probability) * 100,

        2

    )

    churn = round(

        probability * 100,

        2

    )

    fig = px.pie(

        names=[

            "Stay",

            "Churn"

        ],

        values=[

            stay,

            churn

        ],

        hole=.65,

        color_discrete_sequence=[

            SUCCESS,

            DANGER

        ]

    )

    return update_layout(

        fig,

        "Customer Probability"

    )


# ==========================================================
# RISK BAR
# ==========================================================

def risk_bar(probability):

    fig = go.Figure()

    fig.add_trace(

        go.Bar(

            x=["Risk"],

            y=[

                probability * 100

            ],

            marker_color=PRIMARY,

            width=[0.45]

        )

    )

    fig.update_yaxes(

        range=[0,100]

    )

    return update_layout(

        fig,

        "Risk Score"

    )


# ==========================================================
# FINANCIAL IMPACT
# ==========================================================

def financial_chart(loss):

    fig = go.Figure(

        go.Indicator(

            mode="number",

            value=loss,

            number={

                "prefix":"$",

                "valueformat":".2f"

            }

        )

    )

    return update_layout(

        fig,

        "Estimated Annual Revenue Loss"

    )


# ==========================================================
# KPI CARD
# ==========================================================

def kpi_card(

        title,

        value,

        color=PRIMARY

):

    fig = go.Figure()

    fig.add_annotation(

        text=f"<b>{value}</b>",

        x=.5,

        y=.6,

        showarrow=False,

        font=dict(

            size=38,

            color=color

        )

    )

    fig.add_annotation(

        text=title,

        x=.5,

        y=.28,

        showarrow=False,

        font=dict(

            size=17,

            color="white"

        )

    )

    fig.update_layout(

        template="plotly_dark",

        paper_bgcolor=CARD,

        plot_bgcolor=CARD,

        xaxis=dict(

            visible=False

        ),

        yaxis=dict(

            visible=False

        ),

        margin=dict(

            l=0,

            r=0,

            t=0,

            b=0

        )

    )

    return fig


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_importance_chart(df):

    fig = px.bar(

        df.head(10),

        x="Importance",

        y="Feature",

        orientation="h",

        color="Importance",

        color_continuous_scale="Blues"

    )

    fig.update_layout(

        yaxis=dict(

            autorange="reversed"

        )

    )

    return update_layout(

        fig,

        "Top Important Features"

    )


# ==========================================================
# CUSTOMER HEALTH
# ==========================================================

def customer_health(probability):

    health = (1-probability)*100

    fig = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=health,

            number={

                "suffix":"%"

            },

            gauge={

                "axis":{

                    "range":[0,100]

                },

                "bar":{

                    "color":SUCCESS

                }

            }

        )

    )

    return update_layout(

        fig,

        "Customer Health Score"

    )


# ==========================================================
# TIMELINE
# ==========================================================

def timeline():

    data = pd.DataFrame({

        "Stage":[

            "Customer Joined",

            "Usage Increased",

            "Risk Detected",

            "Retention Action",

            "Prediction"

        ],

        "Value":[

            1,

            2,

            3,

            4,

            5

        ]

    })

    fig = px.line(

        data,

        x="Stage",

        y="Value",

        markers=True

    )

    return update_layout(

        fig,

        "Customer Journey"

    )
"""
===========================================================
TelcoGuard AI
Dashboard Components
===========================================================
"""

import streamlit as st

from config import (
    PROJECT_NAME,
    HERO_TITLE,
    HERO_SUBTITLE
)


# ==========================================================
# HERO SECTION
# ==========================================================

def render_hero():

    st.markdown(
        f"""

        <div class="hero">

            <h1>
                🚀 {HERO_TITLE}
            </h1>


            <p>
                {HERO_SUBTITLE}
            </p>


            <br>

            <span style="
            background:#2563EB;
            padding:8px 18px;
            border-radius:20px;
            font-weight:600;
            ">

            XGBoost Powered Intelligence

            </span>

        </div>

        """,

        unsafe_allow_html=True

    )


# ==========================================================
# KPI CARD
# ==========================================================

def metric_card(

        title,

        value,

        icon="📊"

):

    st.markdown(

        f"""

        <div class="metric-card">

            <div style="
            font-size:30px;
            ">
            {icon}
            </div>


            <div class="metric-title">

            {title}

            </div>


            <div class="metric-value">

            {value}

            </div>


        </div>

        """,

        unsafe_allow_html=True

    )


# ==========================================================
# KPI ROW
# ==========================================================

def render_kpis(

        prediction_result

):

    col1,col2,col3,col4 = st.columns(4)


    probability = (
        prediction_result["probability_percent"]
    )


    with col1:

        metric_card(

            "Churn Probability",

            f"{probability}%",

            "⚠️"

        )


    with col2:

        metric_card(

            "Risk Level",

            prediction_result["risk"],

            "🔥"

        )


    with col3:

        metric_card(

            "Confidence",

            f"{prediction_result['confidence']}%",

            "🎯"

        )


    with col4:

        metric_card(

            "Financial Risk",

            f"${prediction_result['financial_loss']}",

            "💰"

        )


# ==========================================================
# RISK BADGE
# ==========================================================

def risk_badge(level):


    if level == "Low":

        css="badge-low"

        icon="🟢"


    elif level == "Medium":

        css="badge-medium"

        icon="🟡"


    else:

        css="badge-high"

        icon="🔴"



    st.markdown(

        f"""

        <span class="{css}">

        {icon} {level} Risk

        </span>

        """,

        unsafe_allow_html=True

    )


# ==========================================================
# CUSTOMER SUMMARY
# ==========================================================

def customer_summary(

        customer_data,

        prediction_result

):

    st.markdown(

        """

        <h2>

        👤 Customer Intelligence Summary

        </h2>

        """,

        unsafe_allow_html=True

    )


    col1,col2 = st.columns(2)



    with col1:

        st.markdown(

            f"""

            <div class="glass">


            <h4>
            Customer Profile
            </h4>


            <p>
            Tenure:
            <b>
            {customer_data.get('tenure')}
            </b>
            months
            </p>


            <p>
            Monthly Charges:
            <b>
            ${customer_data.get('MonthlyCharges')}
            </b>
            </p>


            <p>
            Contract:
            <b>
            {customer_data.get('Contract')}
            </b>
            </p>


            </div>

            """,

            unsafe_allow_html=True

        )



    with col2:

        st.markdown(

            f"""

            <div class="glass">


            <h4>
            AI Prediction
            </h4>


            <p>
            Status:
            <b>
            {prediction_result['status']}
            </b>
            </p>


            <p>
            Risk:
            </p>

            """,

            unsafe_allow_html=True

        )


        risk_badge(

            prediction_result["risk"]

        )


        st.markdown(

            "</div>",

            unsafe_allow_html=True

        )


# ==========================================================
# AI RECOMMENDATION
# ==========================================================

def recommendation_card(

        prediction_result

):

    rec = prediction_result["recommendation"]


    st.markdown(

        f"""

        <div class="ai-card">


        <h2>
        🤖 AI Retention Recommendation
        </h2>


        <h4>

        Priority:

        <span style="
        color:#38BDF8;
        ">

        {rec['priority']}

        </span>

        </h4>


        <p>

        {rec['action']}

        </p>


        </div>


        """,

        unsafe_allow_html=True

    )
# ==========================================================
# EXECUTIVE DASHBOARD HEADER
# ==========================================================

def section_title(title, subtitle=None):

    st.markdown(

        f"""

        <div class="title">

        {title}

        </div>

        """,

        unsafe_allow_html=True

    )


    if subtitle:

        st.markdown(

            f"""

            <div class="subtitle">

            {subtitle}

            </div>

            """,

            unsafe_allow_html=True

        )



# ==========================================================
# CHART CONTAINER
# ==========================================================

def chart_container(

        title,

        chart

):

    st.markdown(

        f"""

        <div class="chart-card">

        <h3>

        {title}

        </h3>

        </div>

        """,

        unsafe_allow_html=True

    )


    st.plotly_chart(

        chart,

        use_container_width=True

    )



# ==========================================================
# RISK ANALYSIS SECTION
# ==========================================================

def risk_analysis(

        prediction_result,

        gauge,

        probability_chart,

        health_chart

):

    section_title(

        "🎯 AI Risk Intelligence",

        "Advanced churn probability analysis"

    )


    col1,col2 = st.columns(2)


    with col1:

        st.plotly_chart(

            gauge,

            use_container_width=True

        )


    with col2:

        st.plotly_chart(

            probability_chart,

            use_container_width=True

        )


    st.plotly_chart(

        health_chart,

        use_container_width=True

    )



# ==========================================================
# FINANCIAL IMPACT
# ==========================================================

def financial_card(

        loss

):


    st.markdown(

        f"""

        <div class="glass">


        <h2>

        💰 Financial Impact

        </h2>


        <h1 style="
        color:#EF4444;
        ">

        ${loss}

        </h1>


        <p>

        Estimated yearly revenue at risk

        </p>


        </div>

        """,

        unsafe_allow_html=True

    )



# ==========================================================
# TIMELINE COMPONENT
# ==========================================================

def render_timeline():

    section_title(

        "📅 Customer Journey Timeline"

    )


    steps = [

        (
            "🟢",
            "Customer Joined",
            "Initial subscription created"
        ),

        (
            "🔵",
            "Usage Monitoring",
            "Behavior analysis started"
        ),

        (
            "🟡",
            "Risk Detection",
            "AI detected possible churn"
        ),

        (
            "🟣",
            "Retention Action",
            "Personalized offer generated"
        ),

        (
            "🔴",
            "Follow Up",
            "Customer success team contact"
        )

    ]


    html=""


    for icon,title,text in steps:


        html += f"""

        <div class="timeline-item">

        <h4>

        {icon} {title}

        </h4>


        <p>

        {text}

        </p>


        </div>

        """


    st.markdown(

        f"""

        <div class="timeline">

        {html}

        </div>

        """,

        unsafe_allow_html=True

    )



# ==========================================================
# WHAT IF SIMULATOR INPUTS
# ==========================================================

def scenario_simulator():

    section_title(

        "🧪 What-If Scenario Simulator",

        "Test how customer changes affect churn probability"

    )


    col1,col2,col3 = st.columns(3)



    with col1:

        contract = st.selectbox(

            "Contract",

            [

                "Month-to-month",

                "One year",

                "Two year"

            ],

            key="scenario_contract"

        )


    with col2:

        monthly = st.slider(

            "Monthly Charges",

            10,

            150,

            70,

            key="scenario_monthly"

        )


    with col3:

        tenure = st.slider(

            "Tenure",

            0,

            80,

            20,

            key="scenario_tenure"

        )


    return {

        "Contract": contract,

        "MonthlyCharges": monthly,

        "tenure": tenure

    }



# ==========================================================
# DOWNLOAD CENTER
# ==========================================================

def download_center(

        report_file

):

    section_title(

        "📥 Report Center"

    )


    with open(

        report_file,

        "rb"

    ) as file:


        st.download_button(

            label="Download Customer Report",

            data=file,

            file_name="TelcoGuard_Report.pdf",

            mime="application/pdf"

        )



# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_section(

        chart

):

    section_title(

        "🔍 Explainable AI"

    )


    st.markdown(

        """

        <div class="glass">

        Model explanation using feature contribution analysis.

        </div>

        """,

        unsafe_allow_html=True

    )


    st.plotly_chart(

        chart,

        use_container_width=True

    )



# ==========================================================
# FOOTER
# ==========================================================

def footer():

    st.markdown(

        """

        <div class="footer">

        🚀 TelcoGuard AI | Customer Churn Intelligence Platform

        </div>

        """,

        unsafe_allow_html=True

    )
"""
===========================================================
TelcoGuard AI
Financial Intelligence Engine
===========================================================
"""


from config import (
    DEFAULT_CUSTOMER_VALUE,
    DEFAULT_RETENTION_COST,
    DEFAULT_PROFIT_MARGIN
)



# ==========================================================
# CUSTOMER LIFETIME VALUE
# ==========================================================

def calculate_clv(

        monthly_charge,

        months=24

):

    """
    Estimate Customer Lifetime Value
    """

    value = (

        monthly_charge

        *

        months

    )


    return round(value,2)



# ==========================================================
# CHURN LOSS
# ==========================================================

def calculate_churn_loss(

        probability,

        monthly_charge

):

    """

    Expected revenue loss

    """

    yearly_value = (

        monthly_charge

        *

        12

    )


    loss = (

        probability

        *

        yearly_value

    )


    return round(loss,2)



# ==========================================================
# RETENTION COST
# ==========================================================

def retention_cost(

        customers=1

):


    return (

        DEFAULT_RETENTION_COST

        *

        customers

    )



# ==========================================================
# SAVING OPPORTUNITY
# ==========================================================

def saving_opportunity(

        probability,

        monthly_charge

):


    loss = calculate_churn_loss(

        probability,

        monthly_charge

    )


    opportunity = loss * 0.75


    return round(opportunity,2)



# ==========================================================
# ROI CALCULATION
# ==========================================================

def calculate_roi(

        saved_value,

        campaign_cost

):


    if campaign_cost == 0:

        return 0


    roi = (

        (saved_value - campaign_cost)

        /

        campaign_cost

    ) * 100


    return round(roi,2)



# ==========================================================
# PROFIT IMPACT
# ==========================================================

def profit_impact(

        revenue

):


    profit = (

        revenue

        *

        DEFAULT_PROFIT_MARGIN

    )


    return round(profit,2)



# ==========================================================
# COMPLETE FINANCIAL REPORT
# ==========================================================

def financial_report(

        probability,

        monthly_charge

):


    clv = calculate_clv(

        monthly_charge

    )


    loss = calculate_churn_loss(

        probability,

        monthly_charge

    )


    saving = saving_opportunity(

        probability,

        monthly_charge

    )


    roi = calculate_roi(

        saving,

        DEFAULT_RETENTION_COST

    )


    return {


        "customer_lifetime_value":

            clv,


        "expected_loss":

            loss,


        "saving_opportunity":

            saving,


        "retention_roi":

            roi,


        "profit_impact":

            profit_impact(loss)

    }
"""
===========================================================
TelcoGuard AI
Explainable AI Recommendation Engine
===========================================================
"""


# ==========================================================
# RISK EXPLANATION
# ==========================================================

def analyze_risk_factors(customer):

    factors = []


    # Contract Analysis

    if customer.get("Contract") == "Month-to-month":

        factors.append({

            "factor":
                "Monthly Contract",

            "impact":
                "High",

            "reason":
                "Monthly contracts have higher churn probability."

        })


    # Tenure Analysis

    tenure = customer.get(
        "tenure",
        0
    )


    if tenure < 12:

        factors.append({

            "factor":
                "New Customer",

            "impact":
                "Medium",

            "reason":
                "Customers in early lifecycle need stronger engagement."

        })


    # Charges Analysis

    monthly = customer.get(

        "MonthlyCharges",

        0

    )


    if monthly > 80:

        factors.append({

            "factor":
                "High Monthly Charges",

            "impact":
                "High",

            "reason":
                "High pricing may increase customer dissatisfaction."

        })


    # Support

    if customer.get("TechSupport") == "No":

        factors.append({

            "factor":
                "No Technical Support",

            "impact":
                "Medium",

            "reason":
                "Lack of support increases churn risk."

        })


    return factors



# ==========================================================
# RECOMMENDATION GENERATOR
# ==========================================================

def generate_recommendation(

        probability,

        factors

):


    if probability >= 0.85:

        priority = "Critical"

        action = [

            "Contact customer immediately",

            "Offer personalized retention discount",

            "Assign customer success manager"

        ]


    elif probability >= 0.70:


        priority = "High"


        action = [

            "Launch loyalty campaign",

            "Provide service upgrade",

            "Monitor customer activity"

        ]


    elif probability >= 0.40:


        priority = "Medium"


        action = [

            "Send engagement offers",

            "Improve communication"

        ]


    else:


        priority = "Low"


        action = [

            "Maintain normal relationship",

            "Continue monitoring"

        ]


    return {


        "priority":

            priority,


        "actions":

            action,


        "risk_factors":

            factors

    }



# ==========================================================
# EXECUTIVE SUMMARY
# ==========================================================

def executive_summary(

        customer,

        probability,

        factors

):


    risk = (

        "High Risk"

        if probability >= 0.7

        else

        "Moderate Risk"

        if probability >= 0.4

        else

        "Low Risk"

    )


    summary = f"""

Customer is classified as:

{risk}


The AI model estimates churn probability of:

{round(probability*100,2)}%


Main detected risk factors:

"""


    for factor in factors:

        summary += (

            f"- {factor['factor']} "

            f"({factor['impact']} Impact)\n"

        )


    return summary



# ==========================================================
# COMPLETE AI ANALYSIS
# ==========================================================

def generate_ai_analysis(

        customer,

        probability

):


    factors = analyze_risk_factors(

        customer

    )


    recommendation = generate_recommendation(

        probability,

        factors

    )


    summary = executive_summary(

        customer,

        probability,

        factors

    )


    return {


        "summary":

            summary,


        "recommendation":

            recommendation,


        "factors":

            factors

    }
"""
===========================================================
TelcoGuard AI
Scenario Simulator Engine
===========================================================
"""


from predictor import predict_customer



# ==========================================================
# APPLY CUSTOMER CHANGES
# ==========================================================

def apply_changes(

        customer,

        changes

):

    updated_customer = customer.copy()


    for key, value in changes.items():

        updated_customer[key] = value


    return updated_customer



# ==========================================================
# RUN SCENARIO
# ==========================================================

def run_scenario(

        original_customer,

        changes,

        model,

        scaler,

        columns

):


    modified_customer = apply_changes(

        original_customer,

        changes

    )


    result = predict_customer(

        modified_customer,

        model,

        scaler,

        columns

    )


    return {


        "customer":

            modified_customer,


        "result":

            result

    }



# ==========================================================
# COMPARE RESULTS
# ==========================================================

def compare_results(

        original_result,

        scenario_result

):


    old_probability = (

        original_result["probability"]

    )


    new_probability = (

        scenario_result["probability"]

    )


    improvement = (

        old_probability

        -

        new_probability

    )


    improvement_percent = (

        improvement * 100

    )


    if improvement > 0:

        status = "Improved"

        message = (

            "Scenario reduced churn probability."

        )


    elif improvement < 0:

        status = "Worse"

        message = (

            "Scenario increased churn probability."

        )


    else:

        status = "No Change"

        message = (

            "Scenario did not affect prediction."

        )


    return {


        "old_probability":

            round(old_probability*100,2),


        "new_probability":

            round(new_probability*100,2),


        "improvement":

            round(improvement_percent,2),


        "status":

            status,


        "message":

            message

    }



# ==========================================================
# RETENTION SCENARIOS
# ==========================================================

def recommended_scenarios(customer):


    scenarios = []


    # Contract Upgrade

    if customer.get("Contract") == "Month-to-month":


        scenarios.append({

            "name":

                "Upgrade Contract",


            "changes":

                {

                "Contract":

                    "One year"

                }

        })



    # Lower Price Scenario

    if customer.get("MonthlyCharges",0) > 70:


        scenarios.append({

            "name":

                "Discount Offer",


            "changes":

                {

                "MonthlyCharges":

                    customer["MonthlyCharges"]*0.85

                }

        })



    # Support Scenario

    if customer.get("TechSupport") == "No":


        scenarios.append({

            "name":

                "Add Technical Support",


            "changes":

                {

                "TechSupport":

                    "Yes"

                }

        })


    return scenarios



# ==========================================================
# RUN ALL SCENARIOS
# ==========================================================

def evaluate_scenarios(

        customer,

        model,

        scaler,

        columns

):


    scenarios = recommended_scenarios(

        customer

    )


    results = []


    for scenario in scenarios:


        output = run_scenario(

            customer,

            scenario["changes"],

            model,

            scaler,

            columns

        )


        results.append({

            "scenario":

                scenario["name"],


            "result":

                output["result"]

        })


    return results
"""
===========================================================
TelcoGuard AI
Report Generator
===========================================================
"""

from datetime import datetime

from pathlib import Path

from config import REPORT_DIR, PROJECT_NAME



# ==========================================================
# HTML STYLE
# ==========================================================

REPORT_STYLE = """

<style>

body{

font-family:Arial, sans-serif;

background:#0f172a;

color:white;

padding:40px;

}


.container{

background:#1e293b;

padding:30px;

border-radius:20px;

}


.title{

font-size:35px;

font-weight:bold;

color:#4f8bf9;

}


.card{

background:#334155;

padding:20px;

border-radius:15px;

margin:15px 0;

}


.risk{

font-size:25px;

font-weight:bold;

}


table{

width:100%;

border-collapse:collapse;

}


td,th{

padding:12px;

border-bottom:1px solid #475569;

}


</style>

"""



# ==========================================================
# CREATE REPORT
# ==========================================================

def create_report(

        customer,

        prediction,

        ai_analysis,

        financial

):


    timestamp = datetime.now().strftime(

        "%Y-%m-%d %H:%M"

    )


    risk_color = (

        "#22c55e"

        if prediction["risk"]=="Low"

        else

        "#f59e0b"

        if prediction["risk"]=="Medium"

        else

        "#ef4444"

    )



    html = f"""

<html>

<head>

{REPORT_STYLE}

</head>


<body>


<div class="container">


<div class="title">

🚀 {PROJECT_NAME}

</div>


<p>

Generated:

{timestamp}

</p>



<div class="card">


<h2>

Customer Prediction

</h2>


<table>


<tr>

<td>Status</td>

<td>

{prediction['status']}

</td>

</tr>



<tr>

<td>

Churn Probability

</td>


<td>

{prediction['probability_percent']}%

</td>

</tr>



<tr>

<td>

Confidence

</td>


<td>

{prediction['confidence']}%

</td>

</tr>



<tr>

<td>

Risk

</td>


<td style="color:{risk_color}">

{prediction['risk']}

</td>


</tr>


</table>


</div>





<div class="card">


<h2>

🤖 AI Recommendation

</h2>


<p>

{ai_analysis['summary']}

</p>


</div>





<div class="card">


<h2>

💰 Financial Impact

</h2>


<table>


<tr>

<td>

Expected Loss

</td>


<td>

${financial['expected_loss']}

</td>


</tr>



<tr>

<td>

Saving Opportunity

</td>


<td>

${financial['saving_opportunity']}

</td>


</tr>



<tr>

<td>

Retention ROI

</td>


<td>

{financial['retention_roi']}%

</td>


</tr>


</table>


</div>




<div class="card">


<h2>

Risk Factors

</h2>


<ul>

"""


    for factor in ai_analysis["factors"]:


        html += f"""

<li>

<b>

{factor['factor']}

</b>

-

{factor['reason']}

</li>

"""


    html += """

</ul>


</div>



</div>


</body>


</html>

"""



    file_path = (

        Path(REPORT_DIR)

        /

        "TelcoGuard_Report.html"

    )


    with open(

        file_path,

        "w",

        encoding="utf-8"

    ) as file:


        file.write(html)



    return file_path
"""
===========================================================
TelcoGuard AI
Main Streamlit Application
===========================================================
"""


import streamlit as st


# ==========================================================
# INTERNAL MODULES
# ==========================================================

from styles import load_css


from loader import load_all


from predictor import predict_customer


from components import (

    render_hero,

    render_kpis,

    customer_summary,

    recommendation_card,

    scenario_simulator,

    render_timeline,

    footer

)


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================


st.set_page_config(

    page_title="TelcoGuard AI",

    page_icon="🚀",

    layout="wide",

    initial_sidebar_state="expanded"

)



# ==========================================================
# LOAD STYLE
# ==========================================================


load_css()



# ==========================================================
# LOAD MODEL RESOURCES
# ==========================================================


@st.cache_resource

def initialize():

    return load_all()



model, scaler, model_columns = initialize()



# ==========================================================
# HERO
# ==========================================================


render_hero()



# ==========================================================
# SIDEBAR
# ==========================================================


with st.sidebar:


    st.markdown(

        """

        ## 🚀 TelcoGuard AI


        ### Customer Churn Prediction


        Powered by:

        **XGBoost + Explainable AI**


        ---

        """

    )


    st.markdown(

        """

        ### Navigation


        Select customer data below


        """

    )



# ==========================================================
# CUSTOMER INPUT SECTION
# ==========================================================


st.markdown(

    """

    <div class="title">

    👤 Customer Information

    </div>


    <div class="subtitle">

    Enter customer details to predict churn risk

    </div>

    """,

    unsafe_allow_html=True

)



# ==========================================================
# INPUT FORM
# ==========================================================


col1, col2, col3 = st.columns(3)



with col1:


    gender = st.selectbox(

        "Gender",

        [

            "Male",

            "Female"

        ]

    )


    tenure = st.number_input(

        "Tenure (Months)",

        min_value=0,

        max_value=100,

        value=12

    )


    Partner = st.selectbox(

        "Partner",

        [

            "Yes",

            "No"

        ]

    )


    Dependents = st.selectbox(

        "Dependents",

        [

            "Yes",

            "No"

        ]

    )



with col2:


    MonthlyCharges = st.number_input(

        "Monthly Charges",

        min_value=0.0,

        value=70.0

    )


    TotalCharges = st.number_input(

        "Total Charges",

        min_value=0.0,

        value=800.0

    )


    Contract = st.selectbox(

        "Contract",

        [

            "Month-to-month",

            "One year",

            "Two year"

        ]

    )



    InternetService = st.selectbox(

        "Internet Service",

        [

            "DSL",

            "Fiber optic",

            "No"

        ]

    )



with col3:


    TechSupport = st.selectbox(

        "Tech Support",

        [

            "Yes",

            "No"

        ]

    )


    OnlineSecurity = st.selectbox(

        "Online Security",

        [

            "Yes",

            "No"

        ]

    )


    PaperlessBilling = st.selectbox(

        "Paperless Billing",

        [

            "Yes",

            "No"

        ]

    )


    PaymentMethod = st.selectbox(

        "Payment Method",

        [

            "Electronic check",

            "Mailed check",

            "Bank transfer (automatic)",

            "Credit card (automatic)"

        ]

    )



# ==========================================================
# CUSTOMER DICTIONARY
# ==========================================================


customer = {


    "gender":

        gender,


    "tenure":

        tenure,


    "Partner":

        Partner,


    "Dependents":

        Dependents,


    "MonthlyCharges":

        MonthlyCharges,


    "TotalCharges":

        TotalCharges,


    "Contract":

        Contract,


    "InternetService":

        InternetService,


    "TechSupport":

        TechSupport,


    "OnlineSecurity":

        OnlineSecurity,


    "PaperlessBilling":

        PaperlessBilling,


    "PaymentMethod":

        PaymentMethod

}
# ==========================================================
# PREDICTION BUTTON
# ==========================================================


st.divider()


predict_button = st.button(

    "🚀 Analyze Customer Churn Risk"

)



# ==========================================================
# RUN MODEL
# ==========================================================


if predict_button:


    with st.spinner(

        "AI model is analyzing customer behavior..."

    ):


        result = predict_customer(

            customer,

            model,

            scaler,

            model_columns

        )


        st.session_state["prediction"] = result


        st.session_state["customer"] = customer



# ==========================================================
# DISPLAY RESULTS
# ==========================================================


if "prediction" in st.session_state:


    prediction_result = st.session_state["prediction"]


    customer_data = st.session_state["customer"]



    st.divider()



    # ======================================================
    # KPI SECTION
    # ======================================================


    render_kpis(

        prediction_result

    )



    st.divider()



    # ======================================================
    # DASHBOARD TABS
    # ======================================================


    tab1, tab2, tab3, tab4, tab5 = st.tabs(

        [

            "📊 Overview",

            "🎯 Risk Analysis",

            "🤖 AI Insights",

            "💰 Financial Impact",

            "🔍 Explainable AI"

        ]

    )



    # ======================================================
    # TAB 1 OVERVIEW
    # ======================================================


    with tab1:


        customer_summary(

            customer_data,

            prediction_result

        )



        st.write("")



        recommendation_card(

            prediction_result

        )



        render_timeline()



    # ======================================================
    # TAB 2 RISK ANALYSIS
    # ======================================================


    with tab2:


        from charts import (

            gauge_chart,

            probability_chart,

            customer_health

        )



        st.markdown(

            """

            ### Churn Probability Analysis

            """

        )


        col1,col2 = st.columns(2)



        with col1:


            st.plotly_chart(

                gauge_chart(

                    prediction_result["probability"]

                ),

                use_container_width=True

            )



        with col2:


            st.plotly_chart(

                probability_chart(

                    prediction_result["probability"]

                ),

                use_container_width=True

            )



        st.plotly_chart(

            customer_health(

                prediction_result["probability"]

            ),

            use_container_width=True

        )



    # ======================================================
    # TAB 3 AI INSIGHTS
    # ======================================================


    with tab3:


        from ai_engine import generate_ai_analysis



        ai_result = generate_ai_analysis(

            customer_data,

            prediction_result["probability"]

        )


        st.markdown(

            """

            ## 🤖 AI Executive Summary

            """

        )


        st.info(

            ai_result["summary"]

        )


        st.markdown(

            """

            ## Recommended Actions

            """

        )


        for action in ai_result["recommendation"]["actions"]:


            st.success(

                "✔ " + action

            )



        st.markdown(

            """

            ## Detected Risk Factors

            """

        )


        for factor in ai_result["factors"]:


            st.warning(

                f"""

                **{factor['factor']}**

               

                {factor['reason']}

                """

            )



    # ======================================================
    # TAB 4 FINANCIAL IMPACT
    # ======================================================


    with tab4:


        from financial import financial_report



        financial = financial_report(

            prediction_result["probability"],

            customer_data["MonthlyCharges"]

        )


        col1,col2,col3 = st.columns(3)



        with col1:


            st.metric(

                "Customer Lifetime Value",

                f"${financial['customer_lifetime_value']}"

            )



        with col2:


            st.metric(

                "Expected Loss",

                f"${financial['expected_loss']}"

            )



        with col3:


            st.metric(

                "Retention ROI",

                f"{financial['retention_roi']}%"

            )



        st.markdown(

            """

            ### 💡 Business Opportunity

            """

        )


        st.success(

            f"""

            Saving Opportunity:

            ${financial['saving_opportunity']}

            """

        )



    # ======================================================
    # TAB 5 EXPLAINABLE AI
    # ======================================================


    with tab5:


        st.markdown(

            """

            ## 🔍 Model Explanation

            """

        )


        st.write(

            """

            The model analyzes customer behavior,

            contract information,

            payment patterns,

            and service usage

            to estimate churn probability.

            """

        )



        st.progress(

            prediction_result["probability"]

        )
# ==========================================================
# SCENARIO SIMULATOR
# ==========================================================


with st.expander(

    "🧪 What-If Analysis & Scenario Simulator"

):


    scenario_changes = scenario_simulator()



    if st.button(

        "Run Scenario Simulation"

    ):


        from simulator import (

            run_scenario,

            compare_results

        )


        scenario_output = run_scenario(

            customer_data,

            scenario_changes,

            model,

            scaler,

            model_columns

        )


        comparison = compare_results(

            prediction_result,

            scenario_output["result"]

        )



        st.subheader(

            "Scenario Result"

        )



        col1,col2,col3 = st.columns(3)



        with col1:


            st.metric(

                "Original Risk",

                f"{comparison['old_probability']}%"

            )



        with col2:


            st.metric(

                "New Risk",

                f"{comparison['new_probability']}%"

            )



        with col3:


            st.metric(

                "Improvement",

                f"{comparison['improvement']}%"

            )



        if comparison["status"] == "Improved":


            st.success(

                comparison["message"]

            )


        elif comparison["status"] == "Worse":


            st.error(

                comparison["message"]

            )


        else:


            st.info(

                comparison["message"]

            )



# ==========================================================
# REPORT GENERATION
# ==========================================================


st.divider()



st.markdown(

    """

    ## 📥 Download Center

    Generate a complete AI customer report

    """

)



if st.button(

    "Generate Customer Report"

):


    from report import create_report


    from ai_engine import generate_ai_analysis


    from financial import financial_report



    ai_result = generate_ai_analysis(

        customer_data,

        prediction_result["probability"]

    )



    financial = financial_report(

        prediction_result["probability"],

        customer_data["MonthlyCharges"]

    )



    report_path = create_report(

        customer_data,

        prediction_result,

        ai_result,

        financial

    )



    st.session_state["report"] = report_path



if "report" in st.session_state:


    with open(

        st.session_state["report"],

        "rb"

    ) as file:


        st.download_button(

            label="⬇ Download TelcoGuard Report",

            data=file,

            file_name="TelcoGuard_AI_Report.html",

            mime="text/html"

        )



# ==========================================================
# FOOTER
# ==========================================================


footer()
from pathlib import Path


BASE_DIR = Path(__file__).parent


MODEL_DIR = BASE_DIR / "models"


MODEL_PATH = MODEL_DIR / "xgb_model.pkl"

SCALER_PATH = MODEL_DIR / "scaler.pkl"

MODEL_COLUMNS_PATH = MODEL_DIR / "model_columns.pkl"



REPORT_DIR = BASE_DIR / "reports"


PROJECT_NAME = "TelcoGuard AI"


HERO_TITLE = "AI Customer Churn Intelligence"


HERO_SUBTITLE = (
    "Predict churn risk, understand customer behavior, "
    "and generate retention strategies."
)


DEFAULT_RETENTION_COST = 20

DEFAULT_PROFIT_MARGIN = 0.35

