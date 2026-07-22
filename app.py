import streamlit as st
import os
import pandas as pd
import plotly.graph_objects as go
from model_utils import load_model, predict_churn
import numpy as np
# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="TelcoGuard AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');


* {
    font-family: 'Cairo', sans-serif;
}


html, body, [class*="css"] {
    direction: rtl;
}


.stApp {

    background:
    radial-gradient(
        circle at top right,
        #16213e,
        #0b0f19 45%
    );

}


/* Sidebar */

section[data-testid="stSidebar"] {

    background:
    linear-gradient(
        180deg,
        #111827,
        #0b1220
    );

    border-left:
    1px solid rgba(255,255,255,0.08);

}



/* Main Header */

.hero {

    background:
    linear-gradient(
        135deg,
        #0f4c81,
        #071426
    );

    padding:40px;

    border-radius:25px;

    text-align:center;

    margin-bottom:30px;

    box-shadow:
    0 20px 40px rgba(0,0,0,.35);

}


.hero h1 {

    color:white;

    font-size:3rem;

    font-weight:900;

}


.hero p {

    color:#b8c7d9;

    font-size:1.2rem;

}



/* Cards */


.card {


background:
linear-gradient(
145deg,
#182233,
#10151f
);


padding:25px;

border-radius:20px;

text-align:center;

border:
1px solid rgba(255,255,255,.08);


box-shadow:
0 10px 25px rgba(0,0,0,.3);


}


.card-title {

color:#9ca3af;

font-size:1rem;

}


.card-value {


font-size:2.3rem;

font-weight:900;

margin-top:10px;


}



/* Section */

.section-title {


font-size:1.5rem;

font-weight:800;

color:white;

border-right:
5px solid #00c6ff;

padding-right:12px;

margin:30px 0 20px;


}



/* Buttons */


.stButton button {


width:100%;

background:
linear-gradient(
90deg,
#0072ff,
#00c6ff
);


color:white;

border:none;

border-radius:12px;

padding:12px;

font-weight:700;

}



.stButton button:hover {


box-shadow:
0 0 20px #00c6ff;


}



/* Tabs */


button[data-baseweb="tab"] {


font-size:1.1rem;

font-weight:700;

}



</style>

""",
unsafe_allow_html=True)



# =========================================================
# HEADER
# =========================================================


st.markdown("""

<div class="hero">

<h1>
📡 TelcoGuard AI
</h1>

<p>
Intelligent Telecom Customer Churn Prediction Platform
<br>
Powered by XGBoost & Machine Learning
</p>


</div>


""",
unsafe_allow_html=True)



# =========================================================
# SIDEBAR
# =========================================================


with st.sidebar:


    st.markdown(

    """

    <h1 style="
    color:#00c6ff;
    text-align:center;
    ">
    TelcoGuard
    </h1>

    """,

    unsafe_allow_html=True
    )


    st.markdown("---")


    st.markdown(
    """
    ### ⚙️ System Status
    """
    )


    model_path = "models/xgboost.pkl"


    if os.path.exists(model_path):

        st.success(
            "Model Loaded"
        )

    else:

        st.warning(
            "Model Not Found"
        )



    st.markdown("---")


    st.markdown(
    """

    ### 🧠 About

    This system predicts:

    - Customer churn probability
    - Risk level
    - Retention recommendations
    - Business analytics


    """

    )

# =========================================================
# LOAD MODEL
# =========================================================

model = load_model()


if model is not None:

    st.sidebar.success(
        "🟢 XGBoost Model Ready"
    )

else:

    st.sidebar.error(
        "🔴 Model Not Found"
    )

# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():

    files = [

        "data/WA_Fn-UseC_-Telco-Customer-Churn.csv",

        "WA_Fn-UseC_-Telco-Customer-Churn.csv",

        "data/telco.csv"

    ]


    for file in files:

        if os.path.exists(file):

            df = pd.read_csv(file)


            if "TotalCharges" in df.columns:

                df["TotalCharges"] = pd.to_numeric(
                    df["TotalCharges"],
                    errors="coerce"
                )


                df.dropna(
                    inplace=True
                )


            return df


    return None


# =========================================================
# AI RECOMMENDATION ENGINE
# =========================================================

def generate_recommendations(customer, probability):


    recommendations=[]


    if probability >=0.5:


        if customer["Contract"]=="Month-to-month":

            recommendations.append(
                "🔄 تحويل العميل من عقد شهري إلى عقد سنوي أو سنتين"
            )


        if customer["TechSupport"]=="No":

            recommendations.append(
                "🛠️ تفعيل خدمة الدعم الفني لتحسين تجربة العميل"
            )


        if customer["MonthlyCharges"]>80:

            recommendations.append(
                "💰 تقديم خصم أو باقة أقل تكلفة"
            )


        if customer["OnlineSecurity"]=="No":

            recommendations.append(
                "🔐 إضافة خدمة الحماية الإلكترونية"
            )


        if customer["tenure"]<12:

            recommendations.append(
                "🎁 تقديم عرض ترحيبي للحفاظ على العميل الجديد"
            )



    else:


        recommendations.append(
            "✅ العميل مستقر حاليًا"
        )


        recommendations.append(
            "⭐ يمكن تحسين الولاء بعروض إضافية"
        )



    return recommendations

# =========================================================
# SESSION STATE
# =========================================================

if "customer" not in st.session_state:

    st.session_state.customer = {


        "gender":"Female",

        "SeniorCitizen":"No",

        "Partner":"Yes",

        "Dependents":"No",

        "tenure":12,


        "PhoneService":"Yes",

        "MultipleLines":"No",

        "InternetService":"Fiber optic",


        "OnlineSecurity":"No",

        "OnlineBackup":"No",

        "DeviceProtection":"No",

        "TechSupport":"No",

        "StreamingTV":"No",

        "StreamingMovies":"No",


        "Contract":"Month-to-month",

        "PaperlessBilling":"Yes",

        "PaymentMethod":"Electronic check",


        "MonthlyCharges":70.0,

        "TotalCharges":840.0

    }



if "prediction" not in st.session_state:

    st.session_state.prediction=None

# =========================================================
# PLACEHOLDER METRICS
# =========================================================


col1,col2,col3,col4 = st.columns(4)



with col1:

    st.markdown("""

    <div class="card">

    <div class="card-title">
    Total Customers
    </div>


    <div class="card-value"
    style="color:#00c6ff">
    --
    </div>


    </div>

    """,
    unsafe_allow_html=True)




with col2:

    st.markdown("""

    <div class="card">

    <div class="card-title">
    Churn Rate
    </div>


    <div class="card-value"
    style="color:#ff5570">

    --

    </div>


    </div>

    """,
    unsafe_allow_html=True)




with col3:

    st.markdown("""

    <div class="card">

    <div class="card-title">
    High Risk Customers
    </div>


    <div class="card-value"
    style="color:#ffb84d">

    --

    </div>


    </div>

    """,
    unsafe_allow_html=True)




with col4:

    st.markdown("""

    <div class="card">

    <div class="card-title">
    Model Accuracy
    </div>


    <div class="card-value"
    style="color:#3ddc84">

    --

    </div>


    </div>

    """,
    unsafe_allow_html=True)




# =========================================================
# NAVIGATION TABS
# =========================================================


tabs = st.tabs(
[
"👤 Customer Prediction",
"📊 Analytics Dashboard",
"🧪 Scenario Simulator",
"🤖 AI Recommendations"
]
)



with tabs[0]:


    st.markdown(
    '<div class="section-title">👤 Customer Risk Prediction</div>',
    unsafe_allow_html=True
    )



    c=st.session_state.customer



    col1,col2,col3=st.columns(3)



    with col1:

        c["gender"]=st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )


        c["SeniorCitizen"]=st.selectbox(
            "Senior Citizen",
            [
                "No",
                "Yes"
            ]
        )



    with col2:


        c["Partner"]=st.selectbox(
            "Partner",
            [
                "No",
                "Yes"
            ]
        )


        c["Dependents"]=st.selectbox(
            "Dependents",
            [
                "No",
                "Yes"
            ]
        )



    with col3:

        c["tenure"]=st.slider(
            "Tenure Months",
            0,
            72,
            12
        )





    st.markdown(
    '<div class="section-title">Services</div>',
    unsafe_allow_html=True
    )


    col1,col2,col3=st.columns(3)



    with col1:


        c["InternetService"]=st.selectbox(

            "Internet Service",

            [
                "DSL",
                "Fiber optic",
                "No"
            ]

        )


        c["PhoneService"]=st.selectbox(

            "Phone Service",

            [
                "No",
                "Yes"
            ]

        )



    with col2:


        c["OnlineSecurity"]=st.selectbox(

            "Online Security",

            [
                "No",
                "Yes",
                "No internet service"
            ]

        )


        c["TechSupport"]=st.selectbox(

            "Tech Support",

            [
                "No",
                "Yes",
                "No internet service"
            ]

        )



    with col3:


        c["Contract"]=st.selectbox(

            "Contract",

            [
                "Month-to-month",
                "One year",
                "Two year"
            ]

        )


        c["PaymentMethod"]=st.selectbox(

            "Payment Method",

            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]

        )




    st.markdown(
    '<div class="section-title">Financial</div>',
    unsafe_allow_html=True
    )


    col1,col2=st.columns(2)


    with col1:

        c["MonthlyCharges"]=st.number_input(

            "Monthly Charges",

            value=70.0

        )


    with col2:

        c["TotalCharges"]=st.number_input(

            "Total Charges",

            value=840.0

        )




    st.divider()



    if st.button(
        "🚀 Predict Customer Risk"
    ):


        if model is None:

            st.error(
                "Model not loaded"
            )


        else:


            result=predict_churn(
                model,
                c
            )


            st.session_state.prediction=result




    if st.session_state.prediction:


        result=st.session_state.prediction



        probability=result["probability"]



        col1,col2=st.columns(2)



        with col1:


            if probability>=0.5:


                st.error(

                    f"""
                    ## 🚨 High Risk Customer

                    Churn Probability:

                    ### {result['percentage']}%

                    """

                )


            else:


                st.success(

                    f"""
                    ## 🟢 Low Risk Customer

                    Churn Probability:

                    ### {result['percentage']}%

                    """

                )



        with col2:


            fig=go.Figure(
                go.Indicator(

                    mode="gauge+number",

                    value=result["percentage"],

                    gauge={

                        "axis":{
                            "range":[0,100]
                        },

                        "bar":{
                            "color":"#00c6ff"
                        }

                    }

                )

            )


            fig.update_layout(

                height=300,

                paper_bgcolor="rgba(0,0,0,0)"

            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )



        st.markdown(
        "### 🔍 Model Input Features"
        )


        st.dataframe(
            result["features"],
            use_container_width=True
        )
        st.markdown("### 🤖 AI Retention Assistant")

        recommendations = generate_recommendations(
        c,
        result["probability"]
        )

        for item in recommendations:
            st.info(item)




with tabs[1]:


    st.markdown(
    '<div class="section-title">📊 Business Analytics Dashboard</div>',
    unsafe_allow_html=True
    )



    df = load_dataset()



    if df is None:


        st.warning(
            "Dataset not found"
        )


    else:


        # ==============================
        # KPIs
        # ==============================


        total_customers=len(df)


        churn_rate=round(

            (
                df["Churn"]=="Yes"

            ).mean()*100,

            2

        )



        revenue=df["MonthlyCharges"].sum()



        lost_revenue=df[
            df["Churn"]=="Yes"
        ]["MonthlyCharges"].sum()



        avg_tenure=round(

            df["tenure"].mean(),

            1

        )




        c1,c2,c3,c4=st.columns(4)



        with c1:

            st.markdown(
            f"""

            <div class="card">

            <div class="card-title">
            Customers
            </div>

            <div class="card-value"
            style="color:#00c6ff">

            {total_customers:,}

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )



        with c2:

            st.markdown(
            f"""

            <div class="card">

            <div class="card-title">
            Churn Rate
            </div>

            <div class="card-value"
            style="color:#ff5570">

            {churn_rate}%

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )




        with c3:

            st.markdown(
            f"""

            <div class="card">

            <div class="card-title">
            Avg Tenure
            </div>

            <div class="card-value"
            style="color:#3ddc84">

            {avg_tenure}

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )




        with c4:

            st.markdown(
            f"""

            <div class="card">

            <div class="card-title">
            Lost Revenue
            </div>

            <div class="card-value"
            style="color:#ffb84d">

            ${lost_revenue:,.0f}

            </div>

            </div>

            """,

            unsafe_allow_html=True

            )



        st.divider()



        # ==============================
        # Charts
        # ==============================


        col1,col2=st.columns(2)



        with col1:


            churn_contract=(

                df.groupby("Contract")
                ["Churn"]
                .apply(
                    lambda x:
                    (
                        x=="Yes"
                    ).mean()*100
                )
                .reset_index()

            )



            fig1=go.Figure()


            fig1.add_bar(

                x=churn_contract["Contract"],

                y=churn_contract["Churn"],

                text=

                churn_contract["Churn"].round(1)

            )


            fig1.update_layout(

                title="Churn By Contract",

                template="plotly_dark",

                height=400

            )


            st.plotly_chart(

                fig1,

                use_container_width=True

            )




        with col2:


            churn_internet=(

                df.groupby("InternetService")
                ["Churn"]
                .apply(
                    lambda x:
                    (
                        x=="Yes"
                    ).mean()*100
                )

                .reset_index()

            )



            fig2=go.Figure()


            fig2.add_bar(

                x=churn_internet["InternetService"],

                y=churn_internet["Churn"],

                text=

                churn_internet["Churn"].round(1)

            )



            fig2.update_layout(

                title="Churn By Internet",

                template="plotly_dark",

                height=400

            )


            st.plotly_chart(

                fig2,

                use_container_width=True

            )




        col3,col4=st.columns(2)



        with col3:


            fig3=go.Figure()


            fig3.add_histogram(

                x=df["tenure"],

                nbinsx=30

            )


            fig3.update_layout(

                title="Customer Tenure Distribution",

                template="plotly_dark"

            )


            st.plotly_chart(

                fig3,

                use_container_width=True

            )




        with col4:


            fig4=go.Figure()


            fig4.add_box(

                x=df["Churn"],

                y=df["MonthlyCharges"]

            )


            fig4.update_layout(

                title="Monthly Charges vs Churn",

                template="plotly_dark"

            )


            st.plotly_chart(

                fig4,

                use_container_width=True

            )




with tabs[2]:


    st.markdown(
    '<div class="section-title">🧪 Scenario Simulator</div>',
    unsafe_allow_html=True
    )


    if model is None:


        st.warning(
            "Model unavailable"
        )


    else:


        st.write(
        """
        قم بتغيير خصائص العميل وشاهد كيف يتغير خطر المغادرة.
        """
        )



        base=dict(
            st.session_state.customer
        )



        col1,col2,col3=st.columns(3)



        with col1:

            new_contract=st.selectbox(

                "New Contract",

                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]

            )



        with col2:

            new_tenure=st.slider(

                "New Tenure",

                0,

                72,

                base["tenure"]

            )



        with col3:

            new_price=st.slider(

                "Monthly Charges",

                0.0,

                150.0,

                float(
                    base["MonthlyCharges"]
                )

            )




        scenario=dict(base)



        scenario["Contract"]=new_contract

        scenario["tenure"]=new_tenure

        scenario["MonthlyCharges"]=new_price





        if st.button(
            "🔬 Compare Scenario"
        ):



            old=predict_churn(

                model,

                base

            )



            new=predict_churn(

                model,

                scenario

            )




            col1,col2=st.columns(2)



            with col1:


                st.metric(

                    "Original Risk",

                    f"{old['percentage']}%"

                )



            with col2:


                diff=round(

                    new["percentage"]

                    -

                    old["percentage"],

                    2

                )


                st.metric(

                    "New Risk",

                    f"{new['percentage']}%",

                    delta=f"{diff}%"

                )





            fig=go.Figure()



            fig.add_bar(

                x=[
                    "Before",
                    "After"
                ],

                y=[

                    old["percentage"],

                    new["percentage"]

                ]

            )



            fig.update_layout(

                template="plotly_dark",

                title="Risk Change"

            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )




with tabs[3]:


    st.markdown(
    '<div class="section-title">🤖 AI Model Intelligence</div>',
    unsafe_allow_html=True
    )


    if model is None:


        st.warning(
            "Model unavailable"
        )


    else:


        st.write(
        """
        أهم العوامل التي يعتمد عليها النموذج في توقع مغادرة العملاء.
        """
        )


        try:


            importance=model.feature_importances_


            feature_names=[

                "gender",
                "SeniorCitizen",
                "Partner",
                "Dependents",
                "tenure",
                "PhoneService",
                "MultipleLines",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "PaperlessBilling",
                "MonthlyCharges",
                "TotalCharges"

            ]



            imp=pd.DataFrame({

                "Feature":
                feature_names[:len(importance)],

                "Importance":
                importance

            })


            imp=imp.sort_values(

                "Importance",

                ascending=False

            ).head(10)




            fig=go.Figure()



            fig.add_bar(

                x=imp["Importance"],

                y=imp["Feature"],

                orientation="h"

            )


            fig.update_layout(

                template="plotly_dark",

                title="Top Churn Drivers"

            )


            st.plotly_chart(

                fig,

                use_container_width=True

            )



        except:


            st.info(
                "Feature importance unavailable"
            )




# =========================================================
# FOOTER
# =========================================================


st.markdown(
"""

<br><br>

<div style="
text-align:center;
color:#64748b;
">

TelcoGuard AI © 2026
<br>
Machine Learning Customer Intelligence System

</div>


""",

unsafe_allow_html=True
)

<div style="text-align:center; color:#5a6472; font-size:0.85rem;">
نظام توقع مغادرة العملاء — مبني باستخدام Streamlit و XGBoost
</div>
""", unsafe_allow_html=True)
