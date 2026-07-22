import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="نظام توقع مغادرة العملاء",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Cairo', sans-serif;
    direction: rtl;
}

.main {
    background-color: #0f1117;
}

.main-header {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2a 100%);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
    text-align: center;
    box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    border: 1px solid rgba(0, 200, 255, 0.15);
}

.main-header h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 900;
    margin: 0;
}

.main-header p {
    color: #9fb3c8;
    font-size: 1.05rem;
    margin-top: 8px;
}

.metric-card {
    background: linear-gradient(145deg, #1a1f2b, #12151d);
    border-radius: 16px;
    padding: 22px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 6px 18px rgba(0,0,0,0.3);
    margin-bottom: 15px;
}

.metric-card h2 {
    font-size: 2.2rem;
    margin: 5px 0;
}

.metric-card p {
    color: #8892a0;
    font-size: 0.95rem;
    margin: 0;
}

.risk-high {
    background: linear-gradient(145deg, #4a1620, #2b0d13);
    border: 1px solid rgba(255, 70, 90, 0.4);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.risk-high h1 {
    color: #ff4b5c;
    font-size: 3rem;
    margin: 0;
}

.risk-low {
    background: linear-gradient(145deg, #123324, #0b2116);
    border: 1px solid rgba(60, 220, 130, 0.4);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
}

.risk-low h1 {
    color: #3ddc84;
    font-size: 3rem;
    margin: 0;
}

.section-title {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
    border-right: 5px solid #00c6ff;
    padding-right: 12px;
    margin: 20px 0 15px 0;
}

.stButton>button {
    background: linear-gradient(90deg, #0072ff, #00c6ff);
    color: white;
    font-weight: 700;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
    width: 100%;
    transition: 0.3s;
}

.stButton>button:hover {
    box-shadow: 0 0 15px rgba(0, 198, 255, 0.6);
    transform: translateY(-2px);
}

.info-box {
    background: rgba(0, 198, 255, 0.08);
    border: 1px solid rgba(0, 198, 255, 0.3);
    border-radius: 12px;
    padding: 15px 20px;
    color: #d5e8f5;
    margin-bottom: 15px;
}

div[data-testid="stTabs"] button {
    font-weight: 700;
    font-size: 1.05rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    possible_paths = ["xgboost.pkl", "xgb_model.pkl"]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    model = pickle.load(f)
                return model, path
            except Exception:
                continue
    return None, None


FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "InternetService_DSL", "InternetService_Fiber optic", "InternetService_No",
    "Contract_Month-to-month", "Contract_One year", "Contract_Two year",
    "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"
]


def encode_service_column(value, has_three_states=True):
    mapping_three = {"No": 0, "No internet service": 1, "No phone service": 1, "Yes": 2}
    mapping_two = {"No": 0, "Yes": 1}
    if has_three_states:
        return mapping_three.get(value, 0)
    return mapping_two.get(value, 0)


def build_feature_row(data):
    row = {}
    row["gender"] = 1 if data["gender"] == "Male" else 0
    row["SeniorCitizen"] = 1 if data["SeniorCitizen"] == "Yes" else 0
    row["Partner"] = encode_service_column(data["Partner"], has_three_states=False)
    row["Dependents"] = encode_service_column(data["Dependents"], has_three_states=False)
    row["tenure"] = data["tenure"]
    row["PhoneService"] = encode_service_column(data["PhoneService"], has_three_states=False)
    row["MultipleLines"] = encode_service_column(data["MultipleLines"], has_three_states=True)
    row["OnlineSecurity"] = encode_service_column(data["OnlineSecurity"], has_three_states=True)
    row["OnlineBackup"] = encode_service_column(data["OnlineBackup"], has_three_states=True)
    row["DeviceProtection"] = encode_service_column(data["DeviceProtection"], has_three_states=True)
    row["TechSupport"] = encode_service_column(data["TechSupport"], has_three_states=True)
    row["StreamingTV"] = encode_service_column(data["StreamingTV"], has_three_states=True)
    row["StreamingMovies"] = encode_service_column(data["StreamingMovies"], has_three_states=True)
    row["PaperlessBilling"] = encode_service_column(data["PaperlessBilling"], has_three_states=False)
    row["MonthlyCharges"] = data["MonthlyCharges"]
    row["TotalCharges"] = data["TotalCharges"]

    row["InternetService_DSL"] = 1 if data["InternetService"] == "DSL" else 0
    row["InternetService_Fiber optic"] = 1 if data["InternetService"] == "Fiber optic" else 0
    row["InternetService_No"] = 1 if data["InternetService"] == "No" else 0

    row["Contract_Month-to-month"] = 1 if data["Contract"] == "Month-to-month" else 0
    row["Contract_One year"] = 1 if data["Contract"] == "One year" else 0
    row["Contract_Two year"] = 1 if data["Contract"] == "Two year" else 0

    row["PaymentMethod_Bank transfer (automatic)"] = 1 if data["PaymentMethod"] == "Bank transfer (automatic)" else 0
    row["PaymentMethod_Credit card (automatic)"] = 1 if data["PaymentMethod"] == "Credit card (automatic)" else 0
    row["PaymentMethod_Electronic check"] = 1 if data["PaymentMethod"] == "Electronic check" else 0
    row["PaymentMethod_Mailed check"] = 1 if data["PaymentMethod"] == "Mailed check" else 0

    df = pd.DataFrame([row])
    df = df[FEATURE_COLUMNS]
    return df


def predict_churn(model, data_dict):
    df = build_feature_row(data_dict)
    proba = model.predict_proba(df)[0][1]
    return proba, df


def default_customer_data():
    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
    }


if "customer_data" not in st.session_state:
    st.session_state.customer_data = default_customer_data()

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "prediction_features" not in st.session_state:
    st.session_state.prediction_features = None


st.markdown("""
<div class="main-header">
    <h1>📡 نظام توقع مغادرة العملاء</h1>
    <p>Telecom Customer Churn Prediction System — مدعوم بنموذج XGBoost</p>
</div>
""", unsafe_allow_html=True)

model, loaded_path = load_model()

if model is None:
    st.error("⚠️ لم يتم العثور على ملف النموذج. تأكد من وجود ملف `xgboost.pkl` في نفس مجلد التطبيق.")
else:
    st.markdown(f"""
    <div class="info-box">
    ✅ تم تحميل النموذج بنجاح من الملف: <b>{loaded_path}</b>
    </div>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "📝 إدخال بيانات العميل",
    "🎯 التوقع والنتيجة",
    "🧪 محاكاة السيناريوهات",
    "📊 تقارير ورسوم بيانية"
])

with tab1:
    st.markdown('<div class="section-title">البيانات الشخصية</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("النوع (Gender)", ["Female", "Male"],
                               index=["Female", "Male"].index(st.session_state.customer_data["gender"]))
        senior = st.selectbox("مواطن كبير السن (Senior Citizen)", ["No", "Yes"],
                               index=["No", "Yes"].index(st.session_state.customer_data["SeniorCitizen"]))

    with col2:
        partner = st.selectbox("لديه شريك (Partner)", ["No", "Yes"],
                                index=["No", "Yes"].index(st.session_state.customer_data["Partner"]))
        dependents = st.selectbox("لديه معالين (Dependents)", ["No", "Yes"],
                                   index=["No", "Yes"].index(st.session_state.customer_data["Dependents"]))

    with col3:
        tenure = st.number_input("مدة الاشتراك بالأشهر (Tenure)", min_value=0, max_value=100,
                                  value=int(st.session_state.customer_data["tenure"]))

    st.markdown('<div class="section-title">الخدمات المشترك بها</div>', unsafe_allow_html=True)
    col4, col5, col6 = st.columns(3)

    with col4:
        phone_service = st.selectbox("خدمة الهاتف (Phone Service)", ["No", "Yes"],
                                      index=["No", "Yes"].index(st.session_state.customer_data["PhoneService"]))
        multiple_lines = st.selectbox("خطوط متعددة (Multiple Lines)",
                                       ["No", "Yes", "No phone service"],
                                       index=["No", "Yes", "No phone service"].index(st.session_state.customer_data["MultipleLines"]))
        internet_service = st.selectbox("خدمة الإنترنت (Internet Service)",
                                         ["DSL", "Fiber optic", "No"],
                                         index=["DSL", "Fiber optic", "No"].index(st.session_state.customer_data["InternetService"]))

    with col5:
        online_security = st.selectbox("حماية أونلاين (Online Security)",
                                        ["No", "Yes", "No internet service"],
                                        index=["No", "Yes", "No internet service"].index(st.session_state.customer_data["OnlineSecurity"]))
        online_backup = st.selectbox("نسخ احتياطي أونلاين (Online Backup)",
                                      ["No", "Yes", "No internet service"],
                                      index=["No", "Yes", "No internet service"].index(st.session_state.customer_data["OnlineBackup"]))
        device_protection = st.selectbox("حماية الجهاز (Device Protection)",
                                          ["No", "Yes", "No internet service"],
                                          index=["No", "Yes", "No internet service"].index(st.session_state.customer_data["DeviceProtection"]))

    with col6:
        tech_support = st.selectbox("الدعم الفني (Tech Support)",
                                     ["No", "Yes", "No internet service"],
                                     index=["No", "Yes", "No internet service"].index(st.session_state.customer_data["TechSupport"]))
        streaming_tv = st.selectbox("بث تلفزيوني (Streaming TV)",
                                     ["No", "Yes", "No internet service"],
                                     index=["No", "Yes", "No internet service"].index(st.session_state.customer_data["StreamingTV"]))
        streaming_movies = st.selectbox("بث أفلام (Streaming Movies)",
                                         ["No", "Yes", "No internet service"],
                                         index=["No", "Yes", "No internet service"].index(st.session_state.customer_data["StreamingMovies"]))

    st.markdown('<div class="section-title">تفاصيل العقد والدفع</div>', unsafe_allow_html=True)
    col7, col8, col9 = st.columns(3)

    with col7:
        contract = st.selectbox("نوع العقد (Contract)",
                                 ["Month-to-month", "One year", "Two year"],
                                 index=["Month-to-month", "One year", "Two year"].index(st.session_state.customer_data["Contract"]))
        paperless_billing = st.selectbox("فاتورة إلكترونية (Paperless Billing)", ["No", "Yes"],
                                          index=["No", "Yes"].index(st.session_state.customer_data["PaperlessBilling"]))

    with col8:
        payment_method = st.selectbox("طريقة الدفع (Payment Method)",
                                       ["Electronic check", "Mailed check",
                                        "Bank transfer (automatic)", "Credit card (automatic)"],
                                       index=["Electronic check", "Mailed check",
                                              "Bank transfer (automatic)", "Credit card (automatic)"].index(st.session_state.customer_data["PaymentMethod"]))

    with col9:
        monthly_charges = st.number_input("الفاتورة الشهرية (Monthly Charges)",
                                           min_value=0.0, max_value=500.0,
                                           value=float(st.session_state.customer_data["MonthlyCharges"]),
                                           step=0.5)
        total_charges = st.number_input("إجمالي الفواتير (Total Charges)",
                                         min_value=0.0, max_value=10000.0,
                                         value=float(st.session_state.customer_data["TotalCharges"]),
                                         step=1.0)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("💾 حفظ البيانات والانتقال للتوقع"):
        st.session_state.customer_data = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
        }
        st.success("✅ تم حفظ البيانات بنجاح! انتقل الآن إلى تبويب (التوقع والنتيجة).")

with tab2:
    st.markdown('<div class="section-title">نتيجة التوقع الحالية</div>', unsafe_allow_html=True)

    if model is None:
        st.warning("لا يمكن إجراء التوقع لعدم توفر النموذج.")
    else:
        if st.button("🚀 تشغيل التوقع الآن"):
            proba, features_df = predict_churn(model, st.session_state.customer_data)
            st.session_state.prediction_result = proba
            st.session_state.prediction_features = features_df

        if st.session_state.prediction_result is not None:
            proba = st.session_state.prediction_result
            percentage = round(proba * 100, 2)

            col_a, col_b = st.columns([1, 1])

            with col_a:
                if proba >= 0.5:
                    st.markdown(f"""
                    <div class="risk-high">
                        <p style="color:#ffb3ba; font-size:1.1rem;">⚠️ العميل مرشح للمغادرة</p>
                        <h1>{percentage}%</h1>
                        <p style="color:#ffb3ba;">احتمالية المغادرة (Churn)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        <p style="color:#b3ffd1; font-size:1.1rem;">✅ العميل غير مرشح للمغادرة</p>
                        <h1>{percentage}%</h1>
                        <p style="color:#b3ffd1;">احتمالية المغادرة (Churn)</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_b:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=percentage,
                    title={'text': "مؤشر خطورة المغادرة", 'font': {'size': 18}},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#00c6ff"},
                        'steps': [
                            {'range': [0, 40], 'color': "#123324"},
                            {'range': [40, 70], 'color': "#4a3d16"},
                            {'range': [70, 100], 'color': "#4a1620"}
                        ],
                    }
                ))
                fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20),
                                   paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
                st.plotly_chart(fig, use_container_width=True)

            st.markdown('<div class="section-title">ملخص بيانات العميل المستخدمة</div>', unsafe_allow_html=True)
            st.dataframe(st.session_state.prediction_features, use_container_width=True)
        else:
            st.info("اضغط على زر (تشغيل التوقع الآن) لعرض النتيجة بناءً على البيانات المدخلة في التبويب الأول.")

with tab3:
    st.markdown('<div class="section-title">محاكاة السيناريوهات المختلفة</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    عدّل القيم أدناه لمعرفة كيف يتغير احتمال مغادرة نفس العميل عند تغيير نوع العقد، مدة الاشتراك، أو الفاتورة الشهرية.
    </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.warning("لا يمكن إجراء المحاكاة لعدم توفر النموذج.")
    else:
        base_data = dict(st.session_state.customer_data)

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            sim_contract = st.selectbox("نوع العقد (سيناريو)",
                                         ["Month-to-month", "One year", "Two year"],
                                         index=["Month-to-month", "One year", "Two year"].index(base_data["Contract"]),
                                         key="sim_contract")
        with col_s2:
            sim_tenure = st.slider("مدة الاشتراك بالأشهر (سيناريو)", 0, 72, int(base_data["tenure"]), key="sim_tenure")
        with col_s3:
            sim_monthly = st.slider("الفاتورة الشهرية (سيناريو)", 0.0, 150.0, float(base_data["MonthlyCharges"]), step=0.5, key="sim_monthly")

        col_s4, col_s5 = st.columns(2)
        with col_s4:
            sim_internet = st.selectbox("خدمة الإنترنت (سيناريو)",
                                         ["DSL", "Fiber optic", "No"],
                                         index=["DSL", "Fiber optic", "No"].index(base_data["InternetService"]),
                                         key="sim_internet")
        with col_s5:
            sim_security = st.selectbox("حماية أونلاين (سيناريو)",
                                         ["No", "Yes", "No internet service"],
                                         index=["No", "Yes", "No internet service"].index(base_data["OnlineSecurity"]),
                                         key="sim_security")

        scenario_data = dict(base_data)
        scenario_data["Contract"] = sim_contract
        scenario_data["tenure"] = sim_tenure
        scenario_data["MonthlyCharges"] = sim_monthly
        scenario_data["InternetService"] = sim_internet
        scenario_data["OnlineSecurity"] = sim_security

        if st.button("🧪 قارن السيناريو بالوضع الأصلي"):
            original_proba, _ = predict_churn(model, base_data)
            scenario_proba, _ = predict_churn(model, scenario_data)

            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"""
                <div class="metric-card">
                    <p>الوضع الأصلي</p>
                    <h2 style="color:#00c6ff;">{round(original_proba*100,2)}%</h2>
                </div>
                """, unsafe_allow_html=True)

            with col_r2:
                delta = round((scenario_proba - original_proba) * 100, 2)
                color = "#ff4b5c" if delta > 0 else "#3ddc84"
                st.markdown(f"""
                <div class="metric-card">
                    <p>الوضع بعد التعديل (السيناريو)</p>
                    <h2 style="color:{color};">{round(scenario_proba*100,2)}%</h2>
                    <p style="color:{color};">الفرق: {delta}%</p>
                </div>
                """, unsafe_allow_html=True)

            fig_compare = go.Figure(data=[
                go.Bar(name="الوضع الأصلي", x=["احتمالية المغادرة"], y=[original_proba*100], marker_color="#00c6ff"),
                go.Bar(name="السيناريو المعدل", x=["احتمالية المغادرة"], y=[scenario_proba*100], marker_color="#ff4b5c")
            ])
            fig_compare.update_layout(barmode='group', height=350,
                                       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                       font={'color': "white"})
            st.plotly_chart(fig_compare, use_container_width=True)

with tab4:
    st.markdown('<div class="section-title">تقارير ورسوم بيانية عامة</div>', unsafe_allow_html=True)

    csv_candidates = ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "WA_FnUseC_TelcoCustomerChurn.csv"]
    df_raw = None
    for candidate in csv_candidates:
        if os.path.exists(candidate):
            try:
                df_raw = pd.read_csv(candidate)
                break
            except Exception:
                continue

    if df_raw is None:
        st.warning("لم يتم العثور على ملف بيانات العملاء (CSV) لعرض التقارير العامة.")
    else:
        df_raw["TotalCharges"] = pd.to_numeric(df_raw["TotalCharges"], errors="coerce")
        df_raw = df_raw.dropna(subset=["TotalCharges"])

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        churn_rate = round((df_raw["Churn"] == "Yes").mean() * 100, 2)
        avg_tenure = round(df_raw["tenure"].mean(), 1)
        avg_monthly = round(df_raw["MonthlyCharges"].mean(), 2)
        total_customers = len(df_raw)

        with col_m1:
            st.markdown(f"""
            <div class="metric-card"><p>إجمالي العملاء</p><h2 style="color:#00c6ff;">{total_customers}</h2></div>
            """, unsafe_allow_html=True)
        with col_m2:
            st.markdown(f"""
            <div class="metric-card"><p>نسبة المغادرة</p><h2 style="color:#ff4b5c;">{churn_rate}%</h2></div>
            """, unsafe_allow_html=True)
        with col_m3:
            st.markdown(f"""
            <div class="metric-card"><p>متوسط مدة الاشتراك</p><h2 style="color:#3ddc84;">{avg_tenure}</h2></div>
            """, unsafe_allow_html=True)
        with col_m4:
            st.markdown(f"""
            <div class="metric-card"><p>متوسط الفاتورة الشهرية</p><h2 style="color:#f7b733;">{avg_monthly}$</h2></div>
            """, unsafe_allow_html=True)

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            churn_by_contract = df_raw.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig1 = px.bar(churn_by_contract, x="Contract", y="Churn",
                          title="نسبة المغادرة حسب نوع العقد",
                          color="Contract", text_auto=".2f")
            fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig1, use_container_width=True)

        with col_c2:
            churn_by_internet = df_raw.groupby("InternetService")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
            fig2 = px.bar(churn_by_internet, x="InternetService", y="Churn",
                          title="نسبة المغادرة حسب خدمة الإنترنت",
                          color="InternetService", text_auto=".2f")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig2, use_container_width=True)

        col_c3, col_c4 = st.columns(2)

        with col_c3:
            fig3 = px.histogram(df_raw, x="tenure", color="Churn", nbins=30,
                                 title="توزيع مدة الاشتراك حسب المغادرة",
                                 barmode="overlay")
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig3, use_container_width=True)

        with col_c4:
            fig4 = px.box(df_raw, x="Churn", y="MonthlyCharges", color="Churn",
                          title="توزيع الفاتورة الشهرية حسب المغادرة")
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
            st.plotly_chart(fig4, use_container_width=True)

        churn_by_payment = df_raw.groupby("PaymentMethod")["Churn"].apply(lambda s: (s == "Yes").mean() * 100).reset_index()
        fig5 = px.bar(churn_by_payment, x="PaymentMethod", y="Churn",
                      title="نسبة المغادرة حسب طريقة الدفع",
                      color="PaymentMethod", text_auto=".2f")
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
        st.plotly_chart(fig5, use_container_width=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#5a6472; font-size:0.85rem;">
نظام توقع مغادرة العملاء — مبني باستخدام Streamlit و XGBoost
</div>
""", unsafe_allow_html=True)
