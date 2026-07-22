import streamlit as st
import pandas as pd
import joblib

# ضبط إعدادات الصفحة
st.set_page_config(page_title="توقع تسرب العملاء (Churn Risk)", layout="wide")

st.title("📊 نظام التنبؤ بتسرب العملاء - Telco Customer Churn")
st.write("قم بتعديل بيانات العميل لمعرفة نسبة احتمالية الانسحاب في الوقت الفعلي.")

# 1. تحميل الملفات المحفوظة
@st.cache_resource
def load_resources():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_resources()
except Exception as e:
    st.error("❌ حدث خطأ أثناء تحميل الملفات المحفوظة. تأكد من تشغيل train.py أولاً.")
    st.stop()

# 2. إنشاء الواجهة لل مدخلات
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("معلومات الاشتراك")
    tenure = st.slider("مدة الاشتراك (بالشهور)", 0, 72, 12)
    monthly_charges = st.number_input("الفاتورة الشهرية ($)", 18.0, 150.0, 70.0)
    total_charges = st.number_input("إجمالي المدفوعات ($)", 0.0, 10000.0, float(tenure * monthly_charges))
    contract = st.selectbox("نوع العقد", ["Month-to-month", "One year", "Two year"])

with col2:
    st.subheader("الخدمات والشبكة")
    internet_service = st.selectbox("خدمة الإنترنت", ["DSL", "Fiber optic", "No"])
    payment_method = st.selectbox("طريقة الدفع", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tech_support = st.selectbox("الدعم الفني", ["Yes", "No"])
    online_security = st.selectbox("الأمان عبر الإنترنت", ["Yes", "No"])

with col3:
    st.subheader("بيانات شخصية إضافية")
    gender = st.selectbox("النوع", ["Male", "Female"])
    partner = st.selectbox("مرتبط (Partner)", ["Yes", "No"])
    dependents = st.selectbox("يعول إخوة/أطفال", ["Yes", "No"])
    paperless_billing = st.selectbox("فواتير إلكترونية", ["Yes", "No"])

# 3. تجهيز البيانات للتنبؤ
input_dict = {
    'gender': 1 if gender == "Male" else 0,
    'SeniorCitizen': 0,
    'Partner': 1 if partner == "Yes" else 0,
    'Dependents': 1 if dependents == "Yes" else 0,
    'tenure': tenure,
    'PhoneService': 1,
    'MultipleLines': 0,
    'OnlineSecurity': 1 if online_security == "Yes" else 0,
    'OnlineBackup': 0,
    'DeviceProtection': 0,
    'TechSupport': 1 if tech_support == "Yes" else 0,
    'StreamingTV': 0,
    'StreamingMovies': 0,
    'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'InternetService_DSL': 1 if internet_service == "DSL" else 0,
    'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
    'InternetService_No': 1 if internet_service == "No" else 0,
    'Contract_Month-to-month': 1 if contract == "Month-to-month" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
    'PaymentMethod_Bank transfer (automatic)': 1 if payment_method == "Bank transfer (automatic)" else 0,
    'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
    'PaymentMethod_Electronic check': 1 if payment_method == "Electronic check" else 0,
    'PaymentMethod_Mailed check': 1 if payment_method == "Mailed check" else 0,
}

input_df = pd.DataFrame([input_dict])

# إعادة ترتيب الأعمدة لتطابق النموذج
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# تطبيق Scaler للأرقام
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# 4. التنبؤ
prob = model.predict_proba(input_df)[0][1]
churn_percent = round(prob * 100, 2)

st.divider()
st.subheader("🎯 النتيجة وخطر الانسحاب (Real-time Churn Risk):")

# عرض النتيجة حسب درجة الخطورة
if churn_percent >= 70:
    st.error(f"🚨 **مستوى الخطورة: عالي جدًا ({churn_percent}%)**")
    st.write("💡 **توصية:** العميل قريب جدًا من إلغاء الخدمة. يُفضل التواصل معه فوراً وتقديم عرض خصم خاص على العقد.")
elif churn_percent >= 40:
    st.warning(f"⚠️ **مستوى الخطورة: متوسط ({churn_percent}%)**")
    st.write("💡 **توصية:** اقتراح خدمات إضافية مثل الدعم الفني مجانًا لتشجيعه على البقاء.")
else:
    st.success(f"✅ **مستوى الخطورة: منخفض ({churn_percent}%)**")
    st.write("💡 العميل مستقر وحالته ممتازة.")