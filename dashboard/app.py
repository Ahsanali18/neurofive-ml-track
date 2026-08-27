import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# -----------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------
st.set_page_config(
    page_title="Churn Predictor Pro",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------------------
# CUSTOM CSS
# -----------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* ENHANCED MAIN TITLE */
    .main-title {
        font-size: 4.2rem !important;        /* Increased for high visual impact */
        font-weight: 900 !important;
        margin-top: -1.5rem !important;      /* Removes excess top whitespace */
        margin-bottom: 0.75rem !important;
        color: #FFFFFF !important;
        letter-spacing: -1.5px !important;   /* Tight headline tracking */
        text-align: center !important;       /* Forces center alignment */
        line-height: 1.15 !important;
    }

    /* ENHANCED CENTERED SUBTITLE */
    .subtitle {
        color: #94A3B8 !important;
        font-size: 1.35rem !important;       /* Scaled up to match larger title */
        text-align: center !important;       /* Centers multi-line text */
        max-width: 800px;                    /* Restricts line length for readability */
        margin: 0 auto 2.5rem auto !important;/* Keeps the block centered horizontally */
        line-height: 1.6 !important;
        display: block;
    }

    .stApp h3 {
        color: #FFFFFF !important;
    }

    .stSelectbox label p, .stSlider label p, .stNumberInput label p {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    div[data-testid="stSlider"] div[data-testid="stTickBar"] {
        color: #CBD5E1 !important;
    }
    div[data-testid="stSlider"] div[role="slider"] {
        color: #FFFFFF !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        color: #F8FAFC !important;
        border-radius: 8px;
    }
    div[data-baseweb="select"] span {
        color: #F8FAFC !important;
    }
    div[data-baseweb="select"] svg {
        fill: #94A3B8 !important;
    }

    ul[data-testid="stSelectboxVirtualDropdown"] {
        background-color: #1E293B !important;
    }
    li[role="option"] {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }
    li[role="option"]:hover {
        background-color: #334155 !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
        border-radius: 8px;
    }
    div[data-testid="stNumberInput"] button {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
        border: 1px solid #334155 !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%);
        color: white !important;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 0.9rem 0;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(99, 102, 241, 0.2);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(99, 102, 241, 0.4);
        background: linear-gradient(135deg, #4F46E5 0%, #4338CA 100%);
    }

    div[data-testid="stExpander"] {
        background-color: #1E293B !important;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] details,
    div[data-testid="stExpander"] div {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] li {
        color: #CBD5E1 !important;
    }
    div[data-testid="stExpander"] strong {
        color: #FFFFFF !important;
    }

    div[data-testid="stMetric"] {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-size: 2.2rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8 !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------
# RESOURCE CACHING (Performance)
# -----------------------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load('../models/churn_pipeline.pkl')
    except FileNotFoundError:
        return None

model = load_model()


# -----------------------------------------
# HEADER SECTION
# -----------------------------------------
st.markdown('<h1 class="main-title">📉 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">An AI-powered risk assessment Enter customer details below to predict the whether the customer churn or not.</p>', unsafe_allow_html=True)

# -----------------------------------------
# INPUT SECTION (7 / 6 / 7 fields)
# -----------------------------------------
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    with st.container(border=True):
        st.subheader("💳 Account & Billing")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])

with col2:
    with st.container(border=True):
        st.subheader("👤 Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col3:
    with st.container(border=True):
        st.subheader("🌐 Tech Services")
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

st.write("")
predict_clicked = st.button("🔮 Predict Churn Risk", width='stretch')


# -----------------------------------------
# PREDICTION & RESULTS SECTION
# -----------------------------------------
if predict_clicked:
    if model is None:
        st.error("⚠️ **Model file not found.** Please ensure `../models/churn_pipeline.pkl` exists.")
    else:
        st.divider()
        st.markdown("### 📊 Prediction Results")

        avg_monthly_spend = total_charges / max(tenure, 1)
        service_flags = [online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies]
        num_services = sum(1 for s in service_flags if s == "Yes")

        input_df = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges,
            'Avg_Monthly_Spend': avg_monthly_spend,
            'Num_Services': num_services
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        res_col1, res_col2 = st.columns([1, 1.5], gap="large")

        with res_col1:
            st.write("")
            st.write("")
            if prediction == 1:
                st.error("### ⚠️ High Risk of Churn")
                st.write("This customer exhibits patterns strongly associated with leaving the service. Consider immediate retention strategies like discounts or priority support.")
            else:
                st.success("### ✅ Likely to Stay")
                st.write("This customer shows strong loyalty signals. Standard account management procedures are recommended.")

            st.metric(label="Model Confidence", value=f"{probability:.1%}")

        with res_col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={'suffix': "%", 'font': {'color': '#F8FAFC'}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Churn Probability", 'font': {'color': '#94A3B8'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#6366F1"},
                    'bgcolor': "#1E293B",
                    'borderwidth': 2,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 33], 'color': "rgba(16, 185, 129, 0.3)"},
                        {'range': [33, 66], 'color': "rgba(245, 158, 11, 0.3)"},
                        {'range': [66, 100], 'color': "rgba(239, 68, 68, 0.3)"}],
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=300,
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig, width='stretch')


# -----------------------------------------
# FOOTER / HELP SECTION
# -----------------------------------------
st.write("")
with st.expander("ℹ️ Data Dictionary & Feature Meanings"):
    st.markdown("""
    *   **Tenure**: How many months the customer has been with the company.
    *   **Monthly / Total Charges**: The amount billed to the customer currently, and historically.
    *   **Contract Type**: Month-to-month, 1-year, or 2-year commitments.
    *   **Payment Method**: The medium used by the customer to settle their bills.
    *   **Demographics**: Basic personal details (Gender, Age group, Dependents, Partner status).
    *   **Services**: Various add-ons provided by the company (Phone, Internet, Security, Backup, Streaming).
    """)