import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIGURATION (Must be the first Streamlit command)
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GariQeemat | Smart Car Price Estimator",
    page_icon="🚘",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# ═══════════════════════════════════════════════════════════════
# LOAD ARTIFACTS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "best_model.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    with open(MODELS_DIR / "feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    with open(MODELS_DIR / "rare_brands.json", "r") as f:
        rare_brands = json.load(f)
    with open(MODELS_DIR / "model_freq_lookup.json", "r") as f:
        model_freq_lookup = json.load(f)

    fair_price_df = pd.read_csv(PROCESSED_DIR / "fair_price_lookup.csv")
    return model, scaler, feature_columns, rare_brands, model_freq_lookup, fair_price_df

model, scaler, feature_columns, rare_brands, model_freq_lookup, fair_price_df = load_artifacts()

# Build dropdown options from saved feature names
city_options = sorted({c.replace("City_", "") for c in feature_columns if c.startswith("City_")})
fuel_options = sorted({c.replace("Fuel Type_", "") for c in feature_columns if c.startswith("Fuel Type_")})
transmission_options = sorted({c.replace("Transmission_", "") for c in feature_columns if c.startswith("Transmission_")})
brand_options = sorted({c.replace("Brand_Grouped_", "") for c in feature_columns if c.startswith("Brand_Grouped_")})
model_options = sorted(model_freq_lookup.keys())

# ═══════════════════════════════════════════════════════════════
# PKR FORMATTER
# ═══════════════════════════════════════════════════════════════
def format_pkr(value: int) -> str:
    """Format Pakistani Rupees into readable Lac / Crore notation."""
    if value >= 1_00_00_000:          # 1 Crore = 10 million
        return f"₨ {value / 1_00_00_000:.2f} Crore"
    elif value >= 1_00_000:           # 1 Lac = 100,000
        return f"₨ {value / 1_00_000:.2f} Lac"
    else:
        return f"₨ {value:,.0f}"

# ═══════════════════════════════════════════════════════════════
# PREMIUM CUSTOM CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        /* Hero Banner */
        .hero-banner {
            background: linear-gradient(135deg, #01411C 0%, #038a37 100%);
            border-radius: 20px;
            padding: 2.5rem 1rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(1, 65, 28, 0.2);
            margin-bottom: 2rem;
        }
        .hero-banner h1 {
            font-size: 3rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            color: white !important;
            letter-spacing: -1px;
        }
        .hero-banner p {
            font-size: 1.15rem;
            font-weight: 300;
            margin: 0;
            opacity: 0.9;
        }

        /* Metric Cards Flexbox */
        .metrics-container {
            display: flex;
            gap: 15px;
            margin: 1.5rem 0;
            flex-wrap: wrap;
        }
        .metric-card {
            flex: 1;
            min-width: 200px;
            background: white;
            border: 1px solid #edf2f7;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
            transition: all 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 24px rgba(0,0,0,0.08);
        }
        .metric-card.primary {
            background: linear-gradient(135deg, #01411C 0%, #026428 100%);
            color: white;
            border: none;
        }
        .metric-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #718096;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .primary .metric-title {
            color: #c6f6d5;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            color: #1a202c;
        }
        .primary .metric-value {
            color: white;
        }
        .metric-sub {
            font-size: 0.85rem;
            color: #a0aec0;
            font-weight: 400;
        }
        .primary .metric-sub {
            color: #9ae6b4;
        }

        /* Verdict Boxes */
        .verdict-box {
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            font-size: 1.3rem;
            font-weight: 700;
            margin: 1.5rem 0 2.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        .verdict-under {
            background: #f0fdf4;
            color: #166534;
            border: 2px solid #4ade80;
        }
        .verdict-fair {
            background: #fefce8;
            color: #854d0e;
            border: 2px solid #facc15;
        }
        .verdict-over {
            background: #fef2f2;
            color: #991b1b;
            border: 2px solid #f87171;
        }

        /* Insights */
        .insight-row {
            background-color: #f8fafc;
            border-left: 4px solid #01411C;
            padding: 1rem 1.2rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 0.8rem;
            color: #334155;
            font-size: 0.95rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.02);
            transition: background 0.2s;
        }
        .insight-row:hover {
            background-color: #f1f5f9;
        }

        /* Form styling */
        div[data-testid="stForm"] {
            background-color: #ffffff;
            padding: 2.5rem;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.04);
            border: 1px solid #f1f5f9;
        }
        button[kind="formSubmit"] {
            background: linear-gradient(90deg, #01411C, #038a37) !important;
            color: white !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.8rem 2rem !important;
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            transition: all 0.3s ease;
            width: 100%;
            margin-top: 1rem;
        }
        button[kind="formSubmit"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(1,65,28,0.25) !important;
        }

        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-bottom: 2rem;
            font-weight: 400;
        }
    </style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
    <div class="hero-banner">
        <h1>🇵🇰 GariQeemat</h1>
        <p>Pakistan's Smartest Used Car Price Estimator</p>
    </div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# INPUT FORM
# ═══════════════════════════════════════════════════════════════
with st.form("gari_form"):
    st.markdown("### 🚘 Vehicle Details")
    
    c1, c2, c3 = st.columns(3)

    with c1:
        brand = st.selectbox("🏢 Brand", brand_options, index=brand_options.index("Toyota") if "Toyota" in brand_options else 0)
        model_name = st.selectbox("🚗 Model", model_options, index=model_options.index("Corolla") if "Corolla" in model_options else 0)
        year = st.number_input("📅 Model Year", 1990, 2026, 2018)

    with c2:
        fuel = st.selectbox("⛽ Fuel Type", fuel_options, index=fuel_options.index("Petrol") if "Petrol" in fuel_options else 0)
        transmission = st.selectbox("⚙️ Transmission", transmission_options, index=transmission_options.index("Manual") if "Manual" in transmission_options else 0)
        city = st.selectbox("🏙️ City", city_options, index=city_options.index("Lahore") if "Lahore" in city_options else 0)

    with c3:
        mileage = st.number_input("🛣️ Driven (km)", 0, 2_000_000, 50_000, step=1_000)
        engine = st.number_input("🏎️ Engine (CC)", 600, 10_000, 1_300, step=100)

    st.markdown("---")
    asking_price = st.number_input(
        "💰 Seller's Asking Price (PKR)",
        0, 100_000_000, 2_500_000, step=50_000,
        help="Enter the price the seller is asking to see if it's a fair deal"
    )

    submitted = st.form_submit_button("🔍 Maloom Karo Qeemat")

# ═══════════════════════════════════════════════════════════════
# PREDICTION LOGIC & RESULTS
# ═══════════════════════════════════════════════════════════════
if submitted:
    # ── 1. Feature engineering ──
    car_age = 2026 - year
    mileage_per_year = mileage / max(car_age, 1)
    brand_grouped = "Other" if brand in rare_brands else brand
    model_freq = model_freq_lookup.get(model_name, np.mean(list(model_freq_lookup.values())))

    input_dict = {
        "Car_Age": car_age,
        "Mileage_km": mileage,
        "Engine_cc": engine,
        "Mileage_per_year": mileage_per_year,
        "Model_freq": model_freq,
        "City": city,
        "Fuel Type": fuel,
        "Transmission": transmission,
        "Brand_Grouped": brand_grouped,
    }

    df_input = pd.DataFrame([input_dict])

    # ── 2. One-hot encode ──
    df_encoded = pd.get_dummies(
        df_input,
        columns=["City", "Fuel Type", "Transmission", "Brand_Grouped"],
        drop_first=False,
        dtype=int,
    )

    # ── 3. Align columns ──
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]

    # ── 4. Scale numerics ──
    numeric_cols = ["Car_Age", "Mileage_km", "Engine_cc", "Mileage_per_year", "Model_freq"]
    df_encoded[numeric_cols] = scaler.transform(df_encoded[numeric_cols])

    # ── 5. Predict ──
    log_pred = model.predict(df_encoded)[0]
    fair_price = int(np.expm1(log_pred))

    # ── 6. Market median lookup ──
    market_median = fair_price
    try:
        if isinstance(fair_price_df, pd.DataFrame) and "Brand" in fair_price_df.columns:
            median_row = fair_price_df[
                (fair_price_df["Brand"] == brand) & (fair_price_df["Model"] == model_name)
            ]
            if len(median_row) > 0:
                market_median = int(median_row["group_median_price"].values[0])
            else:
                brand_row = fair_price_df[fair_price_df["Brand"] == brand]
                if len(brand_row) > 0:
                    market_median = int(brand_row["group_median_price"].median())
    except Exception:
        market_median = fair_price

    # ── 7. Verdict & Diff ──
    diff = asking_price - fair_price
    diff_pct = (diff / fair_price) * 100 if fair_price > 0 else 0

    # ═══════════════════════════════════════════════════════
    # METRICS DISPLAY (Sleek Flexbox Cards)
    # ═══════════════════════════════════════════════════════
    st.markdown(f"""
        <div class="metrics-container">
            <div class="metric-card primary">
                <div class="metric-title">ML Fair Price</div>
                <div class="metric-value">{format_pkr(fair_price)}</div>
                <div class="metric-sub">({fair_price:,.0f} PKR)</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Market Median</div>
                <div class="metric-value">{format_pkr(market_median)}</div>
                <div class="metric-sub">({market_median:,.0f} PKR)</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Asking Price</div>
                <div class="metric-value">{format_pkr(asking_price)}</div>
                <div class="metric-sub">({asking_price:,.0f} PKR)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # PLOTLY VISUALIZATIONS
    # ═══════════════════════════════════════════════════════
    st.markdown("### 📊 Interactive Analytics")
    pcol1, pcol2 = st.columns([1, 1.2])

    with pcol1:
        # Gauge Chart for Deal Quality
        gauge_color = "#16a34a" if diff_pct < -10 else "#dc2626" if diff_pct > 10 else "#eab308"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=asking_price,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Deal Quality Gauge", 'font': {'size': 18, 'color': '#334155'}},
            delta={'reference': fair_price, 'increasing': {'color': "#dc2626"}, 'decreasing': {'color': "#16a34a"}},
            number={'valueformat': ',.0f', 'prefix': "₨ "},
            gauge={
                'axis': {'range': [None, fair_price * 1.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': gauge_color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e2e8f0",
                'steps': [
                    {'range': [0, fair_price * 0.9], 'color': '#dcfce7'},       # Good Deal Range
                    {'range': [fair_price * 0.9, fair_price * 1.1], 'color': '#fef9c3'}, # Fair Range
                    {'range': [fair_price * 1.1, fair_price * 1.5], 'color': '#fee2e2'}  # Expensive Range
                ],
                'threshold': {
                    'line': {'color': "#1e293b", 'width': 3},
                    'thickness': 0.75,
                    'value': fair_price
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_gauge, width="stretch")  # REPLACED DEPRECATED PARAMETER

    with pcol2:
        # Horizontal Bar Chart for Price Comparison
        categories = ['Asking Price', 'Market Median', 'ML Fair Price']
        values = [asking_price, market_median, fair_price]
        colors = [gauge_color, '#94a3b8', '#01411C']

        fig_bar = go.Figure(go.Bar(
            x=values,
            y=categories,
            orientation='h',
            marker_color=colors,
            text=[format_pkr(v) for v in values],
            textposition='auto',
            textfont=dict(color='white', weight='bold')
        ))
        fig_bar.update_layout(
            title={'text': "Price Breakdown", 'font': {'size': 18, 'color': '#334155'}},
            xaxis_title="Price (PKR)",
            xaxis=dict(showgrid=True, gridcolor='#f1f5f9'),
            yaxis=dict(showgrid=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=300,
            margin=dict(l=0, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_bar, width="stretch")  # REPLACED DEPRECATED PARAMETER

    # ═══════════════════════════════════════════════════════
    # VERDICT BANNER
    # ═══════════════════════════════════════════════════════
    if diff_pct < -10:
        verdict_html = '<div class="verdict-box verdict-under">🔥 SASTI GARI — Great deal, pakki karo!</div>'
    elif diff_pct > 10:
        verdict_html = '<div class="verdict-box verdict-over">⚠️ MEHENGI GARI — Bargain hard or skip it.</div>'
    else:
        verdict_html = '<div class="verdict-box verdict-fair">✅ THEEK QEEMAT — Fair market price.</div>'

    st.markdown(verdict_html, unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # INSIGHTS
    # ═══════════════════════════════════════════════════════
    st.markdown("### 💡 Pakistan Market Insights")

    tips = []
    if brand in ["Toyota", "Honda", "Suzuki"]:
        tips.append("🎌 Japanese brands hold the strongest resale value and liquidity in Pakistan.")
    if car_age > 15:
        tips.append("⏳ 15+ saal purani gari — spare parts scarcity and lower resale volume ahead.")
    if mileage > 150_000:
        tips.append("🔧 High mileage — highly recommend getting engine compression & suspension checked.")
    if city == "Karachi":
        tips.append("🏙️ Karachi market has higher supply; prices run slightly lower compared to Punjab.")
    if fair_price > market_median * 1.15:
        tips.append("📈 ML predicts above average median — likely due to low mileage or premium variant.")
    if fair_price < market_median * 0.85:
        tips.append("📉 ML predicts below average median — check for accident history or severe wear.")
    if transmission == "Automatic" and car_age > 10:
        tips.append("⚙️ Old automatic transmission — can be costly to repair/replace in local markets.")

    for tip in tips:
        st.markdown(f'<div class="insight-row">{tip}</div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════
    # EXPANDER: TECHNICAL
    # ═══════════════════════════════════════════════════════
    with st.expander("⚙️ View Technical Input Data"):
        st.write("Active Model Features (Non-Zero values only):")
        
        active_row = df_encoded.iloc[0]
        active_features = active_row[active_row > 0].to_frame(name="Engineered Value")
        
        st.dataframe(active_features, width="stretch")

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown("""
    <div class="footer">
        <strong>GariQeemat 🇵🇰</strong> — Built for Neurofive ML Track | Data sourced from PakWheels<br>
        Powered by XGBoost, Streamlit & Plotly
    </div>
""", unsafe_allow_html=True)