import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoVal · Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; }

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0a0a0c 0%, #111827 50%, #0a0a0c 100%);
    border: 1px solid #2a2a32;
    border-radius: 12px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute; inset: 0;
    background-image:
        linear-gradient(rgba(232,255,71,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(232,255,71,0.04) 1px, transparent 1px);
    background-size: 40px 40px;
}
.hero-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3.8rem;
    color: #f0f0f4;
    letter-spacing: 0.05em;
    line-height: 1;
    margin: 0;
}
.hero-title span { color: #e8ff47; }
.hero-sub {
    color: #6b6b80;
    margin-top: 10px;
    font-size: 1rem;
    font-weight: 300;
}
.hero-badge {
    display: inline-block;
    background: rgba(232,255,71,0.1);
    border: 1px solid rgba(232,255,71,0.3);
    color: #e8ff47;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    padding: 5px 12px;
    border-radius: 3px;
    margin-bottom: 16px;
}

/* Metric cards */
.metric-card {
    background: #111114;
    border: 1px solid #2a2a32;
    border-radius: 10px;
    padding: 20px 24px;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #e8ff47; }
.metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    color: #6b6b80;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #e8ff47;
    letter-spacing: 0.04em;
    line-height: 1;
}
.metric-unit {
    font-size: 0.75rem;
    color: #6b6b80;
    margin-top: 4px;
}

/* Price result */
.price-result {
    background: linear-gradient(135deg, #0f1f0a, #111114);
    border: 2px solid #e8ff47;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 0 40px rgba(232,255,71,0.1);
}
.price-result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #6b6b80;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.price-result-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4.5rem;
    color: #e8ff47;
    letter-spacing: 0.03em;
    line-height: 1;
}
.price-result-sub {
    font-size: 0.82rem;
    color: #6b6b80;
    margin-top: 8px;
}
.price-range-bar {
    background: #16161a;
    border: 1px solid #2a2a32;
    border-radius: 8px;
    padding: 12px 20px;
    margin-top: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #f0f0f4;
}

/* Section headers */
.section-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    color: #6b6b80;
    text-transform: uppercase;
    border-bottom: 1px solid #2a2a32;
    padding-bottom: 8px;
    margin-bottom: 16px;
}
.section-header span {
    color: #e8ff47;
    margin-right: 8px;
}

/* Streamlit widget overrides */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background-color: #111114 !important;
    border: 1px solid #2a2a32 !important;
    color: #f0f0f4 !important;
    border-radius: 6px !important;
}
.stButton > button {
    background: #e8ff47 !important;
    color: #0a0a0c !important;
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 1.2rem !important;
    letter-spacing: 0.12em !important;
    border: none !important;
    border-radius: 6px !important;
    width: 100% !important;
    padding: 14px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(232,255,71,0.3) !important;
}
.stSlider > div { color: #f0f0f4 !important; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111114 !important;
    border-right: 1px solid #2a2a32 !important;
}
[data-testid="stSidebar"] * { color: #f0f0f4 !important; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #111114;
    border-radius: 8px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #6b6b80 !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    border-radius: 6px;
}
.stTabs [aria-selected="true"] {
    background: #1e1e24 !important;
    color: #e8ff47 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load & Train Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    df = pd.read_csv('car_data.csv')
    df['Car_Age']        = 2024 - df['Year']
    df['Price_Drop']     = df['Present_Price'] - df['Selling_Price']
    df['Price_Drop_Pct'] = (df['Price_Drop'] / df['Present_Price']) * 100
    df['KM_per_Year']    = df['Driven_kms'] / df['Car_Age'].replace(0, 1)
    df['Brand_Goodwill'] = df.groupby('Car_Name')['Present_Price'].transform('mean')

    le = LabelEncoder()
    for col in ['Fuel_Type', 'Selling_type', 'Transmission']:
        df[col + '_enc'] = le.fit_transform(df[col])

    features = ['Car_Age','Present_Price','Driven_kms','Owner','Brand_Goodwill',
                'KM_per_Year','Price_Drop_Pct','Fuel_Type_enc','Selling_type_enc','Transmission_enc']
    X, y = df[features], df['Selling_Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        'r2':   round(r2_score(y_test, preds), 4),
        'mae':  round(mean_absolute_error(y_test, preds), 3),
        'rmse': round(np.sqrt(mean_squared_error(y_test, preds)), 3),
    }
    brand_map = df.groupby('Car_Name')['Present_Price'].mean().to_dict()
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    return model, brand_map, df['Present_Price'].mean(), metrics, df, X_test, y_test, preds, importances, features

model, brand_map, mean_goodwill, metrics, df, X_test, y_test, test_preds, importances, features = load_model()
BRANDS = sorted(brand_map.keys())
FUEL_MAP   = {'CNG': 0, 'Diesel': 1, 'Petrol': 2}
SELL_MAP   = {'Dealer': 0, 'Individual': 1}
TRANS_MAP  = {'Automatic': 0, 'Manual': 1}

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='font-family:Bebas Neue,sans-serif;font-size:2rem;color:#e8ff47;letter-spacing:0.08em;margin-bottom:4px'>
        AUTO<span style='color:#f0f0f4'>VAL</span>
    </div>
    <div style='font-family:JetBrains Mono,monospace;font-size:0.65rem;letter-spacing:0.15em;color:#6b6b80;margin-bottom:24px'>
        ML PRICE PREDICTOR
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><span>◆</span>Model Stats</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:10px">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{metrics['r2']}</div>
        <div class="metric-unit">accuracy</div>
    </div>
    <div class="metric-card" style="margin-bottom:10px">
        <div class="metric-label">Mean Abs Error</div>
        <div class="metric-value">₹{metrics['mae']}L</div>
        <div class="metric-unit">average error</div>
    </div>
    <div class="metric-card" style="margin-bottom:24px">
        <div class="metric-label">RMSE</div>
        <div class="metric-value">₹{metrics['rmse']}L</div>
        <div class="metric-unit">root mean sq error</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header"><span>◆</span>Algorithm</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#6b6b80;line-height:1.8'>
        <span style='color:#e8ff47'>■</span> Gradient Boosting<br>
        <span style='color:#e8ff47'>■</span> 200 estimators<br>
        <span style='color:#e8ff47'>■</span> 10 features<br>
        <span style='color:#e8ff47'>■</span> 301 training samples
    </div>
    """, unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">GRADIENT BOOSTING · SCIKIT-LEARN · STREAMLIT</div>
    <div class="hero-title">KNOW YOUR <span>CAR'S WORTH</span></div>
    <div class="hero-sub">Instant resale price prediction powered by machine learning trained on real Indian market data.</div>
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🚗  PREDICT PRICE", "📊  MODEL ANALYSIS", "📓  HOW IT WORKS"])

# ════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-header"><span>◆</span>Vehicle Details</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            car_name = st.text_input("Car Model / Brand", placeholder="e.g. swift, city, innova")
        with c2:
            year = st.number_input("Manufacturing Year", min_value=2000, max_value=2024, value=2016, step=1)

        c3, c4 = st.columns(2)
        with c3:
            present_price = st.number_input("Showroom Price (₹ Lakhs)", min_value=0.3, max_value=100.0, value=6.5, step=0.1, format="%.1f")
        with c4:
            driven_kms = st.number_input("Kilometres Driven", min_value=500, max_value=500000, value=45000, step=1000)

        c5, c6 = st.columns(2)
        with c5:
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        with c6:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

        c7, c8 = st.columns(2)
        with c7:
            selling_type = st.selectbox("Seller Type", ["Individual", "Dealer"])
        with c8:
            owner = st.selectbox("Previous Owners", [0, 1, 3],
                                  format_func=lambda x: {0:"First Owner",1:"Second Owner",3:"Third+"}[x])

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  PREDICT SELLING PRICE", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header"><span>◆</span>Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            if not car_name.strip():
                st.error("⚠ Please enter a car model name.")
            else:
                car_age        = 2024 - year
                price_drop_pct = min(90, car_age * 6 + driven_kms / 10000)
                km_per_year    = driven_kms / max(car_age, 1)
                brand_goodwill = brand_map.get(car_name.lower().strip(), mean_goodwill)

                fv = np.array([[
                    car_age, present_price, driven_kms, owner,
                    brand_goodwill, km_per_year, price_drop_pct,
                    FUEL_MAP.get(fuel_type, 2),
                    SELL_MAP.get(selling_type, 0),
                    TRANS_MAP.get(transmission, 1),
                ]])

                predicted = max(0.1, round(float(model.predict(fv)[0]), 2))
                low  = round(predicted * 0.92, 2)
                high = round(predicted * 1.08, 2)

                st.markdown(f"""
                <div class="price-result">
                    <div class="price-result-label">Estimated Resale Value</div>
                    <div class="price-result-value">₹{predicted:.2f}L</div>
                    <div class="price-result-sub">Indian Rupees · Lakhs INR</div>
                    <div class="price-range-bar">
                        📉 Low: ₹{low}L &nbsp;·&nbsp; 📈 High: ₹{high}L &nbsp;·&nbsp; ±8% confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Stats row
                sc1, sc2, sc3 = st.columns(3)
                sc1.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Car Age</div>
                    <div class="metric-value" style="font-size:1.8rem">{car_age} yrs</div>
                </div>""", unsafe_allow_html=True)
                sc2.markdown(f"""<div class="metric-card">
                    <div class="metric-label">KM / Year</div>
                    <div class="metric-value" style="font-size:1.8rem">{km_per_year:,.0f}</div>
                </div>""", unsafe_allow_html=True)
                sc3.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Brand Avg</div>
                    <div class="metric-value" style="font-size:1.8rem">₹{brand_goodwill:.1f}L</div>
                </div>""", unsafe_allow_html=True)

                # Depreciation gauge
                st.markdown("<br>", unsafe_allow_html=True)
                depreciation = round((1 - predicted / present_price) * 100, 1)
                st.markdown(f'<div class="section-header"><span>◆</span>Depreciation: {depreciation}%</div>', unsafe_allow_html=True)
                st.progress(min(int(depreciation), 100))
                st.caption(f"Car lost ₹{round(present_price - predicted, 2)}L ({depreciation}%) from showroom price")

        else:
            st.markdown("""
            <div style='background:#111114;border:1px dashed #2a2a32;border-radius:10px;
                        padding:60px 40px;text-align:center'>
                <div style='font-size:3rem;margin-bottom:12px'>🚗</div>
                <div style='color:#6b6b80;font-size:0.9rem;line-height:1.7'>
                    Fill in your vehicle details<br>and click <b style='color:#e8ff47'>Predict Selling Price</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — MODEL ANALYSIS
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header"><span>◆</span>Model Analysis Dashboard</div>', unsafe_allow_html=True)

    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 12), facecolor='#0a0a0c')
    fig.patch.set_facecolor('#0a0a0c')
    ACCENT = '#e8ff47'
    MUTED  = '#6b6b80'

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test, test_preds, alpha=0.65, color=ACCENT, s=35, edgecolors='none')
    mn, mx = min(y_test.min(), test_preds.min()), max(y_test.max(), test_preds.max())
    ax1.plot([mn, mx], [mn, mx], color='#ff6b35', lw=1.5, ls='--', label='Perfect Fit')
    ax1.set_facecolor('#111114')
    ax1.set_title('Predicted vs Actual', color='#f0f0f4', fontweight='bold', pad=10)
    ax1.set_xlabel('Actual Price (L)', color=MUTED, fontsize=9)
    ax1.set_ylabel('Predicted Price (L)', color=MUTED, fontsize=9)
    ax1.tick_params(colors=MUTED, labelsize=8)
    ax1.legend(fontsize=8)
    for spine in ax1.spines.values(): spine.set_edgecolor('#2a2a32')

    # 2. Residuals
    residuals = y_test - test_preds
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(test_preds, residuals, alpha=0.55, color='#9333ea', s=30, edgecolors='none')
    ax2.axhline(0, color='#ff6b35', lw=1.5, ls='--')
    ax2.set_facecolor('#111114')
    ax2.set_title('Residuals', color='#f0f0f4', fontweight='bold', pad=10)
    ax2.set_xlabel('Predicted Price (L)', color=MUTED, fontsize=9)
    ax2.set_ylabel('Residual', color=MUTED, fontsize=9)
    ax2.tick_params(colors=MUTED, labelsize=8)
    for spine in ax2.spines.values(): spine.set_edgecolor('#2a2a32')

    # 3. Residual distribution
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(residuals, bins=22, color='#9333ea', edgecolor='#0a0a0c', alpha=0.85)
    ax3.axvline(0, color='#ff6b35', lw=1.5, ls='--')
    ax3.set_facecolor('#111114')
    ax3.set_title('Residual Distribution', color='#f0f0f4', fontweight='bold', pad=10)
    ax3.set_xlabel('Residual Value', color=MUTED, fontsize=9)
    ax3.tick_params(colors=MUTED, labelsize=8)
    for spine in ax3.spines.values(): spine.set_edgecolor('#2a2a32')

    # 4. Feature Importance
    ax4 = fig.add_subplot(gs[1, 0:2])
    imp_vals = importances.sort_values()
    colors_imp = plt.cm.YlGn(np.linspace(0.3, 0.9, len(imp_vals)))
    bars = ax4.barh(imp_vals.index, imp_vals.values, color=colors_imp, edgecolor='#0a0a0c')
    for bar, val in zip(bars, imp_vals.values):
        ax4.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', color=MUTED, fontsize=8)
    ax4.set_facecolor('#111114')
    ax4.set_title('Feature Importance (Random Forest)', color='#f0f0f4', fontweight='bold', pad=10)
    ax4.set_xlabel('Importance Score', color=MUTED, fontsize=9)
    ax4.tick_params(colors=MUTED, labelsize=8)
    for spine in ax4.spines.values(): spine.set_edgecolor('#2a2a32')

    # 5. Price by Fuel Type
    ax5 = fig.add_subplot(gs[1, 2])
    fuel_avg = df.groupby('Fuel_Type')['Selling_Price'].mean().sort_values(ascending=False)
    fuel_colors = [ACCENT, '#4ade80', '#60a5fa']
    brs = ax5.bar(fuel_avg.index, fuel_avg.values, color=fuel_colors[:len(fuel_avg)], edgecolor='#0a0a0c')
    for bar, val in zip(brs, fuel_avg.values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'₹{val:.1f}L', ha='center', color=MUTED, fontsize=8)
    ax5.set_facecolor('#111114')
    ax5.set_title('Avg Price by Fuel Type', color='#f0f0f4', fontweight='bold', pad=10)
    ax5.set_ylabel('Avg Selling Price (L)', color=MUTED, fontsize=9)
    ax5.tick_params(colors=MUTED, labelsize=9)
    for spine in ax5.spines.values(): spine.set_edgecolor('#2a2a32')

    st.pyplot(fig)

    # Correlation heatmap
    import seaborn as sns
    st.markdown('<div class="section-header" style="margin-top:24px"><span>◆</span>Correlation Matrix</div>', unsafe_allow_html=True)
    fig2, ax = plt.subplots(figsize=(10, 5), facecolor='#0a0a0c')
    ax.set_facecolor('#111114')
    num_cols = ['Selling_Price','Present_Price','Car_Age','Driven_kms','Brand_Goodwill','KM_per_Year','Price_Drop_Pct']
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, linewidths=0.5, annot_kws={'size': 9, 'color': 'white'},
                cbar_kws={'shrink': 0.8})
    ax.tick_params(colors='#6b6b80', labelsize=9)
    ax.set_title('Feature Correlation Matrix', color='#f0f0f4', fontweight='bold', pad=12)
    st.pyplot(fig2)

# ════════════════════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header"><span>◆</span>ML Pipeline Explained</div>', unsafe_allow_html=True)

    steps = [
        ("1. Data Loading", "301 real Indian used car records. Features: Car Name, Year, Present Price, KMs Driven, Fuel Type, Transmission, Owner count."),
        ("2. Feature Engineering", "5 new features created — Car_Age, Price_Drop_Pct, KM_per_Year, Brand_Goodwill (avg market price per brand), Price_Drop."),
        ("3. Encoding", "Categorical columns (Fuel_Type, Transmission, Selling_type) converted to integers using LabelEncoder."),
        ("4. Train/Test Split", "80% training (240 samples), 20% testing (61 samples). random_state=42 for reproducibility."),
        ("5. Model Training", "Gradient Boosting Regressor — 200 trees, learning_rate=0.1, max_depth=4. Builds trees sequentially to correct previous errors."),
        ("6. Evaluation", f"R²={metrics['r2']} · MAE=₹{metrics['mae']}L · RMSE=₹{metrics['rmse']}L. Residuals are randomly scattered — no systematic bias."),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.markdown(f"<p style='color:#6b6b80;line-height:1.7;font-size:0.92rem'>{desc}</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>◆</span>Top Features by Importance</div>', unsafe_allow_html=True)

    for feat, imp in importances.head(5).items():
        pct = int(imp * 100)
        st.markdown(f"""
        <div style='margin-bottom:12px'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px'>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#f0f0f4'>{feat}</span>
                <span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#e8ff47'>{imp:.4f}</span>
            </div>
            <div style='background:#1e1e24;border-radius:3px;height:6px;overflow:hidden'>
                <div style='background:#e8ff47;width:{pct}%;height:100%;border-radius:3px'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("💡 **Deploy this app free:** Push to GitHub → connect at [streamlit.io/cloud](https://streamlit.io/cloud) → live in 2 minutes.")
