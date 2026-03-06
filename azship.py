import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from math import radians, sin, cos, sqrt, atan2

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Shipment Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── DARK THEME BASE ── */
html, body, .stApp, [data-testid="stAppViewContainer"],
[data-testid="stMain"], [data-testid="block-container"] {
    background-color: #0d0d0d !important;
    color: #f0f0f0 !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Sidebar dark */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div {
    background-color: #1a1a1a !important;
}

/* Force ALL text white */
h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
p, span, label, li, td, th, div { color: #f0f0f0 !important; }

/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background-color: #1a1a1a !important;
    border-radius: 8px;
    padding: 4px;
}
button[data-baseweb="tab"] {
    background-color: transparent !important;
    color: #aaaaaa !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #4361ee !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

/* Selectbox / multiselect */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background-color: #1e1e1e !important;
    border: 1px solid #333333 !important;
    color: #f0f0f0 !important;
}
[data-testid="stSelectbox"] span,
[data-testid="stMultiSelect"] span {
    color: #f0f0f0 !important;
}

/* Slider */
[data-testid="stSlider"] { color: #f0f0f0 !important; }

/* File uploader */
[data-testid="stFileUploader"] section {
    background-color: #1a1a2e !important;
    border: 2px dashed #4361ee !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] div {
    color: #f0f0f0 !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button span {
    background-color: #4361ee !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] * { color: #f0f0f0 !important; }
[data-testid="stDataFrame"] { background-color: #1a1a1a !important; }

/* Divider */
hr { border-color: #333333 !important; }

/* Markdown text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span {
    color: #f0f0f0 !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p { color: #aaaaaa !important; }

.banner {
    background: #2a1f00;
    border: 2px solid #f59e0b;
    border-radius: 10px;
    padding: 14px 20px;
    margin-bottom: 18px;
    display: flex;
    align-items: flex-start;
    gap: 12px;
}
.banner-icon { font-size: 1.4rem; flex-shrink: 0; margin-top: 2px; }
.banner-text { font-size: 0.88rem; color: #fde68a !important; line-height: 1.55; }
.banner-text a { color: #fbbf24 !important; font-weight: 600; }
.banner-title { font-weight: 700; font-size: 0.95rem; color: #fde68a !important; margin-bottom: 3px; }

.kpi-card {
    background: #1a1a2e;
    border: 1px solid #2e2e4e;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 6px;
}
.kpi-value { font-size: 2rem; font-weight: 700; color: #ffffff !important; line-height: 1; }
.kpi-label { font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #aaaaaa !important; margin-top: 5px; }

.section-title {
    font-size: 1rem; font-weight: 600; color: #ffffff !important;
    border-left: 3px solid #4361ee; padding-left: 10px;
    margin: 20px 0 12px 0;
}

.rec-card {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-left: 4px solid #4361ee;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
}
.rec-card.orange { border-left-color: #f77f00; }
.rec-card.green  { border-left-color: #2dc653; }
.rec-card.purple { border-left-color: #9b5de5; }
.rec-card.red    { border-left-color: #ff4d6d; }
.rec-header { font-weight: 700; font-size: 0.95rem; color: #ffffff !important; margin-bottom: 5px; }
.rec-body { font-size: 0.85rem; color: #cccccc !important; line-height: 1.6; }
.rec-body b { color: #ffffff !important; }
.rec-tag {
    display: inline-block;
    background: #1e2a6e; color: #a5b4fc !important;
    font-size: 0.72rem; font-weight: 600;
    padding: 2px 8px; border-radius: 20px;
    margin-bottom: 8px;
}
.rec-tag.orange { background: #3a1f00; color: #ffb347 !important; }
.rec-tag.green  { background: #0a2e14; color: #4ade80 !important; }
.rec-tag.purple { background: #2a1050; color: #c084fc !important; }
.rec-tag.red    { background: #3a0010; color: #ff8096 !important; }

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ────────────────────────────────────────────────────────────────
CITY_COORDS = {
    "MUMBAI": (19.076, 72.877), "PUNE": (18.520, 73.856),
    "BENGALURU": (12.971, 77.594), "BANGALORE": (12.971, 77.594),
    "BANGLORE": (12.971, 77.594), "HYDERABAD": (17.385, 78.487),
    "CHENNAI": (13.083, 80.270), "NEW DELHI": (28.613, 77.209),
    "DELHI": (28.613, 77.209), "NOIDA": (28.535, 77.391),
    "GURUGRAM": (28.459, 77.026), "GURGAON": (28.459, 77.026),
    "AHMEDABAD": (23.023, 72.572), "KOLKATA": (22.572, 88.364),
    "JAIPUR": (26.912, 75.787), "SURAT": (21.170, 72.831),
    "THANE": (19.218, 72.978), "NAVI MUMBAI": (19.033, 73.029),
    "SECUNDERABAD": (17.444, 78.498), "BHUBANESWAR": (20.296, 85.825),
    "LUCKNOW": (26.847, 80.947), "INDORE": (22.719, 75.857),
    "CHANDIGARH": (30.733, 76.779), "KOCHI": (9.931, 76.267),
    "THIRUVANANTHAPURAM": (8.524, 76.936), "COIMBATORE": (11.017, 76.956),
    "NAGPUR": (21.145, 79.088), "PATNA": (25.594, 85.137),
    "BHOPAL": (23.259, 77.412), "VADODARA": (22.307, 73.181),
    "VISAKHAPATNAM": (17.686, 83.218), "AGRA": (27.176, 78.008),
    "ALIGARH": (27.884, 78.082), "HALDWANI": (29.219, 79.513),
    "SOLAN": (30.908, 77.098), "DHARMSALA": (32.219, 76.323),
    "KAVALI": (14.917, 79.994), "TANUKU": (16.861, 81.681),
    "ERODE": (11.341, 77.717), "KALYAN": (19.243, 73.135),
    "ANAND": (22.557, 72.951), "PIMPRI CHINCHWAD": (18.628, 73.804),
    "ABU ROAD": (24.479, 72.779), "PUDUKKOTTAI": (10.380, 78.820),
    "NANDED": (19.161, 77.308), "AURANGABAD": (19.877, 75.343),
    "NASHIK": (19.998, 73.790), "AMRITSAR": (31.634, 74.873),
    "LUDHIANA": (30.901, 75.857), "DEHRADUN": (30.316, 78.032),
    "KARJAT RAIGARH DISTRICT": (18.911, 73.326),
    "RAJKOT": (22.303, 70.802), "MEERUT": (28.984, 77.706),
    "VARANASI": (25.317, 82.973), "JODHPUR": (26.292, 73.017),
    "GUNTUR": (16.307, 80.436), "VIJAYAWADA": (16.506, 80.648),
    "MANGALURU": (12.914, 74.856), "HUBLI": (15.361, 75.124),
    "MYSURU": (12.295, 76.644), "TIRUPPUR": (11.104, 77.341),
    "MADURAI": (9.925, 78.119), "TIRUCHIRAPPALLI": (10.790, 78.704),
    "SALEM": (11.664, 78.146), "VELLORE": (12.916, 79.133),
    "NELLORE": (14.443, 79.987), "KURNOOL": (15.828, 78.037),
    "WARANGAL": (17.977, 79.598), "RAJAHMUNDRY": (17.005, 81.777),
    "KOLHAPUR": (16.705, 74.243), "SOLAPUR": (17.686, 75.904),
    "GWALIOR": (26.218, 78.182), "JABALPUR": (23.181, 79.987),
    "RAIPUR": (21.251, 81.629), "RANCHI": (23.344, 85.310),
    "JAMSHEDPUR": (22.802, 86.185), "GUWAHATI": (26.144, 91.736),
    "SILIGURI": (26.727, 88.432), "HOWRAH": (22.587, 88.264),
    "JAMMU": (32.726, 74.857), "SHIMLA": (31.104, 77.173),
    "UDAIPUR": (24.585, 73.712), "KOTA": (25.182, 75.839),
    "KANPUR": (26.450, 80.331), "GORAKHPUR": (26.760, 83.373),
    "MORADABAD": (28.839, 78.776), "BAREILLY": (28.347, 79.420),
    "MATHURA": (27.492, 77.673), "FIROZABAD": (27.150, 78.395),
}

FC_COORDS = {
    "BOM5": (19.076, 72.877, "Mumbai"),
    "BOM7": (19.200, 72.970, "Mumbai"),
    "BLR7": (12.971, 77.594, "Bengaluru"),
    "BLR8": (13.050, 77.650, "Bengaluru"),
    "DEL4": (28.613, 77.209, "Delhi"),
    "DEL5": (28.700, 77.100, "Delhi"),
}

COLORS = ["#4361ee", "#f72585", "#4cc9f0", "#f77f00", "#2dc653", "#9b5de5", "#fee440", "#ef233c"]

LAYOUT = dict(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#111111",
    font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color="#f0f0f0", size=12),
    margin=dict(l=16, r=16, t=40, b=16), colorway=COLORS,
    xaxis=dict(gridcolor="#2a2a2a", linecolor="#333333", tickfont=dict(size=11, color="#cccccc")),
    yaxis=dict(gridcolor="#2a2a2a", linecolor="#333333", tickfont=dict(size=11, color="#cccccc")),
    legend=dict(bgcolor="rgba(20,20,20,0.9)", bordercolor="#333333", borderwidth=1, font=dict(color="#f0f0f0")),
    title_font=dict(color="#ffffff"),
)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

# ─── Data Loading ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    df["City"] = df["Shipment To City"].str.upper().str.strip()
    df["State"] = df["Shipment To State"].str.upper().str.strip()
    df["Date"] = pd.to_datetime(df["Customer Shipment Date"], utc=True).dt.normalize().dt.tz_localize(None)
    df["Revenue"] = df["Product Amount"] + df["Shipping Amount"]
    df["SKU"] = df["Merchant SKU"].str.strip()
    df["FC"] = df["FC"].str.strip()
    return df

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📦 Shipment Intel")
    st.divider()
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Amazon 30-day shipment export")
    st.divider()

# ─── Always-visible banner ───────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <div class="banner-icon">📋</div>
  <div class="banner-text">
    <div class="banner-title">How to get your data</div>
    Go to <b>Seller Central → Reports → Shipment Sales</b>. Download the
    <b>Customer Shipment Sales CSV</b> for the last 30 days, then upload it using the sidebar.<br>
    <a href="https://sellercentral.amazon.in/reportcentral/SHIPMENT_SALES/1" target="_blank">
      ↗ Open Seller Central — Shipment Sales Report
    </a>
  </div>
</div>
""", unsafe_allow_html=True)

if uploaded is None:
    st.title("📦 Amazon Shipment Intelligence")
    st.info("Upload your CSV using the sidebar to get started.")
    st.markdown("""
    **This app shows you:**
    - Which SKUs ship to which cities (and how many)
    - Which cities have the highest sales per SKU
    - Which warehouses serve which cities — with a map
    - 30-day daily & weekly shipment trends
    - India map with city-level SKU breakdown on hover
    - AI-style recommendations: where to open warehouses, which SKUs to push where
    """)
    st.stop()

df = load_data(uploaded)

with st.sidebar:
    all_skus = sorted(df["SKU"].unique())
    sel_skus = st.multiselect("Filter SKUs", all_skus, default=all_skus)
    all_fcs = sorted(df["FC"].unique())
    sel_fcs = st.multiselect("Filter Warehouses (FC)", all_fcs, default=all_fcs)
    date_min, date_max = df["Date"].min(), df["Date"].max()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                                min_value=date_min, max_value=date_max)
    top_n = st.slider("Top N cities", 5, 25, 10)

fdf = df[
    df["SKU"].isin(sel_skus) &
    df["FC"].isin(sel_fcs) &
    (df["Date"] >= pd.Timestamp(date_range[0])) &
    (df["Date"] <= pd.Timestamp(date_range[1]))
]

# ─── KPI Row ──────────────────────────────────────────────────────────────────
st.markdown("## 📦 Amazon Shipment Intelligence")
k1, k2, k3, k4, k5 = st.columns(5)
top_sku = fdf["SKU"].value_counts().idxmax() if len(fdf) else "—"

for col, val, label in zip(
    [k1, k2, k3, k4, k5],
    [f"{len(fdf):,}", fdf["City"].nunique(), fdf["SKU"].nunique(), fdf["FC"].nunique(), top_sku],
    ["Total Orders", "Cities Reached", "Active SKUs", "Warehouses", "Best Seller SKU"]
):
    col.markdown(f"""<div class='kpi-card'>
    <div class='kpi-value'>{val}</div>
    <div class='kpi-label'>{label}</div></div>""", unsafe_allow_html=True)

st.divider()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏙️ SKU → Cities",
    "📊 City → SKUs",
    "🏭 Warehouse Coverage",
    "📈 30-Day Trend",
    "🗺️ India Map",
    "💡 Recommendations",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SKU → Cities
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-title'>Which cities does each SKU ship to most?</div>", unsafe_allow_html=True)
    selected_sku = st.selectbox("Select SKU", ["ALL"] + sorted(fdf["SKU"].unique()), key="t1_sku")
    sku_city = fdf.groupby(["SKU", "City"]).size().reset_index(name="Orders")

    if selected_sku == "ALL":
        plot_data = sku_city.groupby("City")["Orders"].sum().reset_index().nlargest(top_n, "Orders")
        title = f"Top {top_n} cities — all SKUs"
    else:
        plot_data = sku_city[sku_city["SKU"] == selected_sku].nlargest(top_n, "Orders")
        title = f"Top {top_n} cities for {selected_sku}"

    fig1 = px.bar(plot_data.sort_values("Orders"), x="Orders", y="City", orientation="h",
                  title=title, color="Orders", color_continuous_scale=["#c7d2fe", "#4361ee"])
    fig1.update_layout(**LAYOUT)
    fig1.update_coloraxes(showscale=False)
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("<div class='section-title'>SKU × City Heatmap (orders)</div>", unsafe_allow_html=True)
    top_cities = fdf["City"].value_counts().nlargest(top_n).index
    heat = fdf[fdf["City"].isin(top_cities)].groupby(["SKU", "City"]).size().unstack(fill_value=0)
    heat = heat.reindex(columns=top_cities)
    fig2 = px.imshow(heat, color_continuous_scale=["#f0f4ff", "#4361ee"], aspect="auto",
                     title="Orders: SKU (rows) × City (columns)")
    fig2.update_layout(**LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — City → SKUs
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-title'>Which SKUs sell most in each city?</div>", unsafe_allow_html=True)
    top_cities_list = fdf["City"].value_counts().nlargest(top_n).index.tolist()
    selected_city = st.selectbox("Select City", ["TOP CITIES"] + sorted(top_cities_list), key="t2_city")

    if selected_city == "TOP CITIES":
        city_sku = fdf[fdf["City"].isin(top_cities_list)].groupby(["City", "SKU"]).size().reset_index(name="Orders")
        fig3 = px.bar(city_sku, x="City", y="Orders", color="SKU",
                      title=f"SKU breakdown — top {top_n} cities",
                      color_discrete_sequence=COLORS, barmode="stack")
    else:
        city_sku = fdf[fdf["City"] == selected_city].groupby("SKU").size().reset_index(name="Orders").sort_values("Orders", ascending=False)
        fig3 = px.bar(city_sku, x="SKU", y="Orders", title=f"SKU sales in {selected_city}",
                      color="SKU", color_discrete_sequence=COLORS)
    fig3.update_layout(**LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-title'>Top SKU per city</div>", unsafe_allow_html=True)
    city_top = (fdf.groupby(["City", "SKU"]).size().reset_index(name="Orders")
                .sort_values("Orders", ascending=False).drop_duplicates("City")
                .sort_values("Orders", ascending=False).head(30).reset_index(drop=True))
    city_top.columns = ["City", "Top SKU", "Orders"]
    st.dataframe(city_top, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Warehouse Coverage
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-title'>Warehouse shipment volume & SKU mix</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fc_totals = fdf.groupby("FC").size().reset_index(name="Orders").sort_values("Orders", ascending=False)
        fig4 = px.bar(fc_totals, x="FC", y="Orders", title="Total shipments per warehouse",
                      color="FC", color_discrete_sequence=COLORS)
        fig4.update_layout(**LAYOUT, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)
    with col_b:
        fc_sku = fdf.groupby(["FC", "SKU"]).size().reset_index(name="Orders")
        fig5 = px.bar(fc_sku, x="FC", y="Orders", color="SKU", title="SKU mix per warehouse",
                      color_discrete_sequence=COLORS, barmode="stack")
        fig5.update_layout(**LAYOUT)
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<div class='section-title'>Warehouse → City map (lines show which FC ships to which city)</div>", unsafe_allow_html=True)
    sel_fc_map = st.selectbox("Highlight a warehouse", ["ALL"] + sorted(fdf["FC"].unique()), key="t3_fc_map")
    fc_city_data = fdf.groupby(["FC", "City"]).size().reset_index(name="Orders")
    if sel_fc_map != "ALL":
        fc_city_data = fc_city_data[fc_city_data["FC"] == sel_fc_map]
    fc_city_data["lat"] = fc_city_data["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    fc_city_data["lon"] = fc_city_data["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    fc_city_mapped = fc_city_data.dropna(subset=["lat", "lon"])
    fc_colors = {fc: COLORS[i % len(COLORS)] for i, fc in enumerate(sorted(fdf["FC"].unique()))}

    fig_wmap = go.Figure()
    for _, row in fc_city_mapped.iterrows():
        fc_info = FC_COORDS.get(row["FC"])
        if fc_info:
            fig_wmap.add_trace(go.Scattergeo(
                lat=[fc_info[0], row["lat"]], lon=[fc_info[1], row["lon"]], mode="lines",
                line=dict(width=max(0.5, row["Orders"] * 0.4), color=fc_colors.get(row["FC"], "#aaa")),
                opacity=0.3, showlegend=False, hoverinfo="skip"))
    for fc in fc_city_mapped["FC"].unique():
        sub = fc_city_mapped[fc_city_mapped["FC"] == fc]
        fig_wmap.add_trace(go.Scattergeo(
            lat=sub["lat"], lon=sub["lon"], mode="markers", name=f"{fc}",
            marker=dict(size=np.sqrt(sub["Orders"]) * 5 + 7, color=fc_colors.get(fc, "#aaa"),
                        opacity=0.85, line=dict(width=1, color="white")),
            text=sub["City"], customdata=sub[["FC", "Orders"]].values,
            hovertemplate="<b>%{text}</b><br>Warehouse: %{customdata[0]}<br>Orders: %{customdata[1]}<extra></extra>"))
    for fc, (flat, flon, fname) in FC_COORDS.items():
        if sel_fc_map != "ALL" and fc != sel_fc_map:
            continue
        fig_wmap.add_trace(go.Scattergeo(
            lat=[flat], lon=[flon], mode="markers+text", name=f"⭐ {fc}",
            marker=dict(size=18, color=fc_colors.get(fc, "#333"), symbol="star",
                        line=dict(width=2, color="white")),
            text=[fc], textposition="top center", textfont=dict(size=11, color="#1a1a2e"),
            hovertemplate=f"<b>Warehouse: {fc}</b><br>Location: {fname}<extra></extra>"))
    fig_wmap.update_layout(
        paper_bgcolor="#0d0d0d",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color="#1a1a2e"),
        geo=dict(scope="asia", center=dict(lat=22, lon=80), projection_scale=4.5,
                 bgcolor="#0a0a1a", landcolor="#1a2a1a", oceancolor="#0a1520",
                 lakecolor="#0a1520", subunitcolor="#444444", countrycolor="#555555",
                 showland=True, showocean=True, showlakes=True, showcountries=True, showsubunits=True),
        legend=dict(bgcolor="rgba(20,20,20,0.9)", bordercolor="#333333", borderwidth=1,
                    title=dict(text="Warehouse")),
        margin=dict(l=0, r=0, t=10, b=0), height=620)
    st.plotly_chart(fig_wmap, use_container_width=True)
    st.caption("⭐ Stars = warehouse locations · Bubbles = delivery cities · Line thickness = order volume · Hover any city to see which FC serves it")

    st.markdown("<div class='section-title'>Warehouse → City detail table</div>", unsafe_allow_html=True)
    sel_fc_tbl = st.selectbox("Filter by FC", ["ALL"] + sorted(fdf["FC"].unique()), key="t3_fc_tbl")
    tbl_data = fdf if sel_fc_tbl == "ALL" else fdf[fdf["FC"] == sel_fc_tbl]
    fc_table = (tbl_data.groupby(["FC", "City", "State"]).size().reset_index(name="Orders")
                .sort_values(["FC", "Orders"], ascending=[True, False]).reset_index(drop=True))
    st.dataframe(fc_table, use_container_width=True, hide_index=True)

    st.markdown("<div class='section-title'>FC → State → City drill-down</div>", unsafe_allow_html=True)
    sun_data = fdf.groupby(["FC", "State", "City"]).size().reset_index(name="Orders")
    fig6 = px.sunburst(sun_data, path=["FC", "State", "City"], values="Orders",
                       color="FC", color_discrete_sequence=COLORS)
    fig6.update_layout(paper_bgcolor="#0d0d0d", margin=dict(l=0, r=0, t=10, b=0), height=500,
                       font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"))
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 30-Day Trend
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-title'>Daily shipments by SKU</div>", unsafe_allow_html=True)
    daily = fdf.groupby(["Date", "SKU"]).size().reset_index(name="Orders")
    daily_total = fdf.groupby("Date").size().reset_index(name="Orders")
    fig7 = px.area(daily, x="Date", y="Orders", color="SKU",
                   title="Daily orders by SKU", color_discrete_sequence=COLORS, line_shape="spline")
    fig7.update_layout(**LAYOUT)
    st.plotly_chart(fig7, use_container_width=True)
    col_c, col_d = st.columns(2)
    with col_c:
        fig8 = px.line(daily_total, x="Date", y="Orders", title="Total daily orders", markers=True)
        fig8.update_traces(line_color="#4361ee", marker_color="#4361ee", line_width=2.5)
        fig8.update_layout(**LAYOUT)
        st.plotly_chart(fig8, use_container_width=True)
    with col_d:
        weekly = fdf.copy()
        weekly["Week"] = weekly["Date"].dt.to_period("W").dt.start_time
        w_sku = weekly.groupby(["Week", "SKU"]).size().reset_index(name="Orders")
        fig9 = px.bar(w_sku, x="Week", y="Orders", color="SKU", title="Weekly orders by SKU",
                      color_discrete_sequence=COLORS, barmode="stack")
        fig9.update_layout(**LAYOUT)
        st.plotly_chart(fig9, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — India Map
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-title'>India shipment map — hover a city for top 3 SKUs & volume</div>", unsafe_allow_html=True)
    city_agg = fdf.groupby("City").size().reset_index(name="Total_Orders")
    city_sku_counts = fdf.groupby(["City", "SKU"]).size().reset_index(name="cnt")

    def top3_text(city):
        sub = city_sku_counts[city_sku_counts["City"] == city].nlargest(3, "cnt")
        return "<br>".join([f"{r['SKU']}: {r['cnt']} orders" for _, r in sub.iterrows()])

    city_agg["Top3"] = city_agg["City"].apply(top3_text)
    city_agg["Top_SKU"] = city_agg["City"].apply(
        lambda c: city_sku_counts[city_sku_counts["City"] == c].nlargest(1, "cnt")["SKU"].values[0]
        if len(city_sku_counts[city_sku_counts["City"] == c]) > 0 else "—")
    city_agg["lat"] = city_agg["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    city_agg["lon"] = city_agg["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    mapped = city_agg.dropna(subset=["lat", "lon"])
    st.caption(f"Mapped {len(mapped)} of {len(city_agg)} cities · Bubble size = order volume · Color = top SKU · Hover to see top 3 SKUs")

    sku_colors = {sku: COLORS[i % len(COLORS)] for i, sku in enumerate(sorted(fdf["SKU"].unique()))}
    fig_map = go.Figure()
    for sku in sorted(mapped["Top_SKU"].unique()):
        sub = mapped[mapped["Top_SKU"] == sku]
        fig_map.add_trace(go.Scattergeo(
            lat=sub["lat"], lon=sub["lon"], mode="markers", name=sku,
            marker=dict(size=np.sqrt(sub["Total_Orders"]) * 6 + 8, color=sku_colors.get(sku, "#aaa"),
                        opacity=0.85, line=dict(width=1, color="white")),
            text=sub["City"], customdata=sub[["Top3", "Total_Orders"]].values,
            hovertemplate="<b>%{text}</b><br>Total Orders: %{customdata[1]}<br><br>Top 3 SKUs:<br>%{customdata[0]}<extra></extra>"))
    fig_map.update_layout(
        paper_bgcolor="#0d0d0d",
        font=dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif", color="#1a1a2e"),
        geo=dict(scope="asia", center=dict(lat=22, lon=80), projection_scale=4.5,
                 bgcolor="#0a0a1a", landcolor="#1a2a1a", oceancolor="#0a1520",
                 lakecolor="#0a1520", subunitcolor="#444444", countrycolor="#555555",
                 showland=True, showocean=True, showlakes=True, showcountries=True, showsubunits=True),
        legend=dict(bgcolor="rgba(20,20,20,0.9)", bordercolor="#333333", title=dict(text="Top SKU")),
        margin=dict(l=0, r=0, t=0, b=0), height=620)
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("<div class='section-title'>City breakdown table</div>", unsafe_allow_html=True)
    display_table = mapped[["City", "Total_Orders", "Top_SKU", "Top3"]].copy()
    display_table = display_table.sort_values("Total_Orders", ascending=False).reset_index(drop=True)
    display_table.columns = ["City", "Total Orders", "Top SKU", "Top 3 SKUs"]
    st.dataframe(display_table, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Recommendations
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 💡 Recommendations")
    st.caption("Based on your last 30 days of shipment data. All insights are derived automatically from the CSV you uploaded.")
    st.divider()

    # ── Helper: min distance from city to nearest FC ──────────────────────────
    def nearest_fc_distance(city):
        coords = CITY_COORDS.get(city)
        if not coords:
            return None, None
        min_dist = float("inf")
        nearest = None
        for fc, (flat, flon, _) in FC_COORDS.items():
            d = haversine_km(coords[0], coords[1], flat, flon)
            if d < min_dist:
                min_dist = d
                nearest = fc
        return nearest, round(min_dist)

    # ── 1. States underserved by warehouses ───────────────────────────────────
    st.markdown("<div class='section-title'>🏗️ States that could use a new warehouse</div>", unsafe_allow_html=True)

    state_orders = fdf.groupby(["State", "City"]).size().reset_index(name="Orders")
    state_orders["lat"] = state_orders["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    state_orders["lon"] = state_orders["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    state_orders = state_orders.dropna(subset=["lat", "lon"])

    if len(state_orders) > 0:
        state_orders[["nearest_fc", "dist_km"]] = state_orders["City"].apply(
            lambda c: pd.Series(nearest_fc_distance(c)))
        state_orders = state_orders.dropna(subset=["dist_km"])

        # Weighted avg distance per state (weighted by orders)
        state_dist = state_orders.groupby("State").apply(
            lambda g: np.average(g["dist_km"], weights=g["Orders"])
        ).reset_index(name="Avg_Dist_km")
        state_vol = fdf.groupby("State").size().reset_index(name="Orders")
        state_summary = state_dist.merge(state_vol, on="State").sort_values("Avg_Dist_km", ascending=False)
        underserved = state_summary[state_summary["Orders"] >= 3].head(5)

        for _, row in underserved.iterrows():
            st.markdown(f"""
            <div class='rec-card orange'>
              <div class='rec-tag orange'>⚠️ Underserved State</div>
              <div class='rec-header'>Consider a warehouse near {row['State'].title()}</div>
              <div class='rec-body'>
                Orders from {row['State'].title()} travel an average of <b>{int(row['Avg_Dist_km'])} km</b> to reach customers
                — the farthest of any state in your network. With <b>{int(row['Orders'])} orders</b> in the last 30 days,
                this state has enough volume to justify closer fulfilment. A warehouse here could
                reduce delivery time and shipping costs significantly.
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Not enough location data to compute warehouse gaps.")

    # ── 2. Cities with high demand but far from any FC ────────────────────────
    st.markdown("<div class='section-title'>🏙️ High-demand cities far from any warehouse</div>", unsafe_allow_html=True)

    city_vol = fdf.groupby("City").size().reset_index(name="Orders")
    city_vol[["nearest_fc", "dist_km"]] = city_vol["City"].apply(
        lambda c: pd.Series(nearest_fc_distance(c)))
    city_vol = city_vol.dropna(subset=["dist_km"])
    # Flag cities with above-median orders AND above-median distance
    median_orders = city_vol["Orders"].median()
    median_dist = city_vol["dist_km"].median()
    gap_cities = city_vol[
        (city_vol["Orders"] > median_orders) &
        (city_vol["dist_km"] > median_dist)
    ].sort_values("dist_km", ascending=False).head(5)

    if len(gap_cities) > 0:
        for _, row in gap_cities.iterrows():
            st.markdown(f"""
            <div class='rec-card red'>
              <div class='rec-tag red'>📍 Coverage Gap</div>
              <div class='rec-header'>{row['City'].title()} — {int(row['Orders'])} orders, {int(row['dist_km'])} km from nearest FC ({row['nearest_fc']})</div>
              <div class='rec-body'>
                {row['City'].title()} places a above-average number of orders but is served from <b>{row['nearest_fc']}</b>,
                which is <b>{int(row['dist_km'])} km away</b>. This likely means longer delivery windows and higher
                shipping costs. Stocking inventory in a closer fulfilment centre — or adding a dark store / 3PL
                in this region — could improve conversion and reduce returns here.
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.success("No major coverage gaps detected — your warehouses serve high-demand cities well.")

    # ── 3. SKUs to stock in more FCs ──────────────────────────────────────────
    st.markdown("<div class='section-title'>📦 SKUs that should be in more warehouses</div>", unsafe_allow_html=True)

    sku_fc = fdf.groupby(["SKU", "FC"]).size().reset_index(name="Orders")
    sku_fc_count = sku_fc.groupby("SKU")["FC"].nunique().reset_index(name="FC_Count")
    sku_total = fdf.groupby("SKU").size().reset_index(name="Total_Orders")
    sku_spread = sku_fc_count.merge(sku_total, on="SKU")
    total_fcs = fdf["FC"].nunique()
    limited_spread = sku_spread[sku_spread["FC_Count"] < total_fcs].sort_values("Total_Orders", ascending=False)

    for _, row in limited_spread.iterrows():
        current_fcs = sorted(sku_fc[sku_fc["SKU"] == row["SKU"]]["FC"].unique())
        missing_fcs = [fc for fc in sorted(fdf["FC"].unique()) if fc not in current_fcs]
        # Find top cities for this SKU that are served by the missing FCs
        sku_city_data = fdf[fdf["SKU"] == row["SKU"]].groupby("City").size().reset_index(name="cnt").nlargest(3, "cnt")
        top_cities_str = ", ".join(sku_city_data["City"].str.title().tolist())
        st.markdown(f"""
        <div class='rec-card purple'>
          <div class='rec-tag purple'>📤 Expand Distribution</div>
          <div class='rec-header'>{row['SKU']} — only in {row['FC_Count']} of {total_fcs} warehouses</div>
          <div class='rec-body'>
            <b>{row['SKU']}</b> has <b>{int(row['Total_Orders'])} orders</b> in this period but is only stocked at
            <b>{", ".join(current_fcs)}</b>. It is missing from: <b>{", ".join(missing_fcs)}</b>.
            Its top delivery cities are <b>{top_cities_str}</b> — some of which may be closer to the missing FCs.
            Spreading inventory to these warehouses could reduce delivery time and lower per-order shipping cost.
          </div>
        </div>""", unsafe_allow_html=True)

    if len(limited_spread) == 0:
        st.success("All SKUs are distributed across all active warehouses — great inventory spread!")

    # ── 4. States with growing demand (week-over-week) ────────────────────────
    st.markdown("<div class='section-title'>📈 States with fastest growing demand</div>", unsafe_allow_html=True)

    if fdf["Date"].nunique() >= 14:
        trend_df = fdf.copy()
        trend_df["Week"] = trend_df["Date"].dt.to_period("W")
        weekly_state = trend_df.groupby(["Week", "State"]).size().reset_index(name="Orders")
        weeks_sorted = sorted(weekly_state["Week"].unique())

        if len(weeks_sorted) >= 2:
            last_week = weeks_sorted[-1]
            prev_week = weeks_sorted[-2]
            lw = weekly_state[weekly_state["Week"] == last_week][["State", "Orders"]].rename(columns={"Orders": "Last_Week"})
            pw = weekly_state[weekly_state["Week"] == prev_week][["State", "Orders"]].rename(columns={"Orders": "Prev_Week"})
            wow = lw.merge(pw, on="State", how="inner")
            wow["Growth_pct"] = ((wow["Last_Week"] - wow["Prev_Week"]) / wow["Prev_Week"].replace(0, 1) * 100).round(1)
            growing = wow[wow["Growth_pct"] > 0].sort_values("Growth_pct", ascending=False).head(5)

            if len(growing) > 0:
                for _, row in growing.iterrows():
                    top_sku_in_state = (fdf[fdf["State"] == row["State"]]
                                        .groupby("SKU").size().idxmax())
                    st.markdown(f"""
                    <div class='rec-card green'>
                      <div class='rec-tag green'>📈 Growing Market</div>
                      <div class='rec-header'>{row['State'].title()} — up {row['Growth_pct']}% week-over-week</div>
                      <div class='rec-body'>
                        Orders from <b>{row['State'].title()}</b> grew from <b>{int(row['Prev_Week'])}</b> to
                        <b>{int(row['Last_Week'])}</b> orders last week (+{row['Growth_pct']}%).
                        The best-selling SKU here is <b>{top_sku_in_state}</b>.
                        Consider increasing stock of <b>{top_sku_in_state}</b> at the FC closest to this state
                        before demand outpaces supply.
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No states showed week-over-week growth in the selected period.")
        else:
            st.info("Need at least 2 weeks of data to compute week-over-week growth.")
    else:
        st.info("Need at least 14 days of data to compute weekly growth trends.")
