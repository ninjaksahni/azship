import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Shipment Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
}

h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

.stApp {
    background: #0a0a0f;
    color: #e8e4d9;
}

section[data-testid="stSidebar"] {
    background: #0f0f18 !important;
    border-right: 1px solid #1e1e2e;
}

.metric-card {
    background: linear-gradient(135deg, #12121e 0%, #1a1a2e 100%);
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 8px;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f0e040;
    line-height: 1;
}

.metric-label {
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #6060a0;
    margin-top: 4px;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #f0e040;
    border-left: 3px solid #f0e040;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

.stSelectbox > div > div {
    background: #12121e !important;
    border: 1px solid #2a2a4a !important;
    color: #e8e4d9 !important;
}

div[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: #f0e040 !important;
}

.upload-box {
    background: linear-gradient(135deg, #12121e, #1a1a2e);
    border: 2px dashed #2a2a4a;
    border-radius: 16px;
    padding: 40px;
    text-align: center;
}

footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── City → Coordinates (major Indian cities) ───────────────────────────────
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
    "NAVI MUMBAI": (19.033, 73.029),
}

# ─── Data Loading & Processing ───────────────────────────────────────────────
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

# ─── Plotly theme ────────────────────────────────────────────────────────────
COLORS = ["#f0e040", "#40e0f0", "#f040a0", "#40f090", "#f09040", "#9040f0", "#40a0f0", "#f06040"]
LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#e8e4d9", size=12),
    margin=dict(l=16, r=16, t=40, b=16),
    colorway=COLORS,
    xaxis=dict(gridcolor="#1e1e2e", linecolor="#2a2a4a", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1e1e2e", linecolor="#2a2a4a", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2a4a"),
)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📦 Shipment Intel")
    st.markdown("---")
    uploaded = st.file_uploader("Upload CSV", type=["csv"], help="Amazon 30-day shipment export")
    st.markdown("---")

if uploaded is None:
    st.markdown("""
    <div class='upload-box'>
        <h2 style='font-family:Syne,sans-serif;color:#f0e040;font-size:1.8rem;margin-bottom:12px'>
            📦 Amazon Shipment Intelligence
        </h2>
        <p style='color:#6060a0;font-size:0.85rem;letter-spacing:0.05em'>
            Upload your 30-day Amazon shipment CSV to explore SKU → city flows,<br>
            warehouse coverage, and demand trends.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

df = load_data(uploaded)

# ─── Sidebar Filters ─────────────────────────────────────────────────────────
with st.sidebar:
    all_skus = sorted(df["SKU"].unique())
    sel_skus = st.multiselect("Filter SKUs", all_skus, default=all_skus)

    all_fcs = sorted(df["FC"].unique())
    sel_fcs = st.multiselect("Filter Warehouses (FC)", all_fcs, default=all_fcs)

    date_min, date_max = df["Date"].min(), df["Date"].max()
    date_range = st.date_input("Date Range", value=(date_min, date_max),
                                min_value=date_min, max_value=date_max)

    top_n = st.slider("Top N cities to show", 5, 25, 10)

# ─── Apply Filters ───────────────────────────────────────────────────────────
fdf = df[
    df["SKU"].isin(sel_skus) &
    df["FC"].isin(sel_fcs) &
    (df["Date"] >= pd.Timestamp(date_range[0])) &
    (df["Date"] <= pd.Timestamp(date_range[1]))
]

# ─── KPI Row ─────────────────────────────────────────────────────────────────
st.markdown("### 📦 Amazon Shipment Intelligence")
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-value'>{len(fdf):,}</div>
    <div class='metric-label'>Total Orders</div></div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-value'>{fdf['City'].nunique()}</div>
    <div class='metric-label'>Cities Reached</div></div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-value'>{fdf['SKU'].nunique()}</div>
    <div class='metric-label'>Active SKUs</div></div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-value'>{fdf['FC'].nunique()}</div>
    <div class='metric-label'>Warehouses</div></div>""", unsafe_allow_html=True)
with k5:
    top_sku = fdf["SKU"].value_counts().idxmax() if len(fdf) else "—"
    st.markdown(f"""<div class='metric-card'>
    <div class='metric-value'>{top_sku}</div>
    <div class='metric-label'>Best Seller SKU</div></div>""", unsafe_allow_html=True)

# ─── Tab Layout ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏙️ SKU → Cities",
    "📊 City → SKUs",
    "🏭 Warehouse Coverage",
    "📈 30-Day Trend",
    "🗺️ India Map",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Top Cities per SKU
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Which cities does each SKU ship to most?</div>", unsafe_allow_html=True)

    sku_city = (
        fdf.groupby(["SKU", "City"])
        .size().reset_index(name="Orders")
        .sort_values("Orders", ascending=False)
    )

    selected_sku = st.selectbox("Select SKU", ["ALL"] + sorted(fdf["SKU"].unique()), key="tab1_sku")

    if selected_sku == "ALL":
        plot_data = sku_city.groupby("City")["Orders"].sum().reset_index().nlargest(top_n, "Orders")
        title = f"Top {top_n} cities — all SKUs combined"
        color_col = None
    else:
        plot_data = sku_city[sku_city["SKU"] == selected_sku].nlargest(top_n, "Orders")
        title = f"Top {top_n} cities for {selected_sku}"
        color_col = None

    fig = px.bar(
        plot_data.sort_values("Orders"),
        x="Orders", y="City", orientation="h",
        title=title,
        color="Orders",
        color_continuous_scale=[[0, "#2a2a4a"], [1, "#f0e040"]],
    )
    fig.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
    fig.update_coloraxes(showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Heatmap all SKUs × top cities
    st.markdown("<div class='section-header'>SKU × City Heatmap</div>", unsafe_allow_html=True)
    top_cities = fdf["City"].value_counts().nlargest(top_n).index
    heat = fdf[fdf["City"].isin(top_cities)].groupby(["SKU", "City"]).size().unstack(fill_value=0)
    heat = heat.reindex(columns=top_cities)

    fig2 = px.imshow(
        heat,
        color_continuous_scale=[[0, "#0a0a1a"], [0.3, "#2a2a6a"], [1, "#f0e040"]],
        aspect="auto",
        title="Orders heatmap: SKU rows × top city columns"
    )
    fig2.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Top SKUs per City
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Which SKUs sell most in each city?</div>", unsafe_allow_html=True)

    top_cities_list = fdf["City"].value_counts().nlargest(top_n).index.tolist()
    selected_city = st.selectbox("Select City", ["TOP 10 CITIES"] + sorted(top_cities_list), key="tab2_city")

    if selected_city == "TOP 10 CITIES":
        city_sku = (
            fdf[fdf["City"].isin(top_cities_list)]
            .groupby(["City", "SKU"]).size().reset_index(name="Orders")
        )
        fig3 = px.bar(
            city_sku,
            x="City", y="Orders", color="SKU",
            title=f"SKU breakdown — top {top_n} cities",
            color_discrete_sequence=COLORS,
            barmode="stack",
        )
    else:
        city_sku = (
            fdf[fdf["City"] == selected_city]
            .groupby("SKU").size().reset_index(name="Orders")
            .sort_values("Orders", ascending=False)
        )
        fig3 = px.bar(
            city_sku,
            x="SKU", y="Orders",
            title=f"SKU sales in {selected_city}",
            color="Orders",
            color_continuous_scale=[[0, "#2a2a4a"], [1, "#40e0f0"]],
        )
        fig3.update_coloraxes(showscale=False)

    fig3.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
    st.plotly_chart(fig3, use_container_width=True)

    # Table: top SKU per city
    st.markdown("<div class='section-header'>Top SKU per city (ranked by volume)</div>", unsafe_allow_html=True)
    city_top = (
        fdf.groupby(["City", "SKU"]).size().reset_index(name="Orders")
        .sort_values("Orders", ascending=False)
        .drop_duplicates("City")
        .sort_values("Orders", ascending=False)
        .rename(columns={"City": "City", "SKU": "Top SKU", "Orders": "Orders"})
        .head(30)
        .reset_index(drop=True)
    )
    st.dataframe(city_top, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Warehouse Coverage
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Which warehouses serve which cities?</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        fc_city = fdf.groupby(["FC", "City"]).size().reset_index(name="Orders")
        fc_totals = fc_city.groupby("FC")["Orders"].sum().reset_index().sort_values("Orders", ascending=False)
        fig4 = px.bar(
            fc_totals,
            x="FC", y="Orders",
            title="Total shipments per warehouse",
            color="FC",
            color_discrete_sequence=COLORS,
        )
        fig4.update_layout(**LAYOUT, showlegend=False,
                           title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
        st.plotly_chart(fig4, use_container_width=True)

    with col_b:
        fc_sku = fdf.groupby(["FC", "SKU"]).size().reset_index(name="Orders")
        fig5 = px.bar(
            fc_sku, x="FC", y="Orders", color="SKU",
            title="SKU mix per warehouse",
            color_discrete_sequence=COLORS,
            barmode="stack",
        )
        fig5.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
        st.plotly_chart(fig5, use_container_width=True)

    # Detailed coverage table
    st.markdown("<div class='section-header'>Warehouse → top cities served</div>", unsafe_allow_html=True)
    sel_fc = st.selectbox("Select FC", ["ALL"] + sorted(fdf["FC"].unique()), key="tab3_fc")

    fc_filter = fdf if sel_fc == "ALL" else fdf[fdf["FC"] == sel_fc]
    fc_table = (
        fc_filter.groupby(["FC", "City", "State"]).size().reset_index(name="Orders")
        .sort_values("Orders", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )
    st.dataframe(fc_table, use_container_width=True, hide_index=True)

    # Sunburst: FC → State → City
    st.markdown("<div class='section-header'>Warehouse → State → City drill-down</div>", unsafe_allow_html=True)
    sun_data = fdf.groupby(["FC", "State", "City"]).size().reset_index(name="Orders")
    fig6 = px.sunburst(
        sun_data, path=["FC", "State", "City"], values="Orders",
        color="FC", color_discrete_sequence=COLORS,
        title="FC → State → City hierarchy"
    )
    fig6.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
    st.plotly_chart(fig6, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — 30-Day Trend
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Daily shipment volume over time</div>", unsafe_allow_html=True)

    daily = fdf.groupby(["Date", "SKU"]).size().reset_index(name="Orders")
    daily_total = fdf.groupby("Date").size().reset_index(name="Orders")

    fig7 = px.area(
        daily, x="Date", y="Orders", color="SKU",
        title="Daily orders by SKU",
        color_discrete_sequence=COLORS,
        line_shape="spline",
    )
    fig7.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
    fig7.update_traces(line_width=2)
    st.plotly_chart(fig7, use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fig8 = px.line(
            daily_total, x="Date", y="Orders",
            title="Total daily orders (all SKUs)",
            markers=True,
        )
        fig8.update_traces(line_color="#f0e040", marker_color="#f0e040", line_width=2.5)
        fig8.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
        st.plotly_chart(fig8, use_container_width=True)

    with col_d:
        weekly = fdf.copy()
        weekly["Week"] = weekly["Date"].dt.to_period("W").dt.start_time
        w_sku = weekly.groupby(["Week", "SKU"]).size().reset_index(name="Orders")
        fig9 = px.bar(
            w_sku, x="Week", y="Orders", color="SKU",
            title="Weekly orders by SKU",
            color_discrete_sequence=COLORS,
            barmode="stack",
        )
        fig9.update_layout(**LAYOUT, title_font=dict(family="Syne, sans-serif", size=14, color="#e8e4d9"))
        st.plotly_chart(fig9, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — India Map
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>India shipment map — hover for top 3 SKUs per city</div>", unsafe_allow_html=True)

    # Build city-level aggregation
    city_agg = fdf.groupby("City").size().reset_index(name="Total_Orders")

    # Top 3 SKUs per city for hover
    city_sku_counts = fdf.groupby(["City", "SKU"]).size().reset_index(name="cnt")

    def top3_sku_text(city):
        sub = city_sku_counts[city_sku_counts["City"] == city].nlargest(3, "cnt")
        lines = [f"{row['SKU']}: {row['cnt']}" for _, row in sub.iterrows()]
        return "<br>".join(lines)

    city_agg["Top3_SKUs"] = city_agg["City"].apply(top3_sku_text)
    city_agg["Top_SKU"] = city_agg["City"].apply(
        lambda c: city_sku_counts[city_sku_counts["City"] == c].nlargest(1, "cnt")["SKU"].values[0]
        if len(city_sku_counts[city_sku_counts["City"] == c]) > 0 else "—"
    )

    # Attach coords
    city_agg["lat"] = city_agg["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[0])
    city_agg["lon"] = city_agg["City"].map(lambda c: CITY_COORDS.get(c, (None, None))[1])
    mapped = city_agg.dropna(subset=["lat", "lon"])

    st.caption(f"Showing {len(mapped)} of {len(city_agg)} cities (coords available). Bubble size = order volume.")

    sku_colors = {sku: COLORS[i % len(COLORS)] for i, sku in enumerate(sorted(fdf["SKU"].unique()))}

    fig_map = go.Figure()

    for sku in sorted(mapped["Top_SKU"].unique()):
        sub = mapped[mapped["Top_SKU"] == sku]
        fig_map.add_trace(go.Scattergeo(
            lat=sub["lat"],
            lon=sub["lon"],
            mode="markers",
            name=sku,
            marker=dict(
                size=np.sqrt(sub["Total_Orders"]) * 6 + 8,
                color=sku_colors.get(sku, "#ffffff"),
                opacity=0.82,
                line=dict(width=1, color="#0a0a0f"),
            ),
            customdata=sub[["Top3_SKUs", "Total_Orders"]].values,
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Total Orders: %{customdata[1]}<br><br>"
                "Top 3 SKUs:<br>%{customdata[0]}"
                "<extra></extra>"
            ),
            text=sub["City"],
        ))

    fig_map.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Mono, monospace", color="#e8e4d9"),
        geo=dict(
            scope="asia",
            center=dict(lat=22, lon=80),
            projection_scale=4.5,
            bgcolor="rgba(0,0,0,0)",
            landcolor="#12121e",
            oceancolor="#080810",
            lakecolor="#080810",
            rivercolor="#080810",
            subunitcolor="#2a2a4a",
            countrycolor="#2a2a4a",
            showland=True,
            showocean=True,
            showlakes=True,
            showcountries=True,
            showsubunits=True,
        ),
        legend=dict(
            bgcolor="rgba(15,15,24,0.9)",
            bordercolor="#2a2a4a",
            borderwidth=1,
            title=dict(text="Top SKU", font=dict(size=11)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
    )

    st.plotly_chart(fig_map, use_container_width=True)

    # City table with top 3 SKUs
    st.markdown("<div class='section-header'>City-level breakdown table</div>", unsafe_allow_html=True)
    display_table = mapped[["City", "Total_Orders", "Top_SKU", "Top3_SKUs"]].copy()
    display_table = display_table.sort_values("Total_Orders", ascending=False).reset_index(drop=True)
    display_table.columns = ["City", "Total Orders", "Top SKU", "Top 3 SKUs"]
    st.dataframe(display_table, use_container_width=True, hide_index=True)
