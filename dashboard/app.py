"""
LA Crime Analytics — Interactive Dashboard
=============================================
An urban safety analytics console for exploring LAPD crime incident
data (2020–present): geographic hotspots, temporal patterns, and a
live crime-type prediction model.

Run:
    streamlit run dashboard/app.py
"""

import os
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# Page config + theme
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="LA Crime Analytics",
    page_icon="🌃",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#0B1220"
PANEL = "#131B2E"
PANEL_LIGHT = "#1B2740"
TEXT = "#E8ECF3"
MUTED = "#8B96AC"
AMBER = "#F2A65A"
RED = "#E4572E"
TEAL = "#4F9DDE"
GRID = "#243254"

CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {TEXT};
}}

.stApp {{
    background: {BG};
}}

section[data-testid="stSidebar"] {{
    background: {PANEL};
    border-right: 1px solid {GRID};
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

.console-header {{
    display: flex;
    align-items: baseline;
    gap: 14px;
    padding-bottom: 4px;
    border-bottom: 1px solid {GRID};
    margin-bottom: 24px;
}}
.console-header .dot {{
    width: 10px; height: 10px; border-radius: 50%;
    background: {AMBER};
    box-shadow: 0 0 12px {AMBER};
    display: inline-block;
}}
.console-eyebrow {{
    color: {MUTED};
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    font-weight: 500;
}}

.kpi-card {{
    background: {PANEL};
    border: 1px solid {GRID};
    border-radius: 10px;
    padding: 18px 20px;
    height: 100%;
}}
.kpi-label {{
    color: {MUTED};
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
}}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 600;
    color: {TEXT};
}}
.kpi-sub {{
    color: {MUTED};
    font-size: 0.78rem;
    margin-top: 4px;
}}
.kpi-value.amber {{ color: {AMBER}; }}
.kpi-value.red {{ color: {RED}; }}
.kpi-value.teal {{ color: {TEAL}; }}

div[data-testid="stMetricValue"] {{ color: {TEXT}; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {GRID};
}}
.stTabs [data-baseweb="tab"] {{
    color: {MUTED};
    font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    color: {AMBER} !important;
}}

footer {{visibility: hidden;}}
#MainMenu {{visibility: hidden;}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, family="Inter"),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
)

# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "feature_engineered_crime_data.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "crime_type_model.pkl")
ENCODERS_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "encoders.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "feature_list.pkl")


@st.cache_data(show_spinner="Loading crime records...")
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["DATE OCC"])
    return df


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, encoders, features


df = load_data()
model, encoders, feature_list = load_model()

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="console-header">
        <span class="dot"></span>
        <div>
            <div class="console-eyebrow">Urban Safety Analytics</div>
            <h1 style="margin:0; font-size:1.7rem;">LA Crime Console</h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Sidebar filters
# ----------------------------------------------------------------------
st.sidebar.markdown("### Filters")

years = sorted(df["Year"].dropna().unique().tolist())
year_sel = st.sidebar.multiselect("Year", years, default=years)

areas = sorted(df["AREA NAME"].dropna().unique().tolist())
area_sel = st.sidebar.multiselect("Area", areas, default=[])

crime_filter = st.sidebar.radio("Crime type", ["All", "Violent only", "Non-violent only"], index=0)

hour_range = st.sidebar.slider("Hour of day", 0, 23, (0, 23))

st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df):,} incidents · LAPD, 2020–present")
st.sidebar.caption("Source: data.lacity.org (via Kaggle)")

filtered = df.copy()
if year_sel:
    filtered = filtered[filtered["Year"].isin(year_sel)]
if area_sel:
    filtered = filtered[filtered["AREA NAME"].isin(area_sel)]
if crime_filter == "Violent only":
    filtered = filtered[filtered["Violent Crime Flag"] == 1]
elif crime_filter == "Non-violent only":
    filtered = filtered[filtered["Violent Crime Flag"] == 0]
filtered = filtered[
    (filtered["Crime Hour"] >= hour_range[0]) & (filtered["Crime Hour"] <= hour_range[1])
]

# ----------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------
total_incidents = len(filtered)
violent_pct = filtered["Violent Crime Flag"].mean() * 100 if total_incidents else 0
top_area = filtered["AREA NAME"].value_counts().idxmax() if total_incidents else "—"
peak_hour = filtered["Crime Hour"].value_counts().idxmax() if total_incidents else "—"

k1, k2, k3, k4 = st.columns(4)
for col, label, value, sub, cls in [
    (k1, "Total incidents", f"{total_incidents:,}", "matching current filters", ""),
    (k2, "Violent crime share", f"{violent_pct:.1f}%", "of filtered incidents", "red"),
    (k3, "Highest-incident area", str(top_area), "most reports, current filters", "amber"),
    (k4, "Peak hour", f"{peak_hour}:00" if peak_hour != "—" else "—", "most frequent hour of day", "teal"),
]:
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {cls}">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------
tab_overview, tab_geo, tab_time, tab_predict = st.tabs(
    ["Overview", "Geographic", "Temporal Patterns", "Predict Crime Type"]
)

# --- Overview ---
with tab_overview:
    c1, c2 = st.columns([1.3, 1])

    with c1:
        st.markdown("##### Top 10 crime types")
        top_crimes = filtered["Crm Cd Desc"].value_counts().head(10).sort_values()
        fig = px.bar(
            x=top_crimes.values, y=top_crimes.index, orientation="h",
            color=top_crimes.values, color_continuous_scale=["#243254", AMBER],
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, coloraxis_showscale=False)
        fig.update_yaxes(title=None)
        fig.update_xaxes(title="Incidents")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Violent vs. non-violent")
        vc = filtered["Violent Crime Flag"].value_counts().rename({0: "Non-Violent", 1: "Violent"})
        fig = px.pie(
            values=vc.values, names=vc.index, hole=0.55,
            color=vc.index, color_discrete_map={"Non-Violent": TEAL, "Violent": RED},
        )
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Incidents by area")
    area_counts = filtered["AREA NAME"].value_counts().sort_values()
    fig = px.bar(
        x=area_counts.values, y=area_counts.index, orientation="h",
        color=area_counts.values, color_continuous_scale=["#243254", TEAL],
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, coloraxis_showscale=False, height=500)
    fig.update_yaxes(title=None)
    fig.update_xaxes(title="Incidents")
    st.plotly_chart(fig, use_container_width=True)

# --- Geographic ---
with tab_geo:
    st.markdown("##### Incident density map")
    st.caption("Sampled to 20,000 points for performance.")
    map_df = filtered.dropna(subset=["LAT", "LON"])
    map_df = map_df[(map_df["LAT"] != 0) & (map_df["LON"] != 0)]
    if len(map_df) > 20000:
        map_df = map_df.sample(20000, random_state=42)

    if len(map_df):
    fig = px.density_map(
        map_df, lat="LAT", lon="LON", radius=4, zoom=9,
        center=dict(lat=34.05, lon=-118.25),
        map_style="open-street-map",
        color_continuous_scale=["#0B1220", TEAL, AMBER, RED],
    )
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=560, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No geolocated incidents match the current filters.")

# --- Temporal ---
with tab_time:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("##### Incidents by hour of day")
        hourly = filtered["Crime Hour"].value_counts().sort_index()
        fig = px.line(x=hourly.index, y=hourly.values, markers=True)
        fig.update_traces(line_color=AMBER, marker=dict(color=AMBER, size=6))
        fig.update_layout(**PLOTLY_TEMPLATE["layout"])
        fig.update_xaxes(title="Hour")
        fig.update_yaxes(title="Incidents")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("##### Incidents by day of week")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_counts = filtered["Day of Week"].value_counts().reindex(day_order)
        fig = px.bar(x=day_order, y=day_counts.values, color=day_counts.values,
                      color_continuous_scale=["#243254", TEAL])
        fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, coloraxis_showscale=False)
        fig.update_xaxes(title=None)
        fig.update_yaxes(title="Incidents")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### Monthly trend")
    month_order = ["January","February","March","April","May","June",
                    "July","August","September","October","November","December"]
    monthly = filtered["Month Name"].value_counts().reindex(month_order)
    fig = px.line(x=month_order, y=monthly.values, markers=True)
    fig.update_traces(line_color=RED, marker=dict(color=RED, size=7))
    fig.update_layout(**PLOTLY_TEMPLATE["layout"], height=380)
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Incidents")
    st.plotly_chart(fig, use_container_width=True)

# --- Predict ---
with tab_predict:
    st.markdown("##### Predict likely crime type")
    st.caption(
        "Random Forest classifier trained on the top 10 most frequent crime types. "
        "Given a location and situation profile, estimates the most probable crime category."
    )

    if model is None:
        st.warning("No trained model found. Run `python src/train_model.py` first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            in_area = st.selectbox("Area", areas)
            in_hour = st.slider("Hour of day", 0, 23, 20)
        with c2:
            in_day = st.selectbox("Day of week", day_order)
            in_month = st.selectbox("Month", list(range(1, 13)), format_func=lambda m: month_order[m-1])
        with c3:
            in_age = st.number_input("Victim age", min_value=0, max_value=100, value=30)
            in_sex = st.selectbox("Victim sex", ["M", "F", "X"])

        in_weapon = st.checkbox("Weapon involved")

        if st.button("Predict", type="primary"):
            area_crime_count = int(df[df["AREA NAME"] == in_area]["Area Crime Count"].iloc[0])

            row = pd.DataFrame([{
                "AREA NAME": in_area,
                "Crime Hour": in_hour,
                "Day of Week": in_day,
                "Month": in_month,
                "Vict Age": in_age,
                "Vict Sex": in_sex,
                "Weapon Used Flag": int(in_weapon),
                "Violent Crime Flag": 0,  # unknown at prediction time; left neutral
                "Area Crime Count": area_crime_count,
            }])

            for col in ["AREA NAME", "Day of Week", "Vict Sex"]:
                le = encoders[col]
                row[col] = row[col].apply(lambda v: v if v in le.classes_ else le.classes_[0])
                row[col] = le.transform(row[col])

            row = row[feature_list]

            proba = model.predict_proba(row)[0]
            classes = model.classes_
            result = pd.Series(proba, index=classes).sort_values(ascending=False).head(5)

            st.markdown("###### Top predicted crime types")
            fig = px.bar(
                x=result.values, y=result.index, orientation="h",
                color=result.values, color_continuous_scale=["#243254", AMBER],
            )
            fig.update_layout(**PLOTLY_TEMPLATE["layout"], showlegend=False, coloraxis_showscale=False, height=320)
            fig.update_xaxes(title="Predicted probability")
            fig.update_yaxes(title=None)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "Model accuracy on held-out test data: ~51% across 10 classes "
                "(vs. ~10% for random guessing) — see README for full evaluation."
            )
