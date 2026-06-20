import streamlit as st
import pandas as pd
import nflreadpy as nfl
import plotly.express as px

# Premium Application Configuration
st.set_page_config(
    page_title="BillionBetting NFL Model", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection adapted for high-contrast dark environments
st.markdown("""
    <style>
        .block-container { padding-top: 2rem; padding-bottom: 2rem; }
        h1, h2, h3, h4 { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-weight: 700; }
        
        /* Premium Translucent Card Aesthetic for Dark Mode Metrics */
        .stMetric { 
            background-color: rgba(255, 255, 255, 0.04); 
            padding: 15px; 
            border-radius: 12px; 
            border: 1px solid rgba(255, 255, 255, 0.08); 
        }
        
        /* Dark Sidebar Override to ensure high text contrast */
        [data-testid="stSidebar"] { 
            background-color: #1C1C1E; 
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------
# AUTOMATED DATA PIPELINE (Cached for Speed)
# -------------------------------------------------------------------------
@st.cache_data
def load_model_data(seasons):
    schedules = nfl.load_schedules(seasons).to_pandas()
    try:
        weekly = nfl.load_team_stats(seasons, summary_level="week").to_pandas()
    except Exception:
        st.sidebar.info(f"📅 Note: No game stats recorded yet for {seasons}. Teams initialized to baseline 0.0 EPA.")
        weekly = pd.DataFrame(columns=["season", "week", "team", "passing_epa", "rushing_epa"])
    return weekly, schedules

# Title Typography Hero Section
st.title("🏈 BillionBetting")
st.caption("Automated Spread & Total Forecasting Engine • Premium Analytical Dashboard")

# Modern Sidebar Configuration Panel
with st.sidebar:
    st.markdown("### 🔄 Core Engine Controls")
    selected_season = st.selectbox("Target Season", [2023, 2024, 2025, 2026], index=0)
    
    with st.container(border=True):
        st.markdown("**Power Adjustments**")
        hfa_adjustment = st.slider("Home Field Advantage (Pts)", 0.0, 4.0, 2.5, step=0.5)
        epa_multiplier = st.slider("EPA to Points Multiplier", 1.0, 5.0, 3.0, step=0.1)
    
    if st.button("🔄 REFRESH ENGINE CACHE", use_container_width=True):
        st.cache_data.clear()
        st.sidebar.success("Pipeline refreshed successfully!")

# Pipeline Processing Block
weekly_df, schedules_df = load_model_data([selected_season])

weekly_df["team"] = weekly_df["team"].astype(str).str.upper().str.strip()
weekly_df["offense_epa"] = weekly_df["passing_epa"] + weekly_df["rushing_epa"]
weekly_df = weekly_df.sort_values(["team", "season", "week"])

weekly_df["epa_rolling"] = weekly_df.groupby(["team", "season"])["offense_epa"].transform(
    lambda x: x.rolling(window=4, min_periods=1).mean()
)
weekly_df["epa_pt"] = weekly_df.groupby(["team", "season"])["epa_rolling"].shift(1)

schedules_clean = schedules_df.dropna(subset=["spread_line", "total_line"]).copy()
schedules_clean["home_team"] = schedules_clean["home_team"].astype(str).str.upper().str.strip()
schedules_clean["away_team"] = schedules_clean["away_team"].astype(str).str.upper().str.strip()

home_stats = weekly_df[["season", "week", "team", "epa_pt"]].rename(columns={"team": "home_team", "epa_pt": "epa_home"})
away_stats = weekly_df[["season", "week", "team", "epa_pt"]].rename(columns={"team": "away_team", "epa_pt": "epa_away"})

merged = schedules_clean.merge(home_stats, on=["season", "week", "home_team"], how="left")
merged = merged.merge(away_stats, on=["season", "week", "away_team"], how="left")
merged = merged.fillna(0)

# Base Math Engines
merged["base_predicted_spread"] = (merged["epa_home"] - merged["epa_away"]) * epa_multiplier + hfa_adjustment
merged["base_predicted_total"] = merged["total_line"]

# -------------------------------------------------------------------------
# INTERACTIVE SITUATIONAL CONTEXT ENGINE (iOS Card Style)
# -------------------------------------------------------------------------
st.subheader("🌦️ Situational Overrides & Weather Modules")
selected_week = st.selectbox("Focus Analytics Week", sorted(merged["week"].unique()))
week_games = merged[merged["week"] == selected_week].copy()

adjusted_games = []

# Dynamic Grid Layout for Game Configurations
for idx, row in week_games.iterrows():
    matchup_label = f"🏈 {row['away_team']} at {row['home_team']}"
    
    with st.container(border=True):
        st.markdown(f"#### {matchup_label}")
        col1, col2, col3 = st.columns(3)
        with col1:
            wind = st.slider("Wind Vector (MPH)", 0, 40, 0, key=f"wind_{idx}")
            weather_precip = st.checkbox("Active Precipitation", key=f"precip_{idx}")
        with col2:
            home_qb_out = st.checkbox("Home QB1 Inactive", key=f"hqb_{idx}")
            home_lt_out = st.checkbox("Home LT1 Inactive", key=f"hlt_{idx}")
        with col3:
            away_qb_out = st.checkbox("Away QB1 Inactive", key=f"aqb_{idx}")
            away_lt_out = st.checkbox("Away LT1 Inactive", key=f"alt_{idx}")
            
        total_adj = 0.0
        spread_adj = 0.0
        
        if wind >= 20: total_adj -= 3.0
        elif wind >= 15: total_adj -= 1.5
        if weather_precip: total_adj -= 1.5
        
        if home_qb_out: spread_adj -= 6.0
        if home_lt_out: spread_adj -= 0.75
        if away_qb_out: spread_adj += 6.0
        if away_lt_out: spread_adj += 0.75
        
        row["final_predicted_spread"] = row["base_predicted_spread"] + spread_adj
        row["final_predicted_total"] = row["base_predicted_total"] + total_adj
        adjusted_games.append(row)

final_week_df = pd.DataFrame(adjusted_games)

# Edge & Tier Math Evaluation Pipeline
final_week_df["spread_edge"] = final_week_df["final_predicted_spread"] - final_week_df["spread_line"]
final_week_df["absolute_edge"] = final_week_df["spread_edge"].abs()

def assign_tier(row):
    if row["absolute_edge"] <= 1.0: return "NO BET", 0
    elif row["absolute_edge"] <= 2.0: return "Tier 1", 79
    elif row["absolute_edge"] <= 3.0: return "Tier 1.5", 85
    else: return "Tier 2", 92

tiers_and_scores = final_week_df.apply(assign_tier, axis=1)
final_week_df["Tier"] = [t[0] for t in tiers_and_scores]
final_week_df["Confidence"] = [t[1] for t in tiers_and_scores]

# -------------------------------------------------------------------------
# HIGH-LEVEL INSIGHTS & ANALYTICAL METRICS
# -------------------------------------------------------------------------
st.subheader("📊 Premium Projections & Model Edges")

# App-Style Summary Scorecards
active_bets_count = len(final_week_df[final_week_df["Tier"] != "NO BET"])
max_edge_val = final_week_df["absolute_edge"].max() if not final_week_df.empty else 0.0

m_col1, m_col2, m_col3 = st.columns(3)
with m_col1:
    st.metric(label="Total Active Action Slate", value=f"{active_bets_count} Games")
with m_col2:
    st.metric(label="Maximum Identified Model Edge", value=f"{max_edge_val:.2f} Pts")
with m_col3:
    st.metric(label="Current Model Configuration Scale", value=f"Scale x{epa_multiplier}")

# Clean, Modern Chart Rendering Section
fig = px.bar(
    final_week_df,
    x="home_team",
    y="spread_edge",
    color="Tier",
    title="Quantified Net Market Value Variance Matrix",
    labels={"spread_edge": "Model Advantage Points vs Spread Line", "home_team": "Stadium Anchor (Home Team)"},
    hover_data=["away_team", "spread_line", "final_predicted_spread"],
    barmode="group",
    color_discrete_map={"NO BET": "#8E8E93", "Tier 1": "#34C759", "Tier 1.5": "#FF9500", "Tier 2": "#FF3B30"}
)
fig.update_layout(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_family="-apple-system, BlinkMacSystemFont, sans-serif",
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
)
st.plotly_chart(fig, use_container_width=True)

# Format Tables Column Matrix Output
display_cols = [
    "away_team", "home_team", "spread_line", "final_predicted_spread", 
    "spread_edge", "total_line", "final_predicted_total", "Tier", "Confidence"
]
formatted_card = final_week_df[display_cols].rename(columns={
    "away_team": "Away", "home_team": "Home",
    "spread_line": "Market Line", "final_predicted_spread": "Model Line",
    "spread_edge": "Raw Edge", "total_line": "Market O/U",
    "final_predicted_total": "Model O/U"
})

# Complete Table Section
st.markdown("### 📋 Complete Weekly Projection Board")
st.dataframe(
    formatted_card.style.background_gradient(subset=["Raw Edge"], cmap="RdYlGn"),
    use_container_width=True,
    hide_index=True
)

# Premium Filter Card Output View
st.markdown("### 🔥 Top Qualified Strategic Action Candidates")
bets_only = formatted_card[formatted_card["Tier"] != "NO BET"].sort_values(by="Raw Edge", key=abs, ascending=False)
if not bets_only.empty:
    st.dataframe(bets_only, use_container_width=True, hide_index=True)
else:
    st.info("No matchups currently scale out to surpass the minimal Edge > 1.0 Point Threshold setting.")