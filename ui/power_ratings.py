import streamlit as st
import plotly.express as px


def render_power_ratings(processed_profiles, selected_week):
    st.subheader("👑 Power Ratings — Team Strength Index")
    st.caption("Composite power ratings that summarize recent team form and efficiency for ranking.")
    rating_week = processed_profiles[processed_profiles["week"] == selected_week].copy()
    rating_week["Composite Score"] = (rating_week["epa_roll"] * 10) + (rating_week["success_rate_roll"] * 0.5)
    rating_week = rating_week.sort_values(by="Composite Score", ascending=False)
    fig_power = px.bar(rating_week, x="team", y="Composite Score", title="System Dynamic Strength Matrix Scale")
    st.plotly_chart(fig_power, use_container_width=True)
