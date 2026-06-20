import streamlit as st


def render_team_stats(processed_profiles, selected_week):
    st.subheader("📊 Team Efficiency — Base Metrics")
    st.caption("Key efficiency metrics used as the foundation for model differentials (EPA, YPP, turnovers).")
    latest_week_stats = processed_profiles[processed_profiles["week"] == selected_week]
    st.dataframe(latest_week_stats[["team", "week", "epa", "y_per_play", "turnovers"]], use_container_width=True, hide_index=True)
