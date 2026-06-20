import streamlit as st


def render_schedule(active_week_sched, selected_week):
    st.subheader(f"📅 Operational Slate — Week {selected_week}")
    st.caption("Consensus spreads and totals from the active schedule. Use these to compare against model outputs.")
    st.dataframe(active_week_sched[["game_id", "away_team", "home_team", "spread_line", "total_line"]], use_container_width=True, hide_index=True)
