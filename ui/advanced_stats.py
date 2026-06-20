import streamlit as st


def render_advanced_stats(processed_profiles, selected_week):
    st.subheader("🧠 Advanced Differentials & Rolling Profiles")
    st.caption("Rolling 4-week profile components used to compute composite differentials for matchup-level predictions.")
    st.dataframe(
        processed_profiles[processed_profiles["week"] == selected_week][
            ["team", "y_per_play_roll", "success_rate_roll", "sack_rate_roll", "red_zone_pct_roll"]
        ], use_container_width=True, hide_index=True
    )
