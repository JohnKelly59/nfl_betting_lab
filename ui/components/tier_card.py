import streamlit as st


def render_tier_card(p_row):
    """Render a colored tier recommendation card for a play row (p_row: Series/dict)."""
    color_map = {"Tier 1": "#34C759", "Tier 1.5": "#FF9500", "Tier 2": "#FF3B30"}
    bg_color = color_map.get(p_row.get("Tier", ""), "#333")

    st.markdown(f"""
    <div class="tier-card" style="background-color: {bg_color}; color: white;">
        🚀 {p_row['Tier']} Recommendation Play: {p_row['Target Play']} (Matchup: {p_row['Matchup']}) <br/>
        &nbsp;&nbsp;&nbsp;&nbsp;[Market Line: {p_row['Market Spread']} | Model Line: {p_row['Model Spread']} | Edge Margin: {p_row['Raw Edge']:+.2f} Pts | Confidence: {p_row['Confidence']}%]
    </div>
    """, unsafe_allow_html=True)
