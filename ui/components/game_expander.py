import streamlit as st


def render_game_expander(row, processed_profiles, selected_week, scale_input, hfa_input):
    """Render the model inputs for a single game and return the adjusted game record dict."""
    g_id = row["game_id"]
    h_team = row["home_team"]
    a_team = row["away_team"]

    h_prof = processed_profiles[(processed_profiles["team"] == h_team) & (processed_profiles["week"] == selected_week)]
    a_prof = processed_profiles[(processed_profiles["team"] == a_team) & (processed_profiles["week"] == selected_week)]

    h_epa = h_prof["epa_prev"].values[0] if not h_prof.empty else 0.0
    a_epa = a_prof["epa_prev"].values[0] if not a_prof.empty else 0.0
    h_ypp = h_prof["y_per_play_prev"].values[0] if not h_prof.empty else 5.4
    a_ypp = a_prof["y_per_play_prev"].values[0] if not a_prof.empty else 5.4
    h_sr = h_prof["success_rate_prev"].values[0] if not h_prof.empty else 45.0
    a_sr = a_prof["success_rate_prev"].values[0] if not a_prof.empty else 45.0
    h_sack = h_prof["sack_rate_prev"].values[0] if not h_prof.empty else 6.0
    a_sack = a_prof["sack_rate_prev"].values[0] if not a_prof.empty else 6.0
    h_to = h_prof["turnovers_prev"].values[0] if not h_prof.empty else 1.2
    a_to = a_prof["turnovers_prev"].values[0] if not a_prof.empty else 1.2
    h_rz = h_prof["red_zone_pct_prev"].values[0] if not h_prof.empty else 55.0
    a_rz = a_prof["red_zone_pct_prev"].values[0] if not a_prof.empty else 55.0

    epa_diff = h_epa - a_epa
    ypp_diff = h_ypp - a_ypp
    sr_diff = h_sr - a_sr
    sack_diff = a_sack - h_sack
    to_diff = a_to - h_to
    rz_diff = h_rz - a_rz

    composite_delta = (epa_diff * 4.5) + (ypp_diff * 1.5) + (sr_diff * 0.12) + (sack_diff * 0.08) + (to_diff * 0.40) + (rz_diff * 0.04)

    base_predicted_spread = -(composite_delta * (scale_input / 3.2)) - hfa_input
    base_predicted_total = row["total_line"] + ((h_epa + a_epa) * 2.0)

    with st.expander(f"🏈 {a_team} @ {h_team} (Market Line: {h_team} {row['spread_line']})", expanded=False):
        col_w, col_inj, col_sit = st.columns(3)

        with col_w:
            st.markdown("**🌦️ Weather Systems Engine**")
            wind = st.slider("Wind Vector (MPH)", 0, 35, 0, key=f"w_wind_{g_id}")
            precip = st.selectbox("Precipitation Profile", ["None", "Heavy Rain", "Snow"], key=f"w_precip_{g_id}")
            extreme_cold = st.checkbox("Extreme Cold Cycle Active", key=f"w_cold_{g_id}")

        with col_inj:
            st.markdown("**🏥 Injury Adjustments Matrix**")
            h_qb = st.checkbox("Home QB1 Out (-6.0)", key=f"h_qb_{g_id}")
            h_lt = st.checkbox("Home Starting LT Out (-0.75)", key=f"h_lt_{g_id}")
            h_wr1 = st.checkbox("Home WR1 Out (-1.0)", key=f"h_wr1_{g_id}")
            h_pass = st.checkbox("Home Elite Pass Rusher Out (-0.5)", key=f"h_pass_{g_id}")
            h_ol = st.checkbox("Home Multiple OL Injuries (-0.5)", key=f"h_ol_{g_id}")
            st.divider()
            a_qb = st.checkbox("Away QB1 Out (+6.0)", key=f"a_qb_{g_id}")
            a_lt = st.checkbox("Away Starting LT Out (+0.75)", key=f"a_lt_{g_id}")
            a_wr1 = st.checkbox("Away WR1 Out (+1.0)", key=f"a_wr1_{g_id}")
            a_pass = st.checkbox("Away Elite Pass Rusher Out (+0.5)", key=f"a_pass_{g_id}")
            a_ol = st.checkbox("Away Multiple OL Injuries (+0.5)", key=f"a_ol_{g_id}")

        with col_sit:
            st.markdown("**✈️ Situational & Manual Overrides**")
            rest_adv = st.selectbox("Rest Profile", ["Neutral", "Home Extra Rest", "Away Extra Rest", "Home Short Week", "Away Short Week"], key=f"sit_rest_{g_id}")
            h_travel = st.checkbox("Home Cross-Country Burden (-0.5)", key=f"h_trv_{g_id}")
            a_travel = st.checkbox("Away Cross-Country / West-to-East Burden (+0.5)", key=f"a_trv_{g_id}")
            manual_pts = st.number_input("Coaching / Motivation Points Mod", -5.0, 5.0, 0.0, step=0.5, key=f"manual_{g_id}")

        weather_total_adj = 0.0
        if wind >= 20: weather_total_adj -= 3.0
        elif wind >= 15: weather_total_adj -= 1.5
        if precip in ["Heavy Rain", "Snow"]: weather_total_adj -= 1.5
        if extreme_cold: weather_total_adj -= 0.5

        context_spread_adj = 0.0
        injury_count = 0

        if h_qb: context_spread_adj += 6.0; injury_count += 1
        if h_lt: context_spread_adj += 0.75; injury_count += 1
        if h_wr1: context_spread_adj += 1.0; injury_count += 1
        if h_pass: context_spread_adj += 0.50; injury_count += 1
        if h_ol: context_spread_adj += 0.50; injury_count += 1

        if a_qb: context_spread_adj -= 6.0; injury_count += 1
        if a_lt: context_spread_adj -= 0.75; injury_count += 1
        if a_wr1: context_spread_adj -= 1.0; injury_count += 1
        if a_pass: context_spread_adj -= 0.50; injury_count += 1
        if a_ol: context_spread_adj -= 0.50; injury_count += 1

        if rest_adv == "Home Extra Rest": context_spread_adj -= 1.0
        elif rest_adv == "Away Extra Rest": context_spread_adj += 1.0
        elif rest_adv == "Home Short Week": context_spread_adj += 1.0
        elif rest_adv == "Away Short Week": context_spread_adj -= 1.0

        if h_travel: context_spread_adj += 0.50
        if a_travel: context_spread_adj -= 0.50
        context_spread_adj -= manual_pts

        final_predicted_spread = base_predicted_spread + context_spread_adj
        final_predicted_total = base_predicted_total + weather_total_adj

        home_edge = row["spread_line"] - final_predicted_spread
        abs_edge = abs(home_edge)

        stat_score = min(100.0, (abs_edge / 3.0) * 100.0)
        situational_score = max(50.0, min(100.0, 85.0 + (15.0 if rest_adv != "Neutral" else 0.0) - (10.0 if (h_travel or a_travel) else 0.0) + (manual_pts * 2.0)))
        market_score = min(100.0, 75.0 + abs(row["spread_line"]) * 2.5)
        injury_score = max(40.0, 100.0 - (injury_count * 12.0))

        confidence_score = (0.40 * stat_score) + (0.30 * situational_score) + (0.20 * market_score) + (0.10 * injury_score)
        confidence_score = round(max(50.0, min(100.0, confidence_score)))

        if abs_edge <= 1.0 or confidence_score < 77:
            tier_assignment = "NO BET"
        elif 77 <= confidence_score <= 82:
            tier_assignment = "Tier 1"
        elif 83 <= confidence_score <= 88:
            tier_assignment = "Tier 1.5"
        else:
            tier_assignment = "Tier 2"

        bet_side = h_team if home_edge > 0 else a_team

        # Show a concise model summary for this matchup so the user can validate adjustments immediately
        st.markdown("**Model Summary (Matchup Adjustment Results)**")
        st.write(f"- Predicted Spread (model): {final_predicted_spread:+.2f} pts")
        st.write(f"- Predicted Game Total (model): {final_predicted_total:.1f} pts")
        st.write(f"- Market Spread: {row['spread_line']:+.2f} pts | Raw Edge: {home_edge:+.2f} pts | Abs Edge: {abs_edge:.2f} pts")
        st.write(f"- Confidence: {confidence_score}% | Assigned Tier: {tier_assignment}")

        return {
            "game_id": g_id, "Matchup": f"{a_team} @ {h_team}", "Market Spread": row["spread_line"],
            "Model Spread": round(final_predicted_spread, 2), "Raw Edge": round(home_edge, 2),
            "Abs Edge": abs_edge, "Market Total": row["total_line"], "Model Total": round(final_predicted_total, 2),
            "Confidence": confidence_score, "Tier": tier_assignment, "Target Play": f"{bet_side} " + (f"{home_edge:+.2f}" if bet_side == h_team else f"{-home_edge:+.2f}")
        }
