import streamlit as st


def render_metric(col, label, value, caption=None):
    """Render a metric inside a provided column context."""
    with col:
        st.metric(label, value)
        if caption:
            st.caption(caption)
