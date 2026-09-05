"""
SECTION 13 (Dashboard) — Streamlit interactive Women Safety Index dashboard

Run from project root:
  streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

st.set_page_config(
    page_title="Women Online Safety Index",
    page_icon="🛡️",
    layout="wide",
)

st.title("Women Online Safety Index")
st.caption("WHSI + MRI dual-index system — platform danger vs. moderation accountability")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


whsi = load_csv(os.path.join(ROOT, "data/processed/whsi_scores.csv"))
mri = load_csv(os.path.join(ROOT, "data/processed/mri_scores.csv"))
matrix = load_csv(os.path.join(ROOT, "outputs/results/safety_matrix_data.csv"))
hotspots = load_csv(os.path.join(ROOT, "outputs/results/community_whsi.csv"))
hashtags = load_csv(os.path.join(ROOT, "outputs/results/hashtag_danger.csv"))
subs = load_csv(os.path.join(ROOT, "outputs/results/subreddit_profiles.csv"))

tab1, tab2, tab3, tab4 = st.tabs(
    ["Safety Matrix", "WHSI Platforms", "MRI Accountability", "Hotspots & Hashtags"]
)

with tab1:
    st.subheader("2×2 Safety Matrix (WHSI vs MRI)")
    if matrix.empty:
        st.warning("Run `python run_pipeline.py` to generate safety_matrix_data.csv")
    else:
        fig = px.scatter(
            matrix,
            x="WHSI_score",
            y="MRI_score",
            text="platform",
            size=matrix.get("n_comments", pd.Series([500] * len(matrix))),
            color="quadrant",
            range_x=[0, 100],
            range_y=[0, 100],
            title="Platform positioning: danger (WHSI) vs accountability (MRI)",
        )
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.add_vline(x=50, line_dash="dash", line_color="gray")
        fig.update_traces(textposition="top center")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(matrix, use_container_width=True)

with tab2:
    st.subheader("Women Harassment Severity Index (WHSI)")
    if whsi.empty:
        st.warning("Missing whsi_scores.csv — run fuzzy_engine or full pipeline.")
    else:
        fig = px.bar(
            whsi.sort_values("WHSI_score", ascending=True),
            x="WHSI_score",
            y="platform",
            color="WHSI_category",
            orientation="h",
            title="Platform WHSI (higher = more dangerous for women)",
        )
        st.plotly_chart(fig, use_container_width=True)
        dim_cols = [c for c in ["toxicity", "threat", "frequency", "normalization"] if c in whsi.columns]
        if dim_cols:
            st.markdown("**Dimension breakdown (top platform)**")
            top = whsi.iloc[0]
            fig2 = go.Figure(
                data=go.Scatterpolar(
                    r=[top[c] for c in dim_cols],
                    theta=["Toxicity", "Threat", "Frequency", "Normalization"],
                    fill="toself",
                    name=top["platform"],
                )
            )
            fig2.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
            st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(whsi, use_container_width=True)

with tab3:
    st.subheader("Moderation Responsiveness Index (MRI)")
    if mri.empty:
        st.warning("Missing mri_scores.csv — run mri_engine.py")
    else:
        fig = px.bar(
            mri.sort_values("MRI_score", ascending=True),
            x="MRI_score",
            y="platform",
            color="MRI_label",
            orientation="h",
            title="Platform MRI (higher = more accountable)",
        )
        st.plotly_chart(fig, use_container_width=True)
        comp_cols = [c for c in ["removal_rate_pct", "speed_score", "consistency_pct"] if c in mri.columns]
        if comp_cols:
            st.dataframe(mri[["platform", "MRI_score"] + comp_cols], use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Community hotspots")
        if hotspots.empty:
            st.info("Run hotspot_detection.py after ingesting scraped data with subreddit/channel columns.")
        else:
            st.dataframe(hotspots.head(20), use_container_width=True)
    with c2:
        st.subheader("Dangerous hashtags")
        if hashtags.empty:
            st.info("Run hashtag_analysis.py (needs hashtags in comments).")
        else:
            st.dataframe(hashtags.head(20), use_container_width=True)

    if not subs.empty:
        st.subheader("Subreddit profiles (Section 12)")
        st.dataframe(subs.head(15), use_container_width=True)

st.sidebar.markdown("### Pipeline")
st.sidebar.code("python run_pipeline.py", language="bash")
st.sidebar.markdown("### Ingest your scrapes")
st.sidebar.code("python src/load_scraped.py --all", language="bash")
st.sidebar.markdown("### Audit data quality")
st.sidebar.code("python src/dataset_audit.py --all-scraped", language="bash")
