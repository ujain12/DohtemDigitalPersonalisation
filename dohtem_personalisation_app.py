"""
Dohtem E-Commerce Personalisation Engine
Customer Intelligence & ML Pipeline
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (confusion_matrix, silhouette_score,
                              roc_auc_score, roc_curve)
from sklearn.manifold import TSNE

# ── PAGE CONFIG ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dohtem | Customer Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME CONSTANTS ────────────────────────────────────────────────
BG      = "#07090f"
SURFACE = "#0e1118"
BORDER  = "#1c2030"
PRIMARY = "#4f6ef7"
RED     = "#e05c5c"
GREEN   = "#4cc98a"
MUTED   = "#6b7280"
TEXT    = "#d1d5db"
WHITE   = "#f9fafb"


def apply_layout(fig, title="", height=400, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color=WHITE, family="monospace")),
        paper_bgcolor=SURFACE,
        plot_bgcolor=BG,
        font=dict(family="monospace", color=TEXT, size=11),
        height=height,
        showlegend=legend,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(gridcolor=BORDER, linecolor=BORDER),
        yaxis=dict(gridcolor=BORDER, linecolor=BORDER),
    )
    return fig


# ── CSS ───────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
h1,h2,h3,h4 {{ font-family: 'IBM Plex Mono', monospace; color: {WHITE}; }}

[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER};
}}

.kpi-row {{ display: flex; gap: 12px; margin-bottom: 20px; }}
.kpi {{
    flex: 1;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 2px solid {PRIMARY};
    border-radius: 4px;
    padding: 16px 20px;
}}
.kpi .v {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.75rem;
    font-weight: 600;
    color: {WHITE};
    line-height: 1;
}}
.kpi .l {{
    font-size: 0.7rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 6px;
}}

.info-box {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 3px solid {PRIMARY};
    border-radius: 3px;
    padding: 12px 16px;
    margin: 6px 0 14px 0;
    font-size: 0.87rem;
    color: {TEXT};
    line-height: 1.65;
}}
.info-box b {{ color: {WHITE}; }}

.rationale-box {{
    background: #071420;
    border: 1px solid #0f2a40;
    border-left: 3px solid {GREEN};
    border-radius: 3px;
    padding: 12px 16px;
    margin: 0 0 18px 0;
    font-size: 0.86rem;
    color: {TEXT};
    line-height: 1.65;
}}
.rationale-box b {{ color: {GREEN}; }}

.sec-heading {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    color: {PRIMARY};
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin: 22px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid {BORDER};
}}

.page-title {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.35rem;
    font-weight: 600;
    color: {WHITE};
    margin-bottom: 2px;
}}
.page-sub {{
    font-size: 0.82rem;
    color: {MUTED};
    margin-bottom: 20px;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 0; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {MUTED};
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.79rem;
    border-radius: 0;
}}
.stTabs [aria-selected="true"] {{
    color: {PRIMARY} !important;
    border-bottom: 2px solid {PRIMARY} !important;
    background: transparent !important;
}}

.stButton > button {{
    background: {PRIMARY};
    color: white;
    border: none;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    padding: 8px 20px;
}}
hr {{ border-color: {BORDER} !important; }}
</style>
""", unsafe_allow_html=True)


# ── DATA LOADING ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    for p in [
        "/mnt/user-data/uploads/dohtem_ecommerce_customers.csv",
        "dohtem_ecommerce_customers.csv",
    ]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError("Upload dohtem_ecommerce_customers.csv via the sidebar.")


@st.cache_data
def preprocess(df: pd.DataFrame):
    df = df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    df_enc = df.copy()
    le = LabelEncoder()
    for c in cat_cols:
        df_enc[c] = le.fit_transform(df[c])
    feat_cols = [c for c in df_enc.columns if c != "CustomerID"]
    X_scaled = StandardScaler().fit_transform(df_enc[feat_cols])
    return df, df_enc, X_scaled, feat_cols, cat_cols, num_cols


# ── SIDEBAR ───────────────────────────────────────────────────────
st.sidebar.markdown(
    f'<div style="font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600;'
    f'color:{WHITE};padding:12px 0 2px 0">DOHTEM</div>'
    f'<div style="font-size:0.72rem;color:{MUTED};margin-bottom:16px">'
    f'Customer Intelligence Platform</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload customer CSV", type="csv")

try:
    raw_df = pd.read_csv(uploaded) if uploaded else load_data()
    st.sidebar.markdown(
        f'<div style="font-size:0.75rem;color:{GREEN};padding:4px 0">'
        f'{len(raw_df):,} records loaded</div>', unsafe_allow_html=True
    )
except Exception as e:
    st.sidebar.error(str(e))
    st.stop()

df, df_enc, X_scaled, feat_cols, cat_cols, num_cols = preprocess(raw_df)
# ── FEATURE MATRIX FOR SEGMENTATION / PCA / t-SNE (EXCLUDES OUTCOME) ─────────
# NOTE: X_scaled still exists for any general exploration, but clustering/DR should not include outcomes.
SEG_EXCLUDE = {"CustomerID", "Churn"}
seg_feats = [c for c in feat_cols if c in df_enc.columns and c not in SEG_EXCLUDE]

if len(seg_feats) < 2:
    st.sidebar.error("Not enough usable features for segmentation after excluding CustomerID/Churn.")
    st.stop()

# Fit scaler ONCE so Segmentation + Dimensionality Reduction use the same scaled space
seg_scaler = StandardScaler()
X_seg = seg_scaler.fit_transform(df_enc[seg_feats].values)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<div style="font-size:0.68rem;color:{MUTED};text-transform:uppercase;'
    f'letter-spacing:0.12em;margin-bottom:8px">Navigation</div>',
    unsafe_allow_html=True
)
PAGES = [
    "1. Data Overview",
    "2. Customer Segmentation",
    "3. Dimensionality Reduction",
    "4. Churn Prediction",
    "5. Feature Importance",
    "6. Personalisation Framework",
]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.markdown(
    f'<div style="font-size:0.7rem;color:{MUTED};margin-top:16px">'
    f'{df.shape[0]} rows · {df.shape[1]} columns</div>',
    unsafe_allow_html=True
)


# ════════════════════════════════════════════════════════════════
# 1. DATA OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "1. Data Overview":
    st.markdown('<div class="page-title">1. Data Overview</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">Exploratory summary — {df.shape[0]:,} customers, {df.shape[1]} features</div>',
        unsafe_allow_html=True
    )

    churn_pct = df["Churn"].mean() * 100
    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi"><div class="v">{len(df):,}</div><div class="l">Total Customers</div></div>
      <div class="kpi"><div class="v">{churn_pct:.1f}%</div><div class="l">Churn Rate</div></div>
      <div class="kpi"><div class="v">{df['OrderCount'].mean():.1f}</div><div class="l">Avg Orders</div></div>
      <div class="kpi"><div class="v">{df['HourSpendOnApp'].mean():.1f}h</div><div class="l">Avg App Time</div></div>
      <div class="kpi"><div class="v">£{df['CashbackAmount'].mean():.0f}</div><div class="l">Avg Cashback</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="SatisfactionScore",
                           color=df["Churn"].map({0: "Retained", 1: "Churned"}),
                           barmode="overlay",
                           color_discrete_map={"Retained": PRIMARY, "Churned": RED})
        fig.update_layout(legend_title="Status")
        apply_layout(fig, "Satisfaction Score by Churn Status", 330)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        vc = df["PreferedOrderCat"].value_counts().reset_index()
        vc.columns = ["Category", "Count"]
        fig = px.bar(vc, x="Category", y="Count",
                     color="Count", color_continuous_scale=[[0, BORDER], [1, PRIMARY]])
        apply_layout(fig, "Orders by Product Category", 330, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df, x="MaritalStatus", y="CashbackAmount",
                     color="MaritalStatus",
                     color_discrete_sequence=[PRIMARY, GREEN, RED])
        apply_layout(fig, "Cashback Distribution by Marital Status", 330)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sample = df.sample(min(1500, len(df)), random_state=42)
        fig = px.scatter(sample, x="HourSpendOnApp", y="OrderCount",
                         color=sample["Churn"].map({0: "Retained", 1: "Churned"}),
                         size="CashbackAmount", opacity=0.5,
                         color_discrete_map={"Retained": PRIMARY, "Churned": RED})
        fig.update_layout(legend_title="Status")
        apply_layout(fig, "App Hours vs Order Count  (size = cashback)", 330)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-heading">Missing Value Audit</div>', unsafe_allow_html=True)
    miss = df.isnull().sum()
    miss = miss[miss > 0].reset_index()
    miss.columns = ["Feature", "Missing"]
    if miss.empty:
        st.markdown(f'<div style="color:{GREEN};font-size:0.85rem;padding:8px 0">No missing values detected after imputation.</div>',
                    unsafe_allow_html=True)
    else:
        miss["% Missing"] = (miss["Missing"] / len(df) * 100).round(2)
        st.dataframe(miss, use_container_width=True)

    st.markdown('<div class="sec-heading">Raw Data Sample</div>', unsafe_allow_html=True)
    st.dataframe(df.head(40), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# 2. CUSTOMER SEGMENTATION (ROBUST + CHURN AS OUTCOME, NOT INPUT)
# ════════════════════════════════════════════════════════════════
elif page == "2. Customer Segmentation":
    st.markdown('<div class="page-title">2. Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Unsupervised clustering to identify distinct behavioural archetypes</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="rationale-box">
    <b>How segmentation works here</b><br>
    Customers are grouped using behavioural and profile signals (all available features except identifiers and outcomes).
    <b>Churn is not used to create clusters</b> — it is used afterwards to evaluate which segments are higher risk.
    This avoids circular logic and keeps segments usable for personalisation on active customers.
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # Helpers (robust, no hardcoded feature names)
    # -----------------------------
    def _k_diagnostics(X, k_min=2, k_max=10):
        ks = list(range(k_min, k_max + 1))
        inertias, sils = [], []
        for ki in ks:
            m = KMeans(n_clusters=ki, random_state=42, n_init=10).fit(X)
            inertias.append(m.inertia_)
            sils.append(silhouette_score(X, m.labels_))
        return ks, inertias, sils

    def _segment_summary_auto(df_raw: pd.DataFrame, labels: np.ndarray, max_features=8):
        """
        Segment profile using numeric columns only.
        Automatically selects features that differentiate segments most (variance of segment means).
        """
        tmp = df_raw.copy()
        tmp["Segment"] = labels

        num_cols_local = tmp.select_dtypes(include=np.number).columns.tolist()
        for bad in ["CustomerID", "Segment"]:
            if bad in num_cols_local:
                num_cols_local.remove(bad)

        if not num_cols_local:
            sizes = tmp["Segment"].value_counts().sort_index().to_frame("Count")
            return sizes

        means = tmp.groupby("Segment")[num_cols_local].mean()
        diff = means.var(axis=0).sort_values(ascending=False)
        top_feats = diff.head(max_features).index.tolist()

        summary = means[top_feats].round(2)
        summary.insert(0, "Count", tmp["Segment"].value_counts().sort_index())
        return summary

    def _persona_from_summary(summary: pd.DataFrame):
        feats = [c for c in summary.columns if c != "Count"]
        if not feats:
            return {int(s): {"name": "Customer Group", "desc": "Segment profile derived from data."}
                    for s in summary.index}

        personas = {}
        for seg in summary.index:
            tags = []
            for f in feats[:6]:
                r = float(summary[f].rank(pct=True).loc[seg])
                v = float(summary[f].loc[seg])
                if r >= 0.80:
                    tags.append(f"Higher {f} ({v:.2f})")
                elif r <= 0.20:
                    tags.append(f"Lower {f} ({v:.2f})")

            name = "Balanced Customers"
            if tags:
                if any(t.startswith("Higher") for t in tags):
                    name = "High-Intensity Customers"
                elif any(t.startswith("Lower") for t in tags):
                    name = "Low-Intensity Customers"

            desc = "; ".join(tags[:4]) if tags else "Mixed pattern across key signals."
            personas[int(seg)] = {"name": name, "desc": desc}
        return personas

    # -----------------------------
    # Controls
    # -----------------------------
    c1, c2 = st.columns([1, 4])
    with c1:
        k = st.slider("Segments (k)", 2, 10, int(st.session_state.get("km_k", 4)))
        run = st.button("Run Segmentation")

    # -----------------------------
    # Fit / store labels using global X_seg
    # -----------------------------
    if run or "km_labels" not in st.session_state or st.session_state.get("km_k") != k:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_seg)
        st.session_state["km_labels"] = labels
        st.session_state["km_k"] = k
        st.session_state["km_sil"] = silhouette_score(X_seg, labels)

    labels = st.session_state["km_labels"]
    k = st.session_state["km_k"]
    sil_k = st.session_state["km_sil"]

    # -----------------------------
    # Optimal k diagnostics
    # -----------------------------
    st.markdown('<div class="sec-heading">Optimal k — Elbow & Silhouette Analysis</div>',
                unsafe_allow_html=True)

    ks, inertias, sils = _k_diagnostics(X_seg, 2, 10)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Inertia (Elbow Method)", "Silhouette Score"])
    fig.add_trace(go.Scatter(x=ks, y=inertias, mode="lines+markers",
                              line=dict(color=PRIMARY, width=2),
                              marker=dict(size=7, color=PRIMARY)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ks, y=sils, mode="lines+markers",
                              line=dict(color=GREEN, width=2),
                              marker=dict(size=7, color=GREEN)), row=1, col=2)

    if k in ks:
        idxk = ks.index(k)
        fig.add_trace(go.Scatter(x=[k], y=[inertias[idxk]], mode="markers",
                                 marker=dict(size=11, color=WHITE), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[k], y=[sils[idxk]], mode="markers",
                                 marker=dict(size=11, color=WHITE), showlegend=False), row=1, col=2)

    fig.update_layout(paper_bgcolor=SURFACE, plot_bgcolor=BG,
                      font=dict(family="monospace", color=TEXT, size=11),
                      height=300, showlegend=False, margin=dict(l=40, r=20, t=50, b=40))
    fig.update_xaxes(gridcolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, linecolor=BORDER)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # PCA projection (global X_seg)
    # -----------------------------
    st.markdown(
        f'<div class="sec-heading">PCA Projection — k={k} · Silhouette Score = {sil_k:.3f}</div>',
        unsafe_allow_html=True
    )
    pca2 = PCA(n_components=2, random_state=42)
    X2 = pca2.fit_transform(X_seg)
    df_plot = pd.DataFrame({"PC1": X2[:, 0], "PC2": X2[:, 1],
                            "Segment": ["Segment " + str(l) for l in labels]})
    fig = px.scatter(df_plot, x="PC1", y="PC2", color="Segment", opacity=0.65)
    apply_layout(fig, "Customer Clusters in 2D PCA Space", 430)
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Segment summaries
    # -----------------------------
    st.markdown('<div class="sec-heading">Segment Behavioural Profiles</div>', unsafe_allow_html=True)

    df["Segment"] = labels
    seg_summary = _segment_summary_auto(df, labels, max_features=8)
    st.dataframe(seg_summary, use_container_width=True)

    # -----------------------------
    # Churn as an outcome panel
    # -----------------------------
    if "Churn" in df.columns:
        st.markdown('<div class="sec-heading">Segment Outcomes (Churn is not used for clustering)</div>',
                    unsafe_allow_html=True)

        churn_by_seg = df.groupby("Segment")["Churn"].mean().sort_index()
        overall = float(df["Churn"].mean())

        c1, c2 = st.columns([1.2, 2.8])
        with c1:
            st.markdown(f"""
            <div class="info-box">
            <b>Overall churn rate:</b> {overall*100:.1f}%<br>
            <b>How to read segment churn:</b><br>
            Segment churn rate = % of customers in that segment who churned.
            </div>
            """, unsafe_allow_html=True)

        with c2:
            churn_df = churn_by_seg.reset_index()
            churn_df.columns = ["Segment", "ChurnRate"]
            fig = px.bar(churn_df, x="Segment", y="ChurnRate")
            apply_layout(fig, "Churn Rate by Segment", 260, legend=False)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Radar comparison
    # -----------------------------
    st.markdown('<div class="sec-heading">Segment Radar Comparison</div>', unsafe_allow_html=True)

    radar_feats = [c for c in seg_summary.columns if c != "Count"]
    radar_feats = radar_feats[:6] if len(radar_feats) > 6 else radar_feats

    if len(radar_feats) >= 3:
        norm = seg_summary[radar_feats].copy()
        norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

        fig = go.Figure()
        for seg in norm.index:
            vals = norm.loc[seg, radar_feats].tolist() + [norm.loc[seg, radar_feats[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=radar_feats + [radar_feats[0]],
                fill="toself", name=f"Segment {seg}", opacity=0.55
            ))

        fig.update_layout(paper_bgcolor=SURFACE,
                          polar=dict(bgcolor=BG,
                                     radialaxis=dict(gridcolor=BORDER),
                                     angularaxis=dict(gridcolor=BORDER)),
                          font=dict(family="monospace", color=TEXT, size=11),
                          height=400, legend=dict(bgcolor=SURFACE))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f"""
        <div class="info-box">
        Not enough numeric signals available to render a radar chart (need 3+). Current: {len(radar_feats)}.
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # Personas
    # -----------------------------
    st.markdown('<div class="sec-heading">Persona Definitions</div>', unsafe_allow_html=True)

    personas = _persona_from_summary(seg_summary)
    for seg in sorted(personas.keys()):
        st.markdown(f"""
        <div class="info-box">
        <b>Segment {seg} — {personas[seg]['name']}</b><br>{personas[seg]['desc']}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 3. DIMENSIONALITY REDUCTION (PCA ONLY — shows PC1 + PC2)
# ════════════════════════════════════════════════════════════════
elif page == "3. Dimensionality Reduction":
    st.markdown('<div class="page-title">3. Dimensionality Reduction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Validating latent structure before modelling</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="rationale-box">
    <b>Why PCA?</b><br>
    With 19+ features it's hard to visualise customer data directly. PCA compresses that information into
    2 dimensions so we can see patterns and understand the main behavioural axes (PC1 and PC2).
    </div>
    """, unsafe_allow_html=True)

    # ── PCA variance chart ─────────────────────────────────────────
    pca_feats = [c for c in feat_cols if c != "Churn"]
    X_pca_input = StandardScaler().fit_transform(df_enc[pca_feats].values)

    pca_full = PCA(random_state=42).fit(X_pca_input)
    cumvar   = np.cumsum(pca_full.explained_variance_ratio_)
    n95      = int(np.argmax(cumvar >= 0.95)) + 1

    fig = go.Figure()
    fig.add_bar(
        x=list(range(1, len(cumvar) + 1)),
        y=pca_full.explained_variance_ratio_,
        name="Per Component",
        marker_color=PRIMARY,
        opacity=0.7
    )
    fig.add_scatter(
        x=list(range(1, len(cumvar) + 1)),
        y=cumvar,
        mode="lines+markers",
        name="Cumulative",
        line=dict(color=GREEN, width=2),
        marker=dict(size=5)
    )
    fig.add_hline(
        y=0.95,
        line_dash="dash",
        line_color=RED,
        annotation_text=f"95% threshold — {n95} components",
        annotation_font=dict(color=RED, size=10)
    )
    apply_layout(fig, "PCA — Explained Variance per Component", 340)
    st.plotly_chart(fig, use_container_width=True)

    # ── Segment labels (safe reuse) ────────────────────────────────
    labels = st.session_state.get("km_labels", None)

    # If labels don't exist OR length mismatch, refit a quick kmeans for colouring
    if labels is None or len(labels) != X_pca_input.shape[0]:
        labels = KMeans(n_clusters=4, random_state=42, n_init=10).fit_predict(X_pca_input)

    # ── PCA 2D Projection (two plots) ──────────────────────────────
    st.markdown('<div class="sec-heading">PCA 2D Projection</div>', unsafe_allow_html=True)

    X2 = PCA(n_components=2, random_state=42).fit_transform(X_pca_input)
    df_pca = pd.DataFrame({
        "PC1":     X2[:, 0],
        "PC2":     X2[:, 1],
        "Segment": ["Seg " + str(l) for l in labels],
        "Churn":   df["Churn"].map({0: "Retained", 1: "Churned"}).values,
    })

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(df_pca, x="PC1", y="PC2", color="Segment", opacity=0.6)
        apply_layout(fig, "PCA — Coloured by Segment", 380)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(df_pca, x="PC1", y="PC2", color="Churn", opacity=0.6,
                         color_discrete_map={"Retained": PRIMARY, "Churned": RED})
        apply_layout(fig, "PCA — Coloured by Churn Status", 380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Loadings heatmap ───────────────────────────────────────────
    st.markdown('<div class="sec-heading">Feature Loadings — PC1 &amp; PC2</div>',
                unsafe_allow_html=True)

    loadings = pd.DataFrame(
        pca_full.components_[:2].T,
        index=pca_feats,
        columns=["PC1", "PC2"]
    )

    fig = px.imshow(
        loadings.T,
        color_continuous_scale="RdBu",
        zmin=-0.5, zmax=0.5,
        text_auto=".2f"
    )
    apply_layout(fig, "PCA Feature Loadings Heatmap", 220, legend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── PC1 / PC2 breakdown cards ──────────────────────────────────
    st.markdown('<div class="sec-heading">What PC1 and PC2 Actually Represent</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box">'
        'Each principal component is a weighted combination of the original features. '
        'The cards below list the top drivers (largest absolute loadings). '
        'Positive means the feature increases the component score; negative means it decreases it.'
        '</div>',
        unsafe_allow_html=True
    )

    pc1_sorted = loadings["PC1"].abs().sort_values(ascending=False)
    pc2_sorted = loadings["PC2"].abs().sort_values(ascending=False)

    # Dynamic scaling so bars always show (no invisible bars)
    max_abs = float(np.max(np.abs(loadings.values))) if loadings.size else 1.0
    max_abs = max(max_abs, 1e-9)

    col1, col2 = st.columns(2)

    # ── PC1 column ─────────────────────────────────────────────────
    with col1:
        st.markdown(
            '<div style="background:' + SURFACE + ';border:1px solid ' + BORDER + ';'
            'border-top:3px solid ' + PRIMARY + ';border-radius:4px;'
            'padding:14px 18px;margin-bottom:10px">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
            'color:' + PRIMARY + ';text-transform:uppercase;letter-spacing:0.15em;'
            'margin-bottom:4px">PC1 — Top Feature Drivers</div>'
            '<div style="font-size:0.78rem;color:' + MUTED + ';line-height:1.55">'
            'Largest contributors to PC1 (by absolute loading).</div>'
            '</div>',
            unsafe_allow_html=True
        )

        for feat in pc1_sorted.index[:10]:
            val     = float(loadings.loc[feat, "PC1"])
            colour  = PRIMARY if val >= 0 else RED
            bar_pct = int(min(100, (abs(val) / max_abs) * 100))
            sign    = "+" if val >= 0 else ""
            note    = "Positive → increases PC1 score" if val >= 0 else "Negative → decreases PC1 score"

            st.markdown(
                '<div style="background:#0a0d14;border:1px solid ' + BORDER + ';'
                'border-radius:3px;padding:10px 14px;margin:4px 0">'
                '<div style="display:flex;justify-content:space-between;'
                'align-items:center;margin-bottom:6px">'
                '<span style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;'
                'color:' + WHITE + ';font-weight:500">' + feat + '</span>'
                '<span style="font-family:IBM Plex Mono,monospace;font-size:0.85rem;'
                'color:' + colour + ';font-weight:700">' + sign + str(round(val, 2)) + '</span>'
                '</div>'
                '<div style="background:' + BORDER + ';border-radius:2px;height:5px;width:100%">'
                '<div style="background:' + colour + ';border-radius:2px;height:5px;'
                'width:' + str(bar_pct) + '%"></div></div>'
                '<div style="font-size:0.7rem;color:' + MUTED + ';margin-top:5px">' + note + '</div>'
                '</div>',
                unsafe_allow_html=True
            )

    # ── PC2 column ─────────────────────────────────────────────────
    with col2:
        st.markdown(
            '<div style="background:' + SURFACE + ';border:1px solid ' + BORDER + ';'
            'border-top:3px solid ' + GREEN + ';border-radius:4px;'
            'padding:14px 18px;margin-bottom:10px">'
            '<div style="font-family:IBM Plex Mono,monospace;font-size:0.68rem;'
            'color:' + GREEN + ';text-transform:uppercase;letter-spacing:0.15em;'
            'margin-bottom:4px">PC2 — Top Feature Drivers</div>'
            '<div style="font-size:0.78rem;color:' + MUTED + ';line-height:1.55">'
            'Largest contributors to PC2 (by absolute loading).</div>'
            '</div>',
            unsafe_allow_html=True
        )

        for feat in pc2_sorted.index[:10]:
            val     = float(loadings.loc[feat, "PC2"])
            colour  = GREEN if val >= 0 else RED
            bar_pct = int(min(100, (abs(val) / max_abs) * 100))
            sign    = "+" if val >= 0 else ""
            note    = "Positive → increases PC2 score" if val >= 0 else "Negative → decreases PC2 score"

            st.markdown(
                '<div style="background:#0a0d14;border:1px solid ' + BORDER + ';'
                'border-radius:3px;padding:10px 14px;margin:4px 0">'
                '<div style="display:flex;justify-content:space-between;'
                'align-items:center;margin-bottom:6px">'
                '<span style="font-family:IBM Plex Mono,monospace;font-size:0.8rem;'
                'color:' + WHITE + ';font-weight:500">' + feat + '</span>'
                '<span style="font-family:IBM Plex Mono,monospace;font-size:0.85rem;'
                'color:' + colour + ';font-weight:700">' + sign + str(round(val, 2)) + '</span>'
                '</div>'
                '<div style="background:' + BORDER + ';border-radius:2px;height:5px;width:100%">'
                '<div style="background:' + colour + ';border-radius:2px;height:5px;'
                'width:' + str(bar_pct) + '%"></div></div>'
                '<div style="font-size:0.7rem;color:' + MUTED + ';margin-top:5px">' + note + '</div>'
                '</div>',
                unsafe_allow_html=True
            )

    # ── Key insight callout ────────────────────────────────────────
    st.markdown(
        '<div style="background:#071420;border:1px solid #0f2a40;'
        'border-left:3px solid ' + GREEN + ';border-radius:3px;'
        'padding:14px 18px;margin:16px 0 4px 0;font-size:0.85rem;'
        'color:' + TEXT + ';line-height:1.7">'
        '<span style="color:' + WHITE + ';font-weight:600">Key insight from the loadings: </span>'
        'PC1 reflects the strongest shared behaviour pattern in the dataset; PC2 captures a separate, independent pattern. '
        'Use these axes to understand which customers are high-activity vs low-activity and '
        'high-engagement vs low-engagement.'
        '</div>',
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════
# 4. CHURN PREDICTION
# ════════════════════════════════════════════════════════════════
elif page == "4. Churn Prediction":
    st.markdown('<div class="page-title">4. Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Supervised binary classification — identifying customers at risk of leaving</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="rationale-box">
    <b>Why three models for churn prediction?</b><br>
    Rather than picking one model and hoping it works, three are trained and compared 
    against the same test data. The simplest model (Logistic Regression) acts as a 
    reference point. Random Forest and Gradient Boosting are more powerful and can pick 
    up on complex patterns in the data. The best-performing model is then used in 
    production — with the evidence shown transparently via the ROC curves below.
    </div>
    """, unsafe_allow_html=True)

    target = "Churn"
    fc = [c for c in feat_cols if c != target]
    X_ch = df_enc[fc].values
    y_ch = df_enc[target].values
    X_tr, X_te, y_tr, y_te = train_test_split(X_ch, y_ch, test_size=0.2,
                                               random_state=42, stratify=y_ch)
    sc = StandardScaler()
    X_tr_s, X_te_s = sc.fit_transform(X_tr), sc.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=150, random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, random_state=42),
    }
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_tr_s, y_tr)
        y_prob = mdl.predict_proba(X_te_s)[:, 1]
        y_pred = mdl.predict(X_te_s)
        results[name] = {
            "model": mdl, "y_prob": y_prob, "y_pred": y_pred,
            "auc": roc_auc_score(y_te, y_prob),
            "cv":  cross_val_score(mdl, X_tr_s, y_tr, cv=5, scoring="roc_auc").mean(),
        }

    best_name = max(results, key=lambda n: results[n]["auc"])
    names = list(results.keys())
    cards = "".join([
        f'<div class="kpi"><div class="v">{results[n]["auc"]:.3f}</div>'
        f'<div class="l">{n}<br>Test AUC</div></div>'
        for n in names
    ])
    st.markdown(f'<div class="kpi-row">{cards}</div>', unsafe_allow_html=True)

    fig = go.Figure()
    for (name, r), col_r in zip(results.items(), [PRIMARY, RED, GREEN]):
        fpr, tpr, _ = roc_curve(y_te, r["y_prob"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"{name}  AUC={r['auc']:.3f}",
                                  line=dict(color=col_r, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash", color=MUTED, width=1),
                              name="Random baseline"))
    apply_layout(fig, "ROC Curves — Model Comparison", 400)
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        cm = confusion_matrix(y_te, results[best_name]["y_pred"])
        fig = px.imshow(cm, text_auto=True,
                        color_continuous_scale=[[0, SURFACE], [1, PRIMARY]],
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Retained", "Churned"], y=["Retained", "Churned"])
        apply_layout(fig, f"Confusion Matrix — {best_name}", 320, legend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        df_h = pd.DataFrame({
            "Churn Probability": results[best_name]["y_prob"],
            "Actual": ["Churned" if y else "Retained" for y in y_te]
        })
        fig = px.histogram(df_h, x="Churn Probability", color="Actual",
                            barmode="overlay", nbins=40,
                            color_discrete_map={"Churned": RED, "Retained": PRIMARY})
        apply_layout(fig, "Predicted Churn Probability Distribution", 320)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="info-box">
    <b>Business application:</b> Customers with a predicted churn probability above 0.60 
    are enrolled in a retention flow — a targeted discount surfaced at next login, escalated 
    to a priority support queue, and included in a personalised re-engagement email sequence. 
    The 0.60 threshold balances precision (minimising unnecessary interventions) against recall 
    (catching the majority of genuine churners).
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# 5. FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════
elif page == "5. Feature Importance":
    st.markdown('<div class="page-title">5. Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Understanding which signals drive churn and which data to prioritise</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="rationale-box">
    <b>Why feature importance?</b><br>
    This tells us which pieces of customer data actually matter for predicting churn — 
    and by extension, which behaviours are worth tracking and acting on. Features ranked 
    highly here are the levers the personalisation system should focus on. It also helps 
    prioritise what new data Dohtem should start collecting.
    </div>
    """, unsafe_allow_html=True)

    fc = [c for c in feat_cols if c != "Churn"]
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(df_enc[fc].values, df_enc["Churn"].values)
    imp = pd.DataFrame({"Feature": fc, "Importance": rf.feature_importances_})\
            .sort_values("Importance", ascending=True)

    fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                 color="Importance", color_continuous_scale=[[0, BORDER], [1, PRIMARY]])
    apply_layout(fig, "Feature Importance — Random Forest (Churn Target)", 500, legend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-heading">Correlation Matrix</div>', unsafe_allow_html=True)
    num_only = df_enc.select_dtypes(include=np.number).drop(columns=["CustomerID"], errors="ignore")
    corr = num_only.corr()
    fig = px.imshow(corr, color_continuous_scale="RdBu", zmin=-1, zmax=1, text_auto=".1f")
    apply_layout(fig, "Feature Correlation Heatmap", 540, legend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="sec-heading">Top Signal — Personalisation Mapping</div>',
                unsafe_allow_html=True)
    top5 = imp.tail(5)["Feature"].tolist()[::-1]
    mapping = {
        "Tenure":                   "Short-tenure users receive onboarding guidance; long-tenure users unlock loyalty tier UI elements.",
        "Complain":                 "Customers with recent complaints are shown empathy messaging and a fast-track support entry point.",
        "DaySinceLastOrder":        "Lapsed customers see a re-engagement banner and a personalised offer at next login.",
        "CashbackAmount":           "High-cashback users are shown cashback-eligible products first in the product feed.",
        "SatisfactionScore":        "Low-satisfaction users are served a CSAT survey and a visible support call-to-action.",
        "HourSpendOnApp":           "High app-time users receive push notifications; low-time users are prioritised for email.",
        "OrderCount":               "High-order users receive cross-sell carousels; low-order users receive a discovery-first layout.",
        "CouponUsed":               "Coupon-heavy users see price-sorted results; non-coupon users see quality-led messaging.",
        "NumberOfDeviceRegistered": "Multi-device users see device-relevant categories surfaced in navigation.",
        "CityTier":                 "City tier informs delivery time expectations and determines which local promotions are shown.",
    }
    for feat in top5:
        st.markdown(f"""
        <div class="info-box">
        <b>{feat}</b><br>{mapping.get(feat, "Monitor for future personalisation opportunities.")}
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# 6. PERSONALISATION RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════
elif page == "6. Personalisation Framework":
    st.markdown('<div class="page-title">6. Personalisation Recommendations</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-sub">What Dohtem should do differently — based directly on what the data showed</div>',
        unsafe_allow_html=True
    )

    # ── PCA REFERENCE ─────────────────────────────────────────────
    st.markdown('<div class="sec-heading">What PCA told us — two hidden patterns in the customer base</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    The PCA analysis compressed 19 features into two underlying patterns that explain the most 
    variation across all 5,630 customers. These patterns are not features you can directly observe — 
    they are combinations of features that move together. Understanding them tells us what 
    <em>kind</em> of customers exist before we even build a model.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-top:3px solid {PRIMARY};border-radius:4px;padding:18px 20px">
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:{PRIMARY};
                      text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px">
            PC1 — Purchase Activity Axis
          </div>
          <div style="font-size:0.85rem;color:{TEXT};line-height:1.7">
            The features that load most strongly onto PC1 are 
            <span style="color:{WHITE};font-weight:600">OrderCount (+0.44)</span> and 
            <span style="color:{WHITE};font-weight:600">CouponUsed (+0.41)</span> on the positive side, 
            and <span style="color:{WHITE};font-weight:600">Gender (−0.24)</span> and 
            <span style="color:{WHITE};font-weight:600">PreferredLoginDevice (−0.22)</span> on the negative side.
            <br><br>
            <span style="color:{MUTED}">What this means in plain terms:</span> PC1 is essentially 
            measuring <em>how actively a customer shops</em>. Customers who score high on PC1 
            place more orders and use more coupons. This axis separates casual browsers 
            from frequent buyers — and it is the dominant source of variation in the entire dataset.
            <br><br>
            <span style="color:{MUTED}">Personalisation use:</span> A customer's position on this axis 
            tells you whether to show them a discovery-first homepage (low PC1) or a 
            repeat-purchase, deal-led homepage (high PC1).
          </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};
                    border-top:3px solid {GREEN};border-radius:4px;padding:18px 20px">
          <div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;color:{GREEN};
                      text-transform:uppercase;letter-spacing:0.12em;margin-bottom:10px">
            PC2 — Digital Engagement Axis
          </div>
          <div style="font-size:0.85rem;color:{TEXT};line-height:1.7">
            PC2 is driven by 
            <span style="color:{WHITE};font-weight:600">HourSpendOnApp (+0.47)</span>, 
            <span style="color:{WHITE};font-weight:600">NumberOfDeviceRegistered (+0.41)</span>, and 
            <span style="color:{WHITE};font-weight:600">PreferedOrderCat (+0.43)</span> positively, 
            against <span style="color:{WHITE};font-weight:600">Tenure (−0.33)</span> and 
            <span style="color:{WHITE};font-weight:600">CityTier (−0.21)</span> negatively.
            <br><br>
            <span style="color:{MUTED}">What this means in plain terms:</span> PC2 separates 
            customers by <em>how digitally engaged they are</em>. High-PC2 customers spend 
            more time on the app, use multiple devices, and have a defined category preference. 
            Interestingly, longer-tenure customers score lower here — suggesting established 
            customers may be less digitally active despite being loyal.
            <br><br>
            <span style="color:{MUTED}">Personalisation use:</span> High-PC2 customers are strong 
            candidates for push notifications and app-native experiences. Low-PC2 
            long-tenure customers should be reached via email rather than in-app.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:#071420;border:1px solid #0f2a40;border-radius:4px;
                padding:14px 18px;margin:8px 0 20px 0;font-size:0.85rem;color:{TEXT};line-height:1.6">
    <span style="color:{WHITE};font-weight:600">Why this matters for personalisation:</span>
    Every customer has a position on both axes simultaneously. A customer who is high on PC1 
    (frequent buyer) but low on PC2 (not digitally engaged) is a valuable but hard-to-reach 
    customer — they buy regularly but don't use the app much, so email is the right channel. 
    A customer high on both axes is the ideal target for in-app promotions and push campaigns. 
    These two axes give Dohtem a simple two-dimensional map of their entire customer base 
    that did not exist before this analysis.
    </div>
    """, unsafe_allow_html=True)

    # ── FINDING 1: TENURE ──────────────────────────────────────────
    st.markdown('<div class="sec-heading">Finding 1 — New customers are the highest risk group</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>What the data showed:</b> Tenure was the single most important predictor of churn — 
    by a large margin. Customers in their first few months are significantly more likely to leave 
    than long-standing ones. The data also shows the overall churn rate is 16.8%, meaning roughly 
    1 in 6 customers leaves.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="rationale-box">
        <b>For the customer:</b><br>
        New customers should receive a guided first-visit experience — a simple progress indicator 
        showing them what to explore, a welcome offer tied to their first purchase category, and 
        a clear explanation of how the cashback programme works. The goal is to help them find 
        value quickly, before the habit of using the platform has formed.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="rationale-box">
        <b>For the business:</b><br>
        The first 90 days of a customer's life should be treated as a distinct retention phase. 
        A dedicated onboarding email sequence (days 1, 7, 30) showing personalised product 
        recommendations, cashback progress, and order status summaries would directly address 
        the tenure-churn relationship the model found.
        </div>
        """, unsafe_allow_html=True)

    # ── FINDING 2: CASHBACK ────────────────────────────────────────
    st.markdown('<div class="sec-heading">Finding 2 — Cashback is a retention tool, not just a discount</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>What the data showed:</b> CashbackAmount was the second most important feature. Customers 
    who earn more cashback churn less. Segment 1 — the highest-value group — had an average cashback 
    of £206.92 and an order count of 8.51. The lowest-value segment had £143 average cashback and 
    only 1.8 average orders. The cashback programme is actively keeping high-value customers engaged.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="rationale-box">
        <b>For the customer:</b><br>
        Make cashback progress visible everywhere — on the homepage, in order confirmations, 
        and in the app. Show customers exactly how much cashback they could earn on the items 
        currently in their browsing session. Customers who can see their progress are more 
        likely to complete a purchase to reach the next cashback tier.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="rationale-box">
        <b>For the business:</b><br>
        Customers earning below the average cashback threshold (£177) should receive targeted 
        prompts to shop in higher-cashback categories. This is not a discount — it costs the 
        business only when a purchase is made. Showing low-cashback customers what they are 
        missing is a zero-risk way to increase both order count and retention.
        </div>
        """, unsafe_allow_html=True)

    # ── FINDING 3: COMPLAINTS ──────────────────────────────────────
    st.markdown('<div class="sec-heading">Finding 3 — A complaint that goes unresolved is a churner in waiting</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>What the data showed:</b> Complain was one of the top 5 most important churn predictors. 
    Customers who filed a complaint are disproportionately represented in the churned group. 
    The correlation matrix also shows complaints correlate positively with churn and negatively 
    with tenure — meaning newer customers who complain are at the very highest risk of all.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="rationale-box">
        <b>For the customer:</b><br>
        Any customer who has filed a complaint in the last 30 days should see a different 
        homepage experience — a visible resolution status update at the top of the page, 
        an empathy message acknowledging their experience, and a direct link to support. 
        This shows the customer they have been heard, which is the primary reason 
        post-complaint customers leave.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="rationale-box">
        <b>For the business:</b><br>
        The churn model can be used to flag customers immediately after a complaint is logged — 
        giving the customer service team a ranked list of which complainants are at highest 
        churn risk. Those customers get a proactive outreach call or chat within 24 hours. 
        Resolving complaints fast is cheaper than acquiring a replacement customer.
        </div>
        """, unsafe_allow_html=True)

    # ── FINDING 4: LAPSED CUSTOMERS ───────────────────────────────
    st.markdown('<div class="sec-heading">Finding 4 — Customers go quiet before they leave</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>What the data showed:</b> DaySinceLastOrder was a top-5 churn predictor. Customers who 
    haven't ordered recently are significantly more likely to churn. This is an early warning 
    signal — the customer hasn't left yet, but they've stopped engaging. This is the most 
    actionable finding in the entire dataset because it gives Dohtem a window to intervene 
    before the decision to leave is made.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="rationale-box">
        <b>For the customer:</b><br>
        When a customer logs in after a gap of 14+ days, their homepage should lead with 
        a "welcome back" message and show products based on their last browsed category — 
        not a generic bestsellers list. This tells the customer the platform remembers them 
        and makes returning feel effortless rather than starting from scratch.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="rationale-box">
        <b>For the business:</b><br>
        A re-engagement email triggered at day 14 of inactivity — personalised with the 
        customer's last viewed products and a time-limited cashback boost — directly addresses 
        the DaySinceLastOrder signal. The churn model score determines how aggressive the offer 
        is: high-risk lapsed customers receive a stronger incentive than low-risk ones.
        </div>
        """, unsafe_allow_html=True)

    # ── FINDING 5: THE THREE SEGMENTS ─────────────────────────────
    st.markdown('<div class="sec-heading">Finding 5 — Three distinct customer types need three different experiences</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <b>What the data showed:</b> K-Means identified 3 clear customer groups. Segment 0 (2,286 customers) 
    are long-tenure, multi-address customers — likely families or businesses with complex needs. 
    Segment 1 (838 customers) are high-frequency, high-cashback, high-coupon users — the most 
    commercially valuable group. Segment 2 (2,506 customers) are balanced, lower-engagement customers 
    — the largest group and the one with the most room to grow.
    </div>
    """, unsafe_allow_html=True)

    # Segment summary cards from actual data
    seg_data = {
        "Segment 0": {"size": "2,286", "label": "Established Customers",
                      "signals": "Avg tenure 13.7 · Multiple delivery addresses",
                      "colour": PRIMARY},
        "Segment 1": {"size": "838",   "label": "Power Buyers",
                      "signals": "Avg 8.5 orders · £207 cashback · 4.8 coupons used",
                      "colour": GREEN},
        "Segment 2": {"size": "2,506", "label": "Casual Shoppers",
                      "signals": "Avg 1.8 orders · £143 cashback · lower engagement",
                      "colour": RED},
    }

    cols = st.columns(3)
    for col, (seg, info) in zip(cols, seg_data.items()):
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};
                        border-top:3px solid {info['colour']};border-radius:4px;
                        padding:16px 18px;margin-bottom:12px">
              <div style="font-family:IBM Plex Mono,monospace;font-size:0.7rem;
                          color:{info['colour']};text-transform:uppercase;
                          letter-spacing:0.12em;margin-bottom:6px">{seg}</div>
              <div style="font-size:1.4rem;font-weight:600;color:{WHITE};
                          font-family:IBM Plex Mono,monospace">{info['size']}</div>
              <div style="font-size:0.78rem;color:{WHITE};margin:4px 0">{info['label']}</div>
              <div style="font-size:0.75rem;color:{MUTED};line-height:1.5">{info['signals']}</div>
            </div>
            """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
        <div class="rationale-box">
        <b>Segment 0 — what to do:</b><br>
        These are reliable, long-standing customers with multiple delivery addresses — 
        likely buying for a household or small business. Show them bulk-buy options, 
        multi-address delivery management, and loyalty tier progress. Don't waste their 
        attention with new-user onboarding they don't need.
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="rationale-box">
        <b>Segment 1 — what to do:</b><br>
        This is the most valuable segment. They respond to coupons, earn high cashback, 
        and order frequently. Give them early access to sales, a VIP badge, and 
        personalised product feeds based on their order history. Losing one of these 
        customers costs significantly more than retaining them.
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="rationale-box">
        <b>Segment 2 — what to do:</b><br>
        This is the largest group with the most growth potential. With only 1.8 average 
        orders they are not yet habitual buyers. Show them what they are missing — 
        cashback they could be earning, categories they haven't explored, and 
        a simple first step to get them to their second purchase.
        </div>
        """, unsafe_allow_html=True)

    # ── SUMMARY TABLE ─────────────────────────────────────────────
    st.markdown('<div class="sec-heading">Summary — What changes, for whom, and why</div>',
                unsafe_allow_html=True)
    st.dataframe(pd.DataFrame({
        "Who": [
            "New customers (Tenure < 3 months)",
            "Lapsed customers (14+ days since order)",
            "Customers who complained (last 30 days)",
            "Segment 1 — Power Buyers",
            "Segment 2 — Casual Shoppers",
            "All customers with churn score > 0.60",
        ],
        "What changes on the website": [
            "Guided onboarding banner, cashback explainer, welcome offer",
            "Welcome back message, last-browsed products surfaced first",
            "Complaint resolution status shown at top of homepage",
            "VIP badge, early sale access, order history feed",
            "Discovery layout, cashback progress prompt, second-purchase nudge",
            "Personalised retention offer shown at login",
        ],
        "What changes in communications": [
            "Day 1, 7, 30 email sequence with category recommendations",
            "Re-engagement email with cashback boost at day 14",
            "Proactive support outreach within 24 hours of complaint",
            "Exclusive early-access notifications before public sale",
            "Monthly category inspiration email with cashback potential shown",
            "Targeted discount email before the customer visits a competitor",
        ],
    }), use_container_width=True)


    #python -m streamlit run dohtem_personalisation_app.py <- use this command to run the app