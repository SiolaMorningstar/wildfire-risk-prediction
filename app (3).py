import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import folium
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from streamlit_folium import st_folium
from folium.plugins import HeatMap

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Wildfire Risk Prediction",
    page_icon  = "🔥",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Load all data (cached so it only loads once) ─────────────
@st.cache_data
def load_data():
    risk_df = pd.read_csv('data/risk_predictions.csv')
    with open('data/metrics.json')       as f: metrics = json.load(f)
    with open('data/phase3_config.json') as f: config  = json.load(f)
    with open('data/top_shap_dims.json') as f: shap    = json.load(f)
    return risk_df, metrics, config, shap

@st.cache_resource
def load_models():
    model  = joblib.load('models/xgboost_tuned.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

risk_df, metrics, config, top_shap_dims = load_data()
model,   scaler                         = load_models()

# ── Constants ─────────────────────────────────────────────────
TIER_COLORS = {
    'Critical' : '#c0392b',
    'High'     : '#e67e22',
    'Moderate' : '#f39c12',
    'Low'      : '#2ecc71',
}
TIER_ORDER    = ['Critical', 'High', 'Moderate', 'Low']
MODEL_NAMES   = list(metrics.keys())
MODEL_COLORS  = ['#95a5a6', '#3498db', '#e67e22', '#c0392b']
EMBEDDING_COLS= config['embedding_cols']

tier_counts = risk_df['risk_tier'].value_counts()

# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://github.com/SiolaMorningstar/wildfire-risk-prediction",
        "/blob/main/Flames%20and%20hills%20emblem%20logo.png"
        width=80
    )
    st.title("🔥 Wildfire Risk")
    st.caption("AlphaEarth + XGBoost · California 2023")
    st.divider()
 
    page = st.radio(
        "Navigate",
        ["🏠 Overview",
         "🗺️ Risk Map",
         "📊 Model Performance",
         "🔍 SHAP Insights",
         "🎯 Risk Predictor"],
        label_visibility = "collapsed"
    )
 
    st.divider()
    st.markdown("**Best Model**")
    st.metric("ROC-AUC",  config['best_auc'])
    st.metric("Threshold", config['best_threshold'])
    st.divider()
    st.caption(
        "Data: NASA FIRMS via Google Earth Engine  \n"
        "Embeddings: AlphaEarth Foundations V1  \n"
        "Model: XGBoost (tuned)"
    )

# ════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════
if page == "🏠 Overview":

    st.title("🔥 California Wildfire Risk Prediction System")
    st.markdown(
        "Built using **NASA FIRMS** active fire detections and "
        "**Google DeepMind's AlphaEarth Foundations** 64-dimensional "
        "satellite embeddings, trained with **XGBoost**."
    )

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Best AUC",       config['best_auc'])
    c2.metric("Embedding dims", "64")
    c3.metric("Grid points",    len(risk_df))
    c4.metric("Critical zones", tier_counts.get('Critical', 0))
    c5.metric("Models trained", "4")
    c6.metric("Threshold",      config['best_threshold'])

    st.divider()

    # Risk tier cards
    st.subheader("Risk Tier Summary")
    col1, col2, col3, col4 = st.columns(4)
    cols = [col1, col2, col3, col4]

    for col, tier in zip(cols, TIER_ORDER):
        count = tier_counts.get(tier, 0)
        pct   = count / len(risk_df) * 100
        col.markdown(
            f"""<div style="background:{TIER_COLORS[tier]};
                padding:1.2rem; border-radius:10px;
                text-align:center; color:white;">
                <div style="font-size:2rem;font-weight:700">
                    {count}</div>
                <div style="font-size:0.85rem;opacity:0.9">
                    {tier} Risk</div>
                <div style="font-size:0.8rem;opacity:0.8">
                    {pct:.1f}% of area</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.divider()

    # Project pipeline
    st.subheader("Project Pipeline")
    p1, p2, p3, p4, p5 = st.columns(5)
    phases = [
        ("📡", "Phase 1", "Data ingestion",
         "NASA FIRMS + AlphaEarth via Earth Engine"),
        ("⚙️", "Phase 2", "Preprocessing",
         "Spatial join · sampling · scaling"),
        ("🤖", "Phase 3", "Model training",
         "LR · RF · XGBoost · tuning · SHAP"),
        ("🗺️", "Phase 4", "Risk mapping",
         "Grid inference · folium · validation"),
        ("📊", "Phase 5", "Reporting",
         "Dashboard · HTML report · deployment"),
    ]
    for col, (icon, title, sub, desc) in zip(
            [p1,p2,p3,p4,p5], phases):
        col.markdown(
            f"""<div style="background:#f8f9fa;
                padding:1rem; border-radius:8px;
                border-top:3px solid #c0392b;
                text-align:center;">
                <div style="font-size:1.6rem">{icon}</div>
                <div style="font-weight:700;font-size:0.9rem;
                    margin:0.3rem 0">{title}</div>
                <div style="font-size:0.8rem;color:#c0392b;
                    font-weight:600">{sub}</div>
                <div style="font-size:0.75rem;color:#777;
                    margin-top:0.3rem">{desc}</div>
            </div>""",
            unsafe_allow_html=True
        )

# ════════════════════════════════════════════════════════════
# PAGE 2 — RISK MAP
# ════════════════════════════════════════════════════════════
elif page == "🗺️ Risk Map":

    st.title("🗺️ Interactive Wildfire Risk Map")
    st.caption("California — June to September 2023 fire season")

    # Map controls
    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        map_type = st.radio(
            "Map layer",
            ["Heatmap", "Risk tier points"],
            horizontal=True
        )
    with col_ctrl2:
        show_critical = st.checkbox(
            "Highlight Critical zones", value=True
        )

    # Build folium map
    m = folium.Map(
        location   = [37.5, -119.5],
        zoom_start = 6,
        tiles      = 'CartoDB positron'
    )

    if map_type == "Heatmap":
        heat_data = [
            [r['lat'], r['lon'], r['risk_prob']]
            for _, r in risk_df.iterrows()
        ]
        HeatMap(
            heat_data,
            min_opacity = 0.3,
            radius      = 18,
            blur        = 15,
            gradient    = {
                '0.0' : 'green',
                '0.4' : 'yellow',
                '0.65': 'orange',
                '1.0' : 'red'
            }
        ).add_to(m)

    else:
        for tier in TIER_ORDER:
            subset = risk_df[risk_df['risk_tier'] == tier]
            fg     = folium.FeatureGroup(
                name=f"{tier} risk", show=True
            )
            for _, row in subset.iterrows():
                folium.CircleMarker(
                    location     = [row['lat'], row['lon']],
                    radius       = 5,
                    color        = TIER_COLORS[tier],
                    fill         = True,
                    fill_color   = TIER_COLORS[tier],
                    fill_opacity = 0.7,
                    tooltip      = (
                        f"{tier} Risk: {row['risk_pct']}%"
                    )
                ).add_to(fg)
            fg.add_to(m)
        folium.LayerControl().add_to(m)

    if show_critical:
        critical = risk_df[risk_df['risk_tier'] == 'Critical']
        for _, row in critical.iterrows():
            folium.CircleMarker(
                location     = [row['lat'], row['lon']],
                radius       = 8,
                color        = 'white',
                weight       = 2,
                fill         = True,
                fill_color   = '#c0392b',
                fill_opacity = 0.9,
                tooltip      = folium.Tooltip(
                    f"<b>⚠️ Critical Fire Risk</b><br>"
                    f"Risk: {row['risk_pct']}%<br>"
                    f"Lat: {row['lat']:.3f}  "
                    f"Lon: {row['lon']:.3f}"
                )
            ).add_to(m)

    st_folium(m, width=1100, height=560)

    # Stats below map
    st.divider()
    st.subheader("Risk Statistics")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Mean risk prob",
              f"{risk_df['risk_prob'].mean():.3f}")
    s2.metric("Max risk prob",
              f"{risk_df['risk_prob'].max():.3f}")
    s3.metric("High + Critical pts",
              tier_counts.get('High',0) +
              tier_counts.get('Critical',0))
    s4.metric("Grid resolution", "10 km")

    # Risk distribution histogram
    st.subheader("Risk Probability Distribution")
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.hist(risk_df['risk_prob'], bins=50,
            color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(x=config['best_threshold'],
               color='crimson', linewidth=2,
               linestyle='--',
               label=f"Threshold = {config['best_threshold']:.2f}")
    ax.set_xlabel('Predicted fire risk probability')
    ax.set_ylabel('Grid point count')
    ax.legend()
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":

    st.title("📊 Model Performance Comparison")

    # Metrics table
    st.subheader("All Models — Metrics Summary")
    rows = []
    for name in MODEL_NAMES:
        m = metrics[name]
        if 'roc_auc' not in m:
            continue
        rows.append({
            'Model'         : name,
            'ROC-AUC'       : m['roc_auc'],
            'Avg Precision' : m.get('avg_precision', '—'),
            'F1'            : m['f1'],
            'Precision'     : m['precision'],
            'Recall'        : m['recall'],
            'MCC'           : m.get('mcc', '—'),
        })

    metrics_df = pd.DataFrame(rows)
    st.dataframe(
        metrics_df.style.highlight_max(
            subset=['ROC-AUC','F1','Precision','Recall'],
            color='#fff3cd'
        ).format(precision=4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # AUC bar chart
    st.subheader("ROC-AUC by Model")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    auc_vals   = [metrics[n]['roc_auc'] for n in MODEL_NAMES]
    short_lbls = ['LR', 'Random\nForest',
                  'XGBoost', 'XGBoost\n(tuned)']

    bars = axes[0].bar(
        short_lbls, auc_vals,
        color=MODEL_COLORS, edgecolor='white', width=0.55
    )
    axes[0].set_ylim([0.6, 1.0])
    axes[0].set_ylabel('ROC-AUC')
    axes[0].set_title('ROC-AUC Comparison',
                      fontweight='bold')
    axes[0].axhline(y=auc_vals[0], color='gray',
                    linestyle='--', linewidth=0.8,
                    alpha=0.5, label='Baseline')
    axes[0].grid(True, alpha=0.2, axis='y')
    for bar, val in zip(bars, auc_vals):
        axes[0].text(
            bar.get_x() + bar.get_width()/2,
            val + 0.005, f'{val:.4f}',
            ha='center', va='bottom',
            fontsize=9, fontweight='bold'
        )

    # AUC progression line
    axes[1].plot(
        range(len(auc_vals)), auc_vals,
        'o-', color='steelblue', linewidth=2.5,
        markersize=10, markerfacecolor='white',
        markeredgewidth=2.5
    )
    axes[1].fill_between(
        range(len(auc_vals)), auc_vals,
        min(auc_vals)-0.02,
        alpha=0.1, color='steelblue'
    )
    axes[1].set_xticks(range(len(MODEL_NAMES)))
    axes[1].set_xticklabels(short_lbls, fontsize=9)
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title('AUC Progression',
                      fontweight='bold')
    axes[1].set_ylim([min(auc_vals)-0.05, 1.0])
    axes[1].grid(True, alpha=0.2)
    for i, val in enumerate(auc_vals):
        axes[1].annotate(
            f'{val:.4f}',
            xy=(i, val), xytext=(0, 10),
            textcoords='offset points',
            ha='center', fontsize=9,
            color=MODEL_COLORS[i],
            fontweight='bold'
        )

    st.pyplot(fig)
    plt.close()

    # F1 / Precision / Recall bars
    st.subheader("F1 · Precision · Recall")
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4))
    for ax, key, title in zip(
        axes2,
        ['f1', 'precision', 'recall'],
        ['F1 Score', 'Precision', 'Recall']
    ):
        vals = [metrics[n][key] for n in MODEL_NAMES]
        bars = ax.bar(short_lbls, vals,
                      color=MODEL_COLORS,
                      edgecolor='white', width=0.55)
        ax.set_ylim([0, 1.05])
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                val + 0.02, f'{val:.3f}',
                ha='center', va='bottom', fontsize=9
            )

    st.pyplot(fig2)
    plt.close()

# ════════════════════════════════════════════════════════════
# PAGE 4 — SHAP INSIGHTS
# ════════════════════════════════════════════════════════════
elif page == "🔍 SHAP Insights":

    st.title("🔍 SHAP & AlphaEarth Embedding Insights")
    st.markdown(
        "SHAP (SHapley Additive exPlanations) reveals which of the "
        "**64 AlphaEarth embedding dimensions** most influence "
        "wildfire risk predictions."
    )

    n_dims = st.slider(
        "Number of top dims to display", 5, 20, 15
    )

    shap_dims = list(top_shap_dims.keys())[:n_dims]
    shap_vals = list(top_shap_dims.values())[:n_dims]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Bar chart
    colors = plt.cm.YlOrRd(
        np.linspace(0.3, 0.95, n_dims)
    )[::-1]
    axes[0].barh(
        shap_dims[::-1], shap_vals[::-1],
        color=colors, edgecolor='white'
    )
    axes[0].set_xlabel('Mean |SHAP value|')
    axes[0].set_title(
        f'Top {n_dims} AlphaEarth Dims\n'
        f'(Fire Risk Importance)',
        fontweight='bold'
    )
    axes[0].grid(True, alpha=0.2, axis='x')

    # Cumulative importance
    cumulative = np.cumsum(shap_vals) / sum(
        top_shap_dims.values()
    ) * 100
    axes[1].plot(
        range(1, n_dims+1), cumulative,
        'o-', color='crimson', linewidth=2,
        markersize=6
    )
    axes[1].axhline(y=80, color='gray', linestyle='--',
                    linewidth=1, label='80% threshold')
    axes[1].fill_between(
        range(1, n_dims+1), cumulative,
        alpha=0.15, color='crimson'
    )
    axes[1].set_xlabel('Number of top dims included')
    axes[1].set_ylabel('Cumulative importance (%)')
    axes[1].set_title(
        'Cumulative SHAP Importance',
        fontweight='bold'
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    st.pyplot(fig)
    plt.close()

    # Top dims table
    st.subheader("Top Embedding Dimensions")
    shap_df = pd.DataFrame({
        'Rank'          : range(1, len(shap_dims)+1),
        'Embedding Dim' : shap_dims,
        'Mean |SHAP|'   : [round(v, 6) for v in shap_vals],
        'Relative Imp.' : [
            f"{v/shap_vals[0]*100:.1f}%"
            for v in shap_vals
        ]
    })
    st.dataframe(shap_df, use_container_width=True,
                 hide_index=True)

    st.info(
        "💡 AlphaEarth embedding dimensions have no direct "
        "physical label — they are learned representations "
        "encoding vegetation, moisture, terrain, and climate "
        "signals from multi-sensor satellite data."
    )

# ════════════════════════════════════════════════════════════
# PAGE 5 — RISK PREDICTOR
# ════════════════════════════════════════════════════════════
elif page == "🎯 Risk Predictor":

    st.title("🎯 Location Risk Predictor")
    st.markdown(
        "Enter a California location to get its predicted "
        "wildfire risk based on AlphaEarth embeddings. "
        "This uses the **nearest grid point** from our "
        "pre-computed predictions."
    )

    # ── Session state init ────────────────────────────────────
    if 'lat_input' not in st.session_state:
        st.session_state['lat_input'] = 37.77
    if 'lon_input' not in st.session_state:
        st.session_state['lon_input'] = -122.42
    if 'pred_result' not in st.session_state:
        st.session_state['pred_result'] = None

    # ── Preset buttons — update session_state keys directly ──
    st.markdown("**Or pick a known location:**")
    presets = {
        "San Francisco"  : (37.7749, -122.4194),
        "Los Angeles"    : (34.0522, -118.2437),
        "Sacramento"     : (38.5816, -121.4944),
        "Redding"        : (40.5865, -122.3917),
        "Tahoe (Sierra)" : (38.9399, -119.9772),
    }
    preset_cols = st.columns(len(presets))
    for col, (name, (lat, lon)) in zip(preset_cols, presets.items()):
        if col.button(name):
            st.session_state['lat_input'] = lat
            st.session_state['lon_input'] = lon
            st.session_state['pred_result'] = None

    # ── Inputs bound directly to session_state keys ──────────
    # When a preset button sets session_state['lat_input'],
    # Streamlit reruns and number_input reads the updated value
    col_lat, col_lon = st.columns(2)
    with col_lat:
        user_lat = st.number_input(
            "Latitude",
            min_value=32.5, max_value=42.0,
            step=0.01,
            format="%.4f",
            key="lat_input"          # bound to session_state
        )
    with col_lon:
        user_lon = st.number_input(
            "Longitude",
            min_value=-124.5, max_value=-114.0,
            step=0.01,
            format="%.4f",
            key="lon_input"          # bound to session_state
        )

    if st.button("🔍 Predict Risk", type="primary"):
        distances = np.sqrt(
            (risk_df['lat'] - user_lat)**2 +
            (risk_df['lon'] - user_lon)**2
        )
        nearest = risk_df.loc[distances.idxmin()]
        dist_km = distances.min() * 111
        nearby  = risk_df[
            (risk_df['lat'].between(user_lat-1, user_lat+1)) &
            (risk_df['lon'].between(user_lon-1, user_lon+1))
        ]
        st.session_state['pred_result'] = {
            'nearest' : nearest.to_dict(),
            'dist_km' : dist_km,
            'user_lat': user_lat,
            'user_lon': user_lon,
            'nearby'  : nearby.to_dict('records'),
        }

    if st.session_state.get('pred_result') is not None:
        r        = st.session_state['pred_result']
        nearest  = r['nearest']
        dist_km  = r['dist_km']
        user_lat = r['user_lat']
        user_lon = r['user_lon']
        nearby   = pd.DataFrame(r['nearby'])

        tier  = nearest['risk_tier']
        prob  = nearest['risk_prob']
        color = TIER_COLORS[tier]

        st.divider()
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(
                f"""<div style="background:{color};
                    padding:2rem; border-radius:12px;
                    text-align:center; color:white;">
                    <div style="font-size:3rem">
                      {'🔴' if tier=='Critical'
                       else '🟠' if tier=='High'
                       else '🟡' if tier=='Moderate'
                       else '🟢'}
                    </div>
                    <div style="font-size:1.8rem;
                        font-weight:700;margin:0.5rem 0">
                        {tier} Risk
                    </div>
                    <div style="font-size:1.2rem">
                        {prob*100:.1f}% probability
                    </div>
                </div>""",
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown("**📍 Input location**")
            st.markdown(
                f"Lat: `{user_lat:.4f}`  "
                f"Lon: `{user_lon:.4f}`"
            )
            st.markdown("**📌 Nearest grid point**")
            st.markdown(
                f"Lat: `{nearest['lat']:.4f}`  "
                f"Lon: `{nearest['lon']:.4f}`  "
                f"({dist_km:.1f} km away)"
            )
            st.markdown(f"**🔥 Risk probability:** `{prob:.4f}`")
            st.markdown(f"**⚡ Risk tier:** `{tier}`")
            st.markdown(
                f"**📏 Decision threshold:** "
                f"`{config['best_threshold']:.2f}`"
            )
            st.progress(float(prob),
                        text=f"Risk level: {prob*100:.1f}%")

        st.subheader("Location on risk map")
        mini_map = folium.Map(
            location   = [user_lat, user_lon],
            zoom_start = 9,
            tiles      = 'CartoDB positron'
        )

        for _, row in nearby.iterrows():
            folium.CircleMarker(
                location     = [row['lat'], row['lon']],
                radius       = 6,
                color        = TIER_COLORS[row['risk_tier']],
                fill         = True,
                fill_color   = TIER_COLORS[row['risk_tier']],
                fill_opacity = 0.6,
                tooltip      = f"{row['risk_tier']}: {row['risk_pct']}%"
            ).add_to(mini_map)

        folium.Marker(
            location = [user_lat, user_lon],
            tooltip  = "Your location",
            icon     = folium.Icon(
                color='red', icon='fire', prefix='fa'
            )
        ).add_to(mini_map)

        st_folium(mini_map, width=700, height=350,
                  key="mini_map")
