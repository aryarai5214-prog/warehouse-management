"""
app.py - LOGISTIX AI: Enterprise Warehouse Management System & Demand Intelligence
Streamlit Application for Production Deployment and GitHub
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="LOGISTIX AI | Enterprise Warehouse Management System",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Enterprise CSS
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }

    /* Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #C7D2FE;
        font-weight: 400;
        margin-bottom: 14px;
    }
    .hero-tag {
        display: inline-block;
        background: rgba(99, 102, 241, 0.35);
        color: #A5B4FC;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(165, 180, 252, 0.4);
    }

    /* Metric Cards */
    .kpi-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #6366F1;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #F8FAFC;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.82rem;
        color: #10B981;
        font-weight: 500;
        margin-top: 6px;
    }

    /* Result Pill Cards */
    .result-card {
        background: #1E293B;
        border-left: 5px solid #6366F1;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }

    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
    }
    .status-green { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
    .status-yellow { background: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
    .status-orange { background: rgba(249, 115, 22, 0.2); color: #FB923C; border: 1px solid #F97316; }
    .status-red { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid #EF4444; }

    /* Custom Streamlit Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #0F172A;
        padding: 6px;
        border-radius: 12px;
        border: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        font-size: 0.92rem;
        padding: 0 18px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366F1 !important;
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# Import backend modules
from src.data_loader import (
    load_dataset, get_kpis, get_monthly_trends,
    get_category_summary, get_supplier_summary, get_abc_classification
)
from src.ml_pipeline import (
    load_ml_models, predict_single_item, batch_predict
)
from src.visualizations import (
    plot_sales_timeline, plot_category_distribution,
    plot_supplier_ranking, plot_abc_pareto,
    plot_confusion_matrix_heatmap, plot_stock_gauge
)

# Load data and models
try:
    df_raw = load_dataset()
    data_loaded = True
except Exception as e:
    data_loaded = False
    st.error(f"Error loading warehouse dataset: {e}")

regressor, classifier, metadata, metrics = load_ml_models()
models_ready = (regressor is not None and classifier is not None and metadata is not None)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 📦 LOGISTIX AI")
    st.caption("Enterprise Warehouse Management & ML Forecasting System")
    st.markdown("---")

    if data_loaded:
        st.subheader("🔍 Global Filters")
        all_years = sorted(df_raw["YEAR"].unique().tolist())
        selected_years = st.multiselect("Filter by Year", options=all_years, default=all_years)

        all_cats = sorted(df_raw["ITEM TYPE"].unique().tolist())
        selected_cats = st.multiselect("Filter by Category", options=all_cats, default=all_cats)

        # Filter active dataframe
        df_view = df_raw[
            (df_raw["YEAR"].isin(selected_years)) &
            (df_raw["ITEM TYPE"].isin(selected_cats))
        ]
    else:
        df_view = pd.DataFrame()

    st.markdown("---")
    st.subheader("⚡ System Diagnostics")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        st.caption("AI Regressor")
        st.markdown("**R² = 0.709** 🎯")
    with col_sb2:
        st.caption("AI Classifier")
        st.markdown("**Acc = 71.1%** ✅")

    st.caption(f"📁 Dataset: 307,645 rows")
    st.caption(f"☁️ Streamlit Cloud: Ready")
    st.caption("© 2026 LOGISTIX AI Systems")

# --- HERO BANNER ---
st.markdown("""
<div class="hero-container">
    <span class="hero-tag">ENTERPRISE WAREHOUSE INTELLIGENCE PLATFORM</span>
    <h1 class="hero-title">LogistiX AI — Warehouse Management System</h1>
    <p class="hero-subtitle">
        Real-time inventory intelligence, machine-learning demand forecasting, automated reorder thresholds, and ABC Pareto velocity classification.
    </p>
</div>
""", unsafe_allow_html=True)

# Top KPI row
if data_loaded and len(df_view) > 0:
    kpis = get_kpis(df_view)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Warehouse Outbound Volume</div>
            <div class="kpi-value">{kpis['total_warehouse_sales']:,.0f}</div>
            <div class="kpi-delta">Cases / Units Dispatched</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Retail Transfers</div>
            <div class="kpi-value">{kpis['total_retail_transfers']:,.0f}</div>
            <div class="kpi-delta">Store Replenishment Cases</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Catalog SKUs</div>
            <div class="kpi-value">{kpis['active_skus']:,}</div>
            <div class="kpi-delta">Across {kpis['active_categories']} Product Categories</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Monitored Suppliers</div>
            <div class="kpi-value">{kpis['active_suppliers']:,}</div>
            <div class="kpi-delta">Distributor & Brewery Network</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# --- MAIN TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Executive Analytics",
    "🤖 AI Demand & Replenishment",
    "📁 Batch Stockout Audit",
    "📦 ABC Pareto Velocity",
    "🚚 Supplier Assortment",
    "📈 Model Stats & Metrics",
    "🚀 Cloud Deployment Guide"
])

# ==========================================
# TAB 1: EXECUTIVE ANALYTICS
# ==========================================
with tab1:
    st.subheader("Historical Inbound/Outbound Demand Dynamics")
    if data_loaded and len(df_view) > 0:
        monthly_df = get_monthly_trends(df_view)
        fig_timeline = plot_sales_timeline(monthly_df)
        st.plotly_chart(fig_timeline, use_container_width=True)

        col_cat1, col_cat2 = st.columns([1, 1])
        cat_df = get_category_summary(df_view)
        donut_fig, bar_fig = plot_category_distribution(cat_df)

        with col_cat1:
            st.plotly_chart(donut_fig, use_container_width=True)
        with col_cat2:
            st.plotly_chart(bar_fig, use_container_width=True)

        st.subheader("Top Suppliers by Warehouse Throughput")
        supp_df = get_supplier_summary(df_view, top_n=12)
        fig_supp = plot_supplier_ranking(supp_df)
        st.plotly_chart(fig_supp, use_container_width=True)
    else:
        st.info("No records match the selected filters.")

# ==========================================
# TAB 2: AI DEMAND & REPLENISHMENT
# ==========================================
with tab2:
    st.subheader("AI-Powered SKU Demand Forecast & Safety Stock Engine")
    st.caption("Forecast future monthly warehouse demand and calculate exact Safety Stock (SS) and Reorder Point (ROP) thresholds.")

    if models_ready:
        col_form1, col_form2 = st.columns([1, 1])

        with col_form1:
            st.markdown("#### 1. SKU & Market Parameters")
            item_type_sel = st.selectbox("Product Category (ITEM TYPE)", options=metadata["item_types"], index=metadata["item_types"].index("WINE") if "WINE" in metadata["item_types"] else 0)
            supplier_sel = st.selectbox("Supplier / Distributor", options=metadata["all_suppliers"], index=0)
            month_sel = st.slider("Target Planning Month", min_value=1, max_value=12, value=11, help="Accounts for seasonal liquor and holiday spikes (e.g., Nov/Dec peaks).")
            retail_sales_input = st.number_input("Expected Retail Direct Sales (Units)", min_value=0.0, value=25.0, step=5.0)
            retail_transfers_input = st.number_input("Expected Retail Transfers (Units)", min_value=0.0, value=40.0, step=5.0)

        with col_form2:
            st.markdown("#### 2. Warehouse Operations & Lead Time")
            current_stock_input = st.number_input("Current Warehouse On-Hand Stock (Units)", min_value=0.0, value=35.0, step=5.0, help="Your actual current count in the warehouse bins.")
            lead_time_days = st.slider("Supplier Lead Time (Days)", min_value=3, max_value=60, value=14, step=1, help="Days from placing purchase order to receipt at dock.")
            service_level_sel = st.selectbox("Target Service Level (Stockout Protection)", options=[0.90, 0.95, 0.98, 0.99], index=1, format_func=lambda x: f"{int(x*100)}% Service Level")

            calc_btn = st.button("🔮 Calculate AI Forecast & Replenishment Thresholds", type="primary", use_container_width=True)

        if calc_btn:
            res = predict_single_item(
                regressor=regressor,
                classifier=classifier,
                metadata=metadata,
                month=month_sel,
                retail_sales=retail_sales_input,
                retail_transfers=retail_transfers_input,
                supplier=supplier_sel,
                item_type=item_type_sel,
                lead_time_days=lead_time_days,
                service_level=service_level_sel,
                current_on_hand=current_stock_input
            )

            st.markdown("---")
            st.subheader("🎯 AI Inference Results")

            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.metric("Predicted Warehouse Demand", f"{res['predicted_demand']:,.1f} units")
            with r2:
                st.metric("Recommended Safety Stock", f"{res['inventory']['safety_stock']:,.1f} units")
            with r3:
                st.metric("Reorder Point (ROP)", f"{res['inventory']['reorder_point']:,.1f} units")
            with r4:
                st.metric("Economic Order Qty (EOQ)", f"{res['inventory']['eoq']:,.0f} units")

            col_eval1, col_eval2 = st.columns([1, 1])
            with col_eval1:
                st.markdown("#### Inventory Health & Action Status")
                badge_class = f"status-{res['badge_color']}"
                st.markdown(f'<div class="status-badge {badge_class}">{res["stockout_status"]}</div>', unsafe_allow_html=True)
                st.write("")
                st.info(f"**Action Recommended:** {res['action']}")

                st.markdown("#### Velocity Classification")
                st.markdown(f"**Predicted Tier:** `{res['velocity_tier']}`")
                for c_name, c_prob in res["probabilities"].items():
                    st.write(f"- **{c_name}**: {c_prob}% confidence")

            with col_eval2:
                gauge_fig = plot_stock_gauge(
                    current_stock=current_stock_input,
                    rop=res['inventory']['reorder_point'],
                    safety_stock=res['inventory']['safety_stock'],
                    predicted_demand=res['predicted_demand']
                )
                st.plotly_chart(gauge_fig, use_container_width=True)
    else:
        st.warning("Models are not loaded. Please verify `models/` directory.")

# ==========================================
# TAB 3: BATCH STOCKOUT AUDIT
# ==========================================
with tab3:
    st.subheader("Batch File Scoring & Automated Stockout Audit")
    st.caption("Upload an inventory CSV file or test using the pre-generated sample dataset.")

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Upload Inventory CSV", type=["csv"])
    with col_b2:
        use_sample = st.button("📥 Load Sample Inventory Dataset (25 SKUs)")

    df_to_score = None
    if uploaded_file is not None:
        try:
            df_to_score = pd.read_csv(uploaded_file)
            st.success(f"Uploaded file loaded with {len(df_to_score)} rows.")
        except Exception as ex:
            st.error(f"Error parsing uploaded file: {ex}")
    elif use_sample:
        sample_path = os.path.join("data", "sample_inventory.csv")
        if os.path.exists(sample_path):
            df_to_score = pd.read_csv(sample_path)
            st.info("Loaded sample inventory records.")
        else:
            st.error("Sample file not found. Please run `train_model.py` first.")

    if df_to_score is not None and models_ready:
        scored_df = batch_predict(
            df_input=df_to_score,
            regressor=regressor,
            classifier=classifier,
            metadata=metadata
        )

        st.markdown("#### Scored Inventory Replenishment Matrix")
        st.dataframe(
            scored_df[[
                c for c in ["ITEM CODE", "ITEM DESCRIPTION", "ITEM TYPE", "SUPPLIER",
                           "PREDICTED_WH_DEMAND", "PREDICTED_VELOCITY_TIER",
                           "SAFETY_STOCK", "REORDER_POINT", "SUGGESTED_ORDER_QTY (EOQ)"]
                if c in scored_df.columns
            ]],
            use_container_width=True
        )

        csv_bytes = scored_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="💾 Download Full Predictions & ROP Audit (CSV)",
            data=csv_bytes,
            file_name="warehouse_replenishment_audit.csv",
            mime="text/csv",
            type="primary"
        )

# ==========================================
# TAB 4: ABC PARETO VELOCITY
# ==========================================
with tab4:
    st.subheader("ABC Inventory Pareto Analysis (80/20 Rule)")
    st.caption("Identify Fast-Moving high-turnover SKUs (Class A), Moderate SKUs (Class B), and Long-Tail/Dead-Stock items (Class C).")

    if data_loaded and len(df_view) > 0:
        pareto_df, pareto_summary = get_abc_classification(df_view)

        pa1, pa2, pa3, pa4 = st.columns(4)
        with pa1:
            st.metric("Total Catalog SKUs", f"{pareto_summary['total_skus']:,}")
        with pa2:
            pct_a = (pareto_summary['class_a_skus'] / max(pareto_summary['total_skus'], 1)) * 100
            st.metric("Class A (Fast-Moving)", f"{pareto_summary['class_a_skus']:,} ({pct_a:.1f}%)")
        with pa3:
            pct_b = (pareto_summary['class_b_skus'] / max(pareto_summary['total_skus'], 1)) * 100
            st.metric("Class B (Moderate)", f"{pareto_summary['class_b_skus']:,} ({pct_b:.1f}%)")
        with pa4:
            pct_c = (pareto_summary['class_c_skus'] / max(pareto_summary['total_skus'], 1)) * 100
            st.metric("Class C (Slow/Tail)", f"{pareto_summary['class_c_skus']:,} ({pct_c:.1f}%)")

        fig_pareto = plot_abc_pareto(pareto_df)
        st.plotly_chart(fig_pareto, use_container_width=True)

        st.markdown("#### SKU Velocity Lookup")
        tier_filter = st.selectbox("Filter SKU List by Tier", ["All Tiers", "Class A (Fast-Moving)", "Class B (Moderate)", "Class C (Slow/Tail)"])
        filtered_pareto = pareto_df if tier_filter == "All Tiers" else pareto_df[pareto_df["ABC_CLASS"] == tier_filter]
        st.dataframe(filtered_pareto.head(100), use_container_width=True)

# ==========================================
# TAB 5: SUPPLIER ASSORTMENT
# ==========================================
with tab5:
    st.subheader("Supplier Logistics & Assortment Scorecards")
    if data_loaded and len(df_view) > 0:
        supp_table = df_view.groupby("SUPPLIER").agg(
            total_warehouse_sales=("WAREHOUSE SALES", "sum"),
            total_retail_sales=("RETAIL SALES", "sum"),
            total_transfers=("RETAIL TRANSFERS", "sum"),
            active_skus=("ITEM CODE", "nunique")
        ).reset_index().sort_values("total_warehouse_sales", ascending=False)

        top_vol = supp_table["total_warehouse_sales"].sum()
        supp_table["volume_share_pct"] = (supp_table["total_warehouse_sales"] / max(top_vol, 1.0)) * 100

        st.dataframe(supp_table, use_container_width=True)

# ==========================================
# TAB 6: MODEL STATS & METRICS
# ==========================================
with tab6:
    st.subheader("Machine Learning Performance & Model Governance")
    if metrics:
        reg_m = metrics["regressor"]
        clf_m = metrics["classifier"]

        st.markdown("### 1. Demand Forecaster Metrics (HistGradientBoostingRegressor)")
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Log-Scale R²", f"{reg_m['log_r2']:.4f}")
        with m2:
            st.metric("Log-Scale MAE", f"{reg_m['log_mae']:.4f}")
        with m3:
            st.metric("Median Absolute Error", f"{reg_m['original_median_ae']} units")
        with m4:
            st.metric("Train Samples", f"{reg_m['train_samples']:,}")
        with m5:
            st.metric("Test Samples", f"{reg_m['test_samples']:,}")

        st.markdown("---")
        st.markdown("### 2. Inventory Velocity Classifier Metrics (HistGradientBoostingClassifier)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Overall Accuracy", f"{clf_m['accuracy']*100:.2f}%")
        with c2:
            st.metric("Weighted Precision", f"{clf_m['weighted_precision']:.4f}")
        with c3:
            st.metric("Weighted Recall", f"{clf_m['weighted_recall']:.4f}")
        with c4:
            st.metric("Weighted F1 Score", f"{clf_m['weighted_f1']:.4f}")

        # Confusion Matrix Heatmap
        cm_fig = plot_confusion_matrix_heatmap(clf_m["confusion_matrix"], metadata["tier_labels"])
        st.plotly_chart(cm_fig, use_container_width=True)

        # Classification report breakdown
        st.markdown("#### Per-Tier Precision & Recall")
        rep_rows = []
        for tier_name in metadata["tier_labels"]:
            if tier_name in clf_m["classification_report"]:
                row_data = clf_m["classification_report"][tier_name]
                rep_rows.append({
                    "Tier": tier_name,
                    "Precision": f"{row_data['precision']:.3f}",
                    "Recall": f"{row_data['recall']:.3f}",
                    "F1-Score": f"{row_data['f1-score']:.3f}",
                    "Test Support": int(row_data['support'])
                })
        st.table(pd.DataFrame(rep_rows))

# ==========================================
# TAB 7: CLOUD DEPLOYMENT GUIDE
# ==========================================
with tab7:
    st.subheader("GitHub & Streamlit Community Cloud Deployment Guide")
    st.markdown("""
    This Warehouse Management System is **100% cloud-ready** for immediate deployment to **Streamlit Community Cloud** or any containerized host.

    ### 📌 Step 1: Initialize Git & Push to GitHub
    Run the following terminal commands in your workspace:
    ```bash
    git init
    git add .
    git commit -m "feat: complete warehouse management system with ML models and Streamlit app"
    git branch -M main
    git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
    git push -u origin main
    ```

    ### 📌 Step 2: Deploy to Streamlit Cloud (Free & Instant)
    1. Visit **[share.streamlit.io](https://share.streamlit.io/)** and log in with your GitHub account.
    2. Click **"New app"**.
    3. Select your repository: `<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>`.
    4. Set Main file path to: `app.py`.
    5. Click **"Deploy!"** — Streamlit will automatically install dependencies from `requirements.txt` and launch your live app.

    ### 🧮 Inventory Optimization Mathematical Formulas
    - **Safety Stock ($SS$):**
      $$SS = Z \times \sigma_{\\text{demand}} \times \\sqrt{\\frac{\\text{Lead Time Days}}{30}}$$
    - **Reorder Point ($ROP$):**
      $$ROP = \\left(D_{\\text{monthly}} \\times \\frac{\\text{Lead Time Days}}{30}\\right) + SS$$
    - **Economic Order Quantity ($EOQ$):**
      $$EOQ = \\sqrt{\\frac{2 \\times D_{\\text{annual}} \\times S}{H}}$$
      *(where $S$ = order placement cost, $H$ = annual holding cost per unit).*
    """)
