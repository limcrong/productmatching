import sys
from pathlib import Path

# Ensure repository root and app directory are at head of sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
APP_DIR = ROOT_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    TEST_DEMO_SPLIT_CSV,
)
from src.app_config import (
    load_results_records,
    load_final_config,
    load_latency_benchmark,
    load_error_cases,
    load_sweeps_data,
)

try:
    from app.components.product_card import render_product_card
except (ModuleNotFoundError, ImportError):
    from components.product_card import render_product_card

st.set_page_config(page_title="Experiment Results | Demo", page_icon="📊", layout="wide")

st.title("📊 Academic Experiment Results & Progression")
st.caption(
    "Quantitative benchmark across classical baseline, unimodal deep learning, and multimodal fusion approaches."
)

experiments = load_results_records()
final_cfg = load_final_config()
latency_data = load_latency_benchmark()
error_cases = load_error_cases()
sweeps_data = load_sweeps_data()

if not experiments:
    st.warning(
        "⚠️ No benchmark records found in `experiments/results.json`.\n\n"
        "Run `python scripts/run_experiments.py --experiment 0` for the baseline, or "
        "execute `python scripts/kaggle_run.py` to run the full GPU pipeline."
    )
    st.stop()

# -----------------------------------------------------------------
# 1. Quantitative Benchmark Table
# -----------------------------------------------------------------
st.subheader("1️⃣ Quantitative Benchmark on Held-Out Test Set")
st.caption("Unbiased evaluation measured strictly on previously unseen product groups (zero leakage).")

table_rows = []
for exp in experiments:
    tm = exp.get("test_metrics", {})
    vm = exp.get("validation_metrics", {})
    hp = exp.get("hyperparameters", {})
    delta = exp.get("delta_val_f1", 0.0)
    delta_str = f"{delta:+.3f}" if delta != 0.0 else "-"

    hp_summary = ", ".join(f"{k}={v}" for k, v in hp.items() if k not in ["dimension", "top_k"])

    table_rows.append({
        "Experiment": exp.get("name", exp.get("experiment_id")),
        "Approach": exp.get("approach_type", ""),
        "Model": exp.get("model_name", ""),
        "Hyperparameters": hp_summary or "-",
        "Val F1": f"{vm.get('mean_f1', 0.0):.3f}",
        "Test Precision": f"{tm.get('mean_precision', 0.0):.3f}",
        "Test Recall": f"{tm.get('mean_recall', 0.0):.3f}",
        "Test F1": f"{tm.get('mean_f1', 0.0):.3f}",
        "Δ Val F1": delta_str,
        "Selected": "⭐ Deployed" if exp.get("selected") else "",
    })

df_table = pd.DataFrame(table_rows)
st.dataframe(df_table, use_container_width=True)

if len(experiments) == 1 and experiments[0].get("experiment_id") == "exp_00":
    st.info(
        "📌 **Experimental Pipeline Notice:** Experiment 0 (TF-IDF Baseline) is complete. "
        "Experiments 1 through 5 (SentenceTransformers, timm/CLIP, learned multimodal fusion) "
        "are dispatched to Kaggle GPU via `python scripts/kaggle_run.py`."
    )

# -----------------------------------------------------------------
# 2. Performance Progression & Generalization Gap
# -----------------------------------------------------------------
st.subheader("2️⃣ Performance Progression & Generalization Analysis")

plot_data = []
for exp in experiments:
    tm = exp.get("test_metrics", {})
    vm = exp.get("validation_metrics", {})
    label = exp.get("experiment_id", "")
    plot_data.append({
        "ID": label.upper(),
        "Experiment": exp.get("name", label).split("—")[0].strip(),
        "Validation F1": vm.get("mean_f1", 0.0),
        "Test F1": tm.get("mean_f1", 0.0),
        "Test Precision": tm.get("mean_precision", 0.0),
        "Test Recall": tm.get("mean_recall", 0.0),
    })

plot_df = pd.DataFrame(plot_data)

c1, c2 = st.columns(2)
with c1:
    fig_f1 = go.Figure()
    fig_f1.add_trace(go.Scatter(
        x=plot_df["ID"], y=plot_df["Validation F1"],
        mode="lines+markers", name="Validation F1",
        line=dict(color="#3b82f6", width=3, dash="dot"),
        marker=dict(size=8)
    ))
    fig_f1.add_trace(go.Scatter(
        x=plot_df["ID"], y=plot_df["Test F1"],
        mode="lines+markers", name="Test F1 (Unseen Groups)",
        line=dict(color="#10b981", width=3),
        marker=dict(size=10)
    ))
    fig_f1.update_layout(
        title="Validation vs. Test F1 Progression (Generalization Check)",
        xaxis_title="Experimental Stage",
        yaxis_title="F1 Score",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_f1, use_container_width=True)

with c2:
    fig_bar = px.bar(
        plot_df,
        x="ID",
        y=["Test Precision", "Test Recall", "Test F1"],
        barmode="group",
        title="Test Metric Breakdown (Precision vs. Recall vs. F1)",
        color_discrete_map={
            "Test Precision": "#6366f1",
            "Test Recall": "#f59e0b",
            "Test F1": "#10b981",
        }
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Heatmap if sweeps data exists
heatmap_list = sweeps_data.get("heatmap_4a", [])
if heatmap_list:
    st.subheader("🔥 Multimodal Calibration Surface (Alpha vs. Threshold)")
    hm_df = pd.DataFrame(heatmap_list)
    pivot_hm = hm_df.pivot(index="alpha", columns="threshold", values="f1")
    fig_hm = px.imshow(
        pivot_hm,
        labels=dict(x="Similarity Threshold (τ)", y="Text Weight (α)", color="Validation F1"),
        x=pivot_hm.columns,
        y=pivot_hm.index,
        color_continuous_scale="Viridis",
        title="Grid Calibration Surface for Weighted Multimodal Fusion (Exp 4a)",
    )
    st.plotly_chart(fig_hm, use_container_width=True)

# -----------------------------------------------------------------
# 3. Latency & Scalability Benchmarks
# -----------------------------------------------------------------
st.subheader("3️⃣ Retrieval Scalability: Brute-Force vs. FAISS IndexFlatIP")

if latency_data:
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        st.metric("Database Vectors", f"{latency_data.get('database_size', 5138):,}")
    with lc2:
        st.metric("Brute-Force Latency", f"{latency_data.get('brute_force_latency_ms', 0.0):.3f} ms")
    with lc3:
        st.metric("FAISS IndexFlatIP Latency", f"{latency_data.get('faiss_latency_ms', 0.0):.3f} ms")
    with lc4:
        st.metric("FAISS Throughput", f"{latency_data.get('faiss_qps', 0.0):,.0f} QPS")

    st.caption("FAISS enables exact inner-product search (Recall@50 = 100.0%) with sub-millisecond query response time.")
else:
    st.caption("Run `python scripts/run_experiments.py --benchmark-faiss` to measure local FAISS retrieval throughput.")

st.divider()

# -----------------------------------------------------------------
# 4. Qualitative Error Analysis
# -----------------------------------------------------------------
st.subheader("4️⃣ Qualitative Error Analysis")
st.caption("Comparative error patterns illustrating why multimodal fusion outperforms unimodal models.")

if error_cases:
    for idx, case in enumerate(error_cases[:4]):
        category_title = {
            "fusion_fixed_text_fp": "📱 Text False Positive Resolved by Vision (e.g. Phone vs. Accessory)",
            "fusion_fixed_image_fp": "🎨 Visual False Positive Resolved by Text (e.g. Different Model, Similar Color)",
            "fusion_recovered_fn": "🔍 Missed True Match Recovered by Multimodal Fusion",
            "fusion_still_wrong": "⚠️ Hard Residual Challenge Case",
        }.get(case.get("category"), "Sample Case")

        with st.expander(f"Case {idx + 1}: {category_title}", expanded=True):
            col_q, col_c, col_info = st.columns([1, 1, 2])
            with col_q:
                render_product_card(
                    title=case.get("title_query", ""),
                    image_name=case.get("image_query", ""),
                    label_group=0,
                    status="Query Product",
                )
            with col_c:
                render_product_card(
                    title=case.get("title_candidate", ""),
                    image_name=case.get("image_candidate", ""),
                    label_group=0,
                    status="Candidate Product",
                )
            with col_info:
                st.markdown(f"**Explanation:** {case.get('explanation', '')}")
                st.markdown(f"- Text Similarity: `{case.get('s_text', 0.0):.3f}`")
                st.markdown(f"- Visual Similarity: `{case.get('s_img', 0.0):.3f}`")
                st.markdown(f"- **Fused Score:** `{case.get('s_fused', 0.0):.3f}`")
else:
    q1, q2 = st.columns(2)
    with q1:
        with st.container(border=True):
            st.markdown("#### 📱 Phone vs. Phone Case (Text False Positives)")
            st.write(
                "- **Problem:** Sellers frequently title accessories as *'iPhone 13 Pro Max Silicon Case Cover'*, causing text-only models to falsely match the phone.\n"
                "- **Multimodal Solution:** Visual features clearly distinguish an accessory case from a smartphone chassis."
            )
        with q1:
            with st.container(border=True):
                st.markdown("#### 🎨 Color & Size Variants (Visual False Positives)")
                st.write(
                    "- **Problem:** Distinct colors of an identical apparel item can have visual feature distance.\n"
                    "- **Multimodal Solution:** Text title semantic embeddings anchor the identical brand and model specifications."
                )
    with q2:
        with st.container(border=True):
            st.markdown("#### 🌐 Multilingual Title Discrepancies")
            st.write(
                "- **Problem:** Shopee listings alternate between Indonesian colloquialisms (*baju*, *kaos*) and English.\n"
                "- **Multimodal Solution:** Dense multilingual transformer embeddings align semantic intent across languages."
            )
        with q2:
            with st.container(border=True):
                st.markdown("#### 🔄 Cross-Generation Subtle Discrepancies")
                st.write(
                    "- **Problem:** AirPods Gen 1 vs Gen 2 have identical packaging and physical aesthetics.\n"
                    "- **Multimodal Solution:** Precise text numeric parsing distinguishes generations."
                )

# -----------------------------------------------------------------
# 5. Zero-Leakage Dataset Partitioning Verification
# -----------------------------------------------------------------
st.divider()
st.subheader("5️⃣ Dataset Integrity & Zero Data Leakage Verification")

try:
    train_df = pd.read_csv(TRAIN_SPLIT_CSV)
    val_df = pd.read_csv(VAL_SPLIT_CSV)
    test_df = pd.read_csv(TEST_DEMO_SPLIT_CSV)

    train_groups = set(train_df["label_group"].unique())
    val_groups = set(val_df["label_group"].unique())
    test_groups = set(test_df["label_group"].unique())

    overlap_train_val = len(train_groups.intersection(val_groups))
    overlap_train_test = len(train_groups.intersection(test_groups))
    overlap_val_test = len(val_groups.intersection(test_groups))

    d1, d2, d3 = st.columns(3)
    with d1:
        st.metric("Train Split", f"{len(train_df):,} rows", f"{len(train_groups):,} groups")
    with d2:
        st.metric("Validation Split", f"{len(val_df):,} rows", f"{len(val_groups):,} groups")
    with d3:
        st.metric("Test / Demo Split", f"{len(test_df):,} rows", f"{len(test_groups):,} groups")

    if overlap_train_test == 0 and overlap_train_val == 0 and overlap_val_test == 0:
        st.success(
            "✅ **Strict Zero Leakage Verified:** Label group intersection across all three splits is strictly **0**. "
            "All test evaluations and interactive demo matches are conducted on completely unseen product identities."
        )
    else:
        st.error("⚠️ Warning: Non-zero label group overlap detected between splits.")
except Exception as e:
    st.caption("Split CSVs verification summary will display when processed split files are present.")
