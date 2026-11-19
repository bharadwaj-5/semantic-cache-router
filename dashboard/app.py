"""Streamlit dashboard — cost, cache, latency and quality analytics."""

from __future__ import annotations

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Semantic Cache Router — Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Semantic Cache Router — Dashboard")
st.caption("Cost-aware LLM routing with semantic caching")


# ── Helper ───────────────────────────────────────────────────
@st.cache_data(ttl=5)
def fetch(endpoint: str):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ── Sidebar: test query sender ───────────────────────────────
with st.sidebar:
    st.header("🔍 Send a test query")
    test_q = st.text_area("Query", value="What is the company refund policy?")
    if st.button("Send", use_container_width=True):
        with st.spinner("Querying…"):
            try:
                resp = requests.post(
                    f"{API_BASE}/query",
                    json={"query": test_q},
                    timeout=30,
                )
                data = resp.json()
                st.success("Done!")
                st.json(data)
            except Exception as e:
                st.error(str(e))

    st.divider()
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()


# ── Fetch data ───────────────────────────────────────────────
stats = fetch("/stats")
logs_raw = fetch("/logs")
health = fetch("/health")

if not stats or not logs_raw:
    st.info("No data yet. Send some queries using the sidebar or the API.")
    st.stop()

df = pd.DataFrame(logs_raw)
if df.empty:
    st.info("No request logs yet.")
    st.stop()

# Type conversions
df["cache_hit"] = df["cache_hit"].astype(bool)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# ── KPIs row ─────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Requests", stats["total_requests"])
k2.metric("Cache Hit Rate", f"{stats['cache_hit_rate']:.1%}")
k3.metric("Avg Cost / req", f"${stats['avg_cost']:.6f}")
k4.metric("Total Cost", f"${stats['total_cost']:.4f}")
k5.metric("p95 Latency", f"{stats['p95_latency_ms']:.0f} ms")

st.divider()

# ── Charts ───────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    # Cost per request over time
    st.subheader("💰 Cost per request")
    fig_cost = px.bar(
        df[~df["cache_hit"]],
        x=df[~df["cache_hit"]].index,
        y="cost",
        color="route",
        color_discrete_map={"simple": "#36b37e", "complex": "#ff5630", "escalated": "#ffab00"},
        labels={"x": "Request #", "cost": "Cost (USD)"},
    )
    fig_cost.update_layout(height=350, margin=dict(t=10, b=30))
    st.plotly_chart(fig_cost, use_container_width=True)

with col_right:
    # Cache hit / miss pie
    st.subheader("🎯 Cache Hit Rate")
    hit_counts = df["cache_hit"].value_counts()
    fig_cache = px.pie(
        names=["Cache Miss", "Cache Hit"],
        values=[hit_counts.get(False, 0), hit_counts.get(True, 0)],
        color_discrete_sequence=["#ff5630", "#36b37e"],
        hole=0.45,
    )
    fig_cache.update_layout(height=350, margin=dict(t=10, b=30))
    st.plotly_chart(fig_cache, use_container_width=True)

col_left2, col_right2 = st.columns(2)

with col_left2:
    # Route split
    st.subheader("🔀 Route Split")
    route_data = stats.get("route_split", {})
    if route_data:
        fig_route = px.bar(
            x=list(route_data.keys()),
            y=list(route_data.values()),
            color=list(route_data.keys()),
            color_discrete_map={
                "simple": "#36b37e",
                "complex": "#ff5630",
                "escalated": "#ffab00",
                "cache_hit": "#0065ff",
            },
            labels={"x": "Route", "y": "Count"},
        )
        fig_route.update_layout(height=350, showlegend=False, margin=dict(t=10, b=30))
        st.plotly_chart(fig_route, use_container_width=True)

with col_right2:
    # Latency distribution
    st.subheader("⏱️ Latency Distribution (ms)")
    fig_lat = px.histogram(
        df,
        x="latency_ms",
        color="cache_hit",
        nbins=30,
        color_discrete_map={True: "#36b37e", False: "#ff5630"},
        labels={"latency_ms": "Latency (ms)", "cache_hit": "Cache Hit"},
    )
    fig_lat.update_layout(height=350, margin=dict(t=10, b=30))
    st.plotly_chart(fig_lat, use_container_width=True)

# ── Quality score per route ──────────────────────────────────
st.subheader("📊 Quality Score per Route")
non_cache = df[~df["cache_hit"]]
if not non_cache.empty:
    fig_quality = px.box(
        non_cache,
        x="route",
        y="quality_score",
        color="route",
        color_discrete_map={"simple": "#36b37e", "complex": "#ff5630", "escalated": "#ffab00"},
    )
    fig_quality.update_layout(height=300, showlegend=False, margin=dict(t=10, b=30))
    st.plotly_chart(fig_quality, use_container_width=True)

# ── Cost comparison: baseline vs routed ──────────────────────
st.subheader("🏷️ Cost Comparison: Always-Large vs Smart Routing")
non_cache_df = df[~df["cache_hit"]]
if not non_cache_df.empty:
    actual_total = non_cache_df["cost"].sum()
    # Estimate baseline cost: re-price all tokens at the large-model rate
    large_in_rate = 2.50 / 1_000_000   # gpt-4o input
    large_out_rate = 10.00 / 1_000_000  # gpt-4o output
    baseline_cost = (
        (non_cache_df["tokens_in"] * large_in_rate)
        + (non_cache_df["tokens_out"] * large_out_rate)
    ).sum()

    total_with_cache = df["cost"].sum()

    comp_df = pd.DataFrame({
        "Strategy": ["Always Large Model", "Smart Routing (no cache)", "Smart Routing + Cache"],
        "Total Cost (USD)": [baseline_cost, actual_total, total_with_cache],
    })
    fig_comp = px.bar(
        comp_df,
        x="Strategy",
        y="Total Cost (USD)",
        color="Strategy",
        color_discrete_sequence=["#ff5630", "#ffab00", "#36b37e"],
        text_auto=".6f",
    )
    fig_comp.update_layout(height=350, showlegend=False, margin=dict(t=10, b=30))
    st.plotly_chart(fig_comp, use_container_width=True)

    savings = ((baseline_cost - total_with_cache) / baseline_cost * 100) if baseline_cost > 0 else 0
    st.success(f"💡 **Estimated savings: {savings:.1f}%** vs always using the large model")

# ── Raw logs table ───────────────────────────────────────────
with st.expander("📋 Raw Request Logs"):
    st.dataframe(df, use_container_width=True)

# ── Health check ─────────────────────────────────────────────
with st.expander("🏥 System Health"):
    if health:
        st.json(health)
