import streamlit as st
import json
import torch
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from groq import Groq
from dotenv import load_dotenv
import os
from gnn_model import BlastRadiusGNN, build_pyg_data

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Microservice Failure Impact Prediction",
    page_icon="🔥",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; }
    .metric-card {
        background: #1E2130;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #00C6FF;
        margin: 8px 0;
    }
    .report-box {
        background: #1E2130;
        border-radius: 10px;
        padding: 20px;
        border-left: 4px solid #7B2FBE;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00C6FF, #7B2FBE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ── Load resources ────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_graph():
    with open("dependency_graph.json") as f:
        graph = json.load(f)
    model = BlastRadiusGNN()
    model.load_state_dict(torch.load(
        "blast_radius_gnn.pt", weights_only=True))
    model.eval()
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return graph, model, client

# ── GNN Prediction ────────────────────────────────────────────────────────────
def predict(graph, model, fault_service):
    service_names = graph["service_names"]
    fault_idx     = service_names.index(fault_service)
    data          = build_pyg_data(graph, fault_idx)
    with torch.no_grad():
        preds = model(data).numpy().tolist()
    return {svc: round(preds[i], 4)
            for i, svc in enumerate(service_names)}

# ── LLM Report ────────────────────────────────────────────────────────────────
def generate_report(client, fault_service, predictions, graph):
    service_names = graph["service_names"]
    sorted_preds  = sorted(predictions.items(),
                           key=lambda x: x[1], reverse=True)
    affected = [(s, p) for s, p in sorted_preds
                if p > 0.1 and s != fault_service]
    edges    = [e for e in graph["edge_list"]
                if e["source_name"] == fault_service
                or e["target_name"] == fault_service]

    affected_str = "\n".join(
        [f"  - {s}: {p*100:.1f}% failure probability"
         for s, p in affected]
    ) or "  - No downstream services significantly affected"

    edges_str = "\n".join(
        [f"  - {e['source_name']} → {e['target_name']} "
         f"(weight={e['weight']}, latency={e['avg_duration_ms']}ms)"
         for e in edges]
    ) or "  - No direct edges"

    prompt = f"""You are a senior Site Reliability Engineer.

A failure has been detected in: {fault_service}

GNN Blast Radius Predictions:
AFFECTED SERVICES (>10% failure probability):
{affected_str}

Relevant dependency edges:
{edges_str}

Generate a concise incident report with these sections:
1. INCIDENT SUMMARY (2 sentences)
2. PREDICTED IMPACT
3. IMMEDIATE ACTIONS (3 specific actions)
4. TEAMS TO NOTIFY
5. RISK ASSESSMENT (1 sentence)

Keep under 250 words. Be specific and actionable."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# ── Blast Radius Chart ────────────────────────────────────────────────────────
def blast_radius_chart(predictions, fault_service):
    sorted_preds = sorted(predictions.items(),
                          key=lambda x: x[1], reverse=True)
    services = [s for s, _ in sorted_preds]
    scores   = [p for _, p in sorted_preds]
    colors   = []
    for s, p in sorted_preds:
        if s == fault_service:
            colors.append("#FF4D6D")
        elif p > 0.5:
            colors.append("#FF8C42")
        elif p > 0.1:
            colors.append("#F5A623")
        else:
            colors.append("#00C6FF")

    fig = go.Figure(go.Bar(
        x=scores,
        y=services,
        orientation="h",
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=dict(
            text=f"Blast Radius — {fault_service} failure",
            font=dict(size=18, color="white")
        ),
        plot_bgcolor="#1E2130",
        paper_bgcolor="#1E2130",
        font=dict(color="white"),
        xaxis=dict(
            title="Failure Probability",
            range=[0, 1.2],
            gridcolor="#2E3250",
            tickformat=".0%"
        ),
        yaxis=dict(gridcolor="#2E3250"),
        height=350,
        margin=dict(l=20, r=80, t=50, b=20)
    )
    return fig

# ── Dependency Graph ──────────────────────────────────────────────────────────
def draw_dependency_graph(graph, predictions, fault_service):
    G   = nx.DiGraph()
    for node in graph["node_features"]:
        G.add_node(node["service_name"])
    for edge in graph["edge_list"]:
        G.add_edge(edge["source_name"], edge["target_name"],
                   weight=edge["weight"])

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#1E2130")
    ax.set_facecolor("#1E2130")

    pos = nx.spring_layout(G, seed=42, k=2)

    node_colors = []
    node_sizes  = []
    for node in G.nodes():
        score = predictions.get(node, 0)
        if node == fault_service:
            node_colors.append("#FF4D6D")
            node_sizes.append(3000)
        elif score > 0.5:
            node_colors.append("#FF8C42")
            node_sizes.append(2200)
        elif score > 0.1:
            node_colors.append("#F5A623")
            node_sizes.append(1800)
        else:
            node_colors.append("#00C6FF")
            node_sizes.append(1500)

    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w        = max(edge_weights) if edge_weights else 1
    edge_widths  = [1 + (w / max_w) * 4 for w in edge_weights]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=node_sizes, ax=ax, alpha=0.95)
    nx.draw_networkx_labels(G, pos, font_color="white",
                            font_size=10, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           edge_color="#AAAAAA", arrows=True,
                           arrowsize=20, ax=ax,
                           connectionstyle="arc3,rad=0.1")

    legend = [
        mpatches.Patch(color="#FF4D6D", label="Fault Source"),
        mpatches.Patch(color="#FF8C42", label="High Risk (>50%)"),
        mpatches.Patch(color="#F5A623", label="Medium Risk (>10%)"),
        mpatches.Patch(color="#00C6FF", label="Safe"),
    ]
    ax.legend(handles=legend, loc="lower right",
              facecolor="#1E2130", labelcolor="white",
              fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    return fig

# ── Main Dashboard ────────────────────────────────────────────────────────────
def main():
    graph, model, client = load_model_and_graph()
    service_names        = graph["service_names"]

    # Header
    st.markdown('<p class="title-text">🔥 Microservice Failure Impact Prediction</p>',
                unsafe_allow_html=True)
    st.markdown("**Automated blast radius prediction using Graph Attention Networks + LLM incident reports**")
    st.divider()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        st.markdown("---")
        fault_service = st.selectbox(
            "Select Failing Service",
            service_names,
            index=service_names.index("mysql")
        )
        run_button = st.button(
            "▶ Run Prediction",
            type="primary",
            use_container_width=True
        )
        st.markdown("---")
        st.markdown("**Pipeline Steps:**")
        st.markdown("1. Load GNN model")
        st.markdown("2. Predict blast radius")
        st.markdown("3. Generate LLM report")
        st.markdown("---")
        st.markdown("**Model Info:**")
        st.markdown("Architecture: 3-layer GAT")
        st.markdown("HR@1: 6/6 (100%)")
        st.markdown("MAE: 0.0686")
        st.markdown("Training epochs: 500")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Services", len(service_names))
    with col2:
        st.metric("Dependency Edges", graph["num_edges"])
    with col3:
        st.metric("GNN Accuracy HR@1", "6/6")
    with col4:
        st.metric("Model MAE", "0.0686")

    st.divider()

    if run_button:
        with st.spinner(f"Running GNN prediction for {fault_service} failure..."):
            predictions = predict(graph, model, fault_service)

        # Main content area
        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.subheader("📊 Blast Radius Prediction")
            fig_bar = blast_radius_chart(predictions, fault_service)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_right:
            st.subheader("🕸️ Dependency Graph")
            fig_graph = draw_dependency_graph(
                graph, predictions, fault_service)
            st.pyplot(fig_graph, use_container_width=True)

        # Prediction table
        st.subheader("📋 Detailed Predictions")
        pred_cols = st.columns(len(service_names))
        sorted_preds = sorted(predictions.items(),
                              key=lambda x: x[1], reverse=True)
        for i, (svc, score) in enumerate(sorted_preds):
            with pred_cols[i]:
                delta = "← FAULT" if svc == fault_service else None
                st.metric(
                    label=svc,
                    value=f"{score*100:.1f}%",
                    delta=delta,
                    delta_color="off" if delta else "normal"
                )

        st.divider()

        # LLM Report
        st.subheader("🤖 AI-Generated Incident Report")
        with st.spinner("Generating LLM incident report via Groq..."):
            report = generate_report(
                client, fault_service, predictions, graph)

        st.markdown(f"""
<div class="report-box">
<strong>INCIDENT REPORT — {fault_service.upper()} FAILURE</strong>
<hr style="border-color:#444">
{report}
</div>
""", unsafe_allow_html=True)

        # Save report
        output = {
            "fault_service": fault_service,
            "predictions":   predictions,
            "report":        report
        }
        with open(f"incident_report_{fault_service}.json", "w") as f:
            json.dump(output, f, indent=2)

        st.success(f"Report saved to incident_report_{fault_service}.json")

    else:
        # Welcome screen
        st.info("👈 Select a failing service from the sidebar and click **Run Prediction** to start.")
        st.markdown("### How It Works")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("**Layer 1**\n\n🔍 Collect distributed traces from Jaeger")
        with col2:
            st.markdown("**Layer 2**\n\n🕸️ Build weighted dependency graph from spans")
        with col3:
            st.markdown("**Layer 3**\n\n🧠 GAT predicts blast radius per service")
        with col4:
            st.markdown("**Layer 4**\n\n📝 LLM generates actionable incident report")

if __name__ == "__main__":
    main()
