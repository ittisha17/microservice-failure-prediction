import json
import torch
import numpy as np
from groq import Groq
from dotenv import load_dotenv
import os
from torch_geometric.data import Data
from gnn_model import BlastRadiusGNN, build_pyg_data

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ── Load graph and model ──────────────────────────────────────────────────────
def load_everything():
    with open("dependency_graph.json") as f:
        graph = json.load(f)
    model = BlastRadiusGNN()
    model.load_state_dict(torch.load("blast_radius_gnn.pt",
                                     weights_only=True))
    model.eval()
    return graph, model

# ── Run GNN prediction ────────────────────────────────────────────────────────
def predict_blast_radius(graph, model, fault_service):
    service_names = graph["service_names"]
    if fault_service not in service_names:
        print(f"Service '{fault_service}' not found.")
        print(f"Available: {service_names}")
        return None
    fault_idx = service_names.index(fault_service)
    data      = build_pyg_data(graph, fault_idx)
    with torch.no_grad():
        predictions = model(data).numpy().tolist()
    results = {}
    for i, svc in enumerate(service_names):
        results[svc] = round(predictions[i], 4)
    return results

# ── Format predictions for LLM ───────────────────────────────────────────────
def format_predictions(fault_service, predictions, graph):
    sorted_svcs = sorted(predictions.items(),
                         key=lambda x: x[1], reverse=True)
    affected = [(s, p) for s, p in sorted_svcs
                if p > 0.1 and s != fault_service]
    safe     = [(s, p) for s, p in sorted_svcs
                if p <= 0.1 and s != fault_service]

    edges = graph["edge_list"]
    relevant_edges = []
    for e in edges:
        if (e["source_name"] == fault_service or
            e["target_name"] == fault_service):
            relevant_edges.append(
                f"{e['source_name']} → {e['target_name']} "
                f"(weight={e['weight']}, "
                f"avg_latency={e['avg_duration_ms']}ms)"
            )

    return {
        "fault_service":   fault_service,
        "affected":        affected,
        "safe":            safe,
        "relevant_edges":  relevant_edges,
        "all_predictions": sorted_svcs
    }

# ── Build LLM prompt ──────────────────────────────────────────────────────────
def build_prompt(formatted):
    fault    = formatted["fault_service"]
    affected = formatted["affected"]
    safe     = formatted["safe"]
    edges    = formatted["relevant_edges"]

    affected_str = "\n".join(
        [f"  - {s}: {p*100:.1f}% failure probability"
         for s, p in affected]
    ) or "  - No downstream services significantly affected"

    safe_str = "\n".join([f"  - {s}" for s, p in safe])

    edges_str = "\n".join([f"  - {e}" for e in edges]) or "  - No direct edges found"

    return f"""You are a senior Site Reliability Engineer at a cloud-native company.

A failure has been detected in the microservice: **{fault}**

GNN Blast Radius Predictions:
AFFECTED SERVICES (>10% failure probability):
{affected_str}

SAFE SERVICES (low failure probability):
{safe_str}

Relevant dependency edges for {fault}:
{edges_str}

Generate a concise professional incident report with these exact sections:

1. INCIDENT SUMMARY (2 sentences max)
2. PREDICTED IMPACT (which services, severity, why they are affected)
3. IMMEDIATE ACTIONS (3 specific technical actions the on-call team should take right now)
4. TEAMS TO NOTIFY (which teams based on affected services)
5. RISK ASSESSMENT (one sentence on overall system risk level)

Keep the entire report under 250 words. Be specific and actionable."""

# ── Call Groq API ─────────────────────────────────────────────────────────────
def generate_report(prompt):
    response = client.chat.completions.create(
       model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# ── Print full prediction table ───────────────────────────────────────────────
def print_prediction_table(fault_service, predictions):
    print(f"\n{'='*55}")
    print(f"GNN BLAST RADIUS PREDICTION")
    print(f"Fault Service: {fault_service}")
    print(f"{'='*55}")
    print(f"  {'Service':<16} {'Risk Score':>10} {'Bar'}")
    print(f"  {'-'*45}")
    sorted_preds = sorted(predictions.items(),
                          key=lambda x: x[1], reverse=True)
    for svc, score in sorted_preds:
        bar    = "█" * int(score * 20)
        marker = " ← FAULT SOURCE" if svc == fault_service else ""
        print(f"  {svc:<16} {score:>10.4f}  {bar}{marker}")

# ── Main ──────────────────────────────────────────────────────────────────────
def run_pipeline(fault_service):
    print(f"\n{'='*55}")
    print(f"LLM INCIDENT REPORT PIPELINE")
    print(f"{'='*55}")

    graph, model = load_everything()

    # Step 1 - GNN prediction
    print(f"\n[1] Running GNN prediction for fault: {fault_service}")
    predictions = predict_blast_radius(graph, model, fault_service)
    if not predictions:
        return
    print_prediction_table(fault_service, predictions)

    # Step 2 - Format for LLM
    print(f"\n[2] Formatting predictions for LLM...")
    formatted = format_predictions(fault_service, predictions, graph)

    # Step 3 - Generate report
    print(f"\n[3] Generating LLM incident report via Groq...")
    prompt = build_prompt(formatted)
    report = generate_report(prompt)

    # Step 4 - Display report
    print(f"\n{'='*55}")
    print(f"INCIDENT REPORT — {fault_service.upper()} FAILURE")
    print(f"{'='*55}")
    print(report)
    print(f"{'='*55}")

    # Step 5 - Save report
    output = {
        "fault_service": fault_service,
        "predictions":   predictions,
        "report":        report
    }
    filename = f"incident_report_{fault_service}.json"
    with open(filename, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nReport saved to {filename}")

    return output

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fault = sys.argv[1]
    else:
        fault = "mysql"
    run_pipeline(fault)
