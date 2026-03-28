import subprocess
import sys
import json
import time
import os
from datetime import datetime

def header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def step(num, text):
    print(f"\n[Step {num}] {text}...")

def success(text):
    print(f"  ✅ {text}")

def info(text):
    print(f"  ℹ  {text}")

def run_script(script_name):
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,
        text=True
    )
    return result.returncode == 0

def main():
    start_time = time.time()

    header("MICROSERVICE FAILURE IMPACT PREDICTION PIPELINE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Running full pipeline: Traces → Graph → GNN → Neo4j → Dashboard")

    # ── Step 1: Generate Traffic + Collect Traces ─────────────────────────────
    step(1, "Generating traffic and collecting traces from Jaeger")

    # Auto-generate traffic
    print("  Generating traffic on HotROD...")
    import requests as req
    urls = [
    "http://localhost:8080/dispatch?customer=123&nonse=0.1",
    "http://localhost:8080/dispatch?customer=392&nonse=0.2",
    "http://localhost:8080/dispatch?customer=731&nonse=0.3",
    "http://localhost:8080/dispatch?customer=567&nonse=0.4",
    ]
    for i in range(20):
        for url in urls:
            try:
                req.get(url, timeout=2)
            except:
                pass
        time.sleep(0.5)
    print("  Traffic generated — waiting 5 seconds for traces to appear...")
    time.sleep(5)

    ok = run_script("collect_traces.py")
    if ok:
        with open("raw_spans.json") as f:
            spans = json.load(f)
        if len(spans) == 0:
            print("  ❌ No spans collected — generate traffic manually at http://localhost:8080")
            sys.exit(1)
        success(f"Collected {len(spans)} spans from Jaeger")
    else:
        print("  ❌ Trace collection failed — is Jaeger running?")
        print("     Start it with: docker compose up -d")
        sys.exit(1)
    # ── Step 2: Build Dependency Graph ────────────────────────
    step(2, "Building weighted dependency graph")
    ok = run_script("build_graph.py")
    if ok:
        with open("dependency_graph.json") as f:
            graph = json.load(f)
        success(f"Graph built — {graph['num_nodes']} services, {graph['num_edges']} edges")
    else:
        print("  ❌ Graph build failed")
        sys.exit(1)

    # ── Step 3: Simulate Failures ─────────────────────────────────────────────
    step(3, "Generating synthetic fault scenarios")
    ok = run_script("simulate_failures.py")
    if ok:
        with open("fault_scenarios.json") as f:
            scenarios = json.load(f)
        success(f"Generated {len(scenarios)} fault scenarios")
    else:
        print("  ❌ Fault simulation failed")
        sys.exit(1)

    # ── Step 4: Train GNN ─────────────────────────────────────────────────────
    step(4, "Training GNN model")
    print("  This takes 2-3 minutes...")
    ok = run_script("gnn_model.py")
    if ok:
        success("GNN trained — blast_radius_gnn.pt saved")
    else:
        print("  ❌ GNN training failed")
        sys.exit(1)

    # ── Step 5: Push to Neo4j ─────────────────────────────────────────────────
    step(5, "Pushing graph to Neo4j")
    ok = run_script("neo4j_graph.py")
    if ok:
        success("Graph stored in Neo4j — view at http://localhost:7474")
    else:
        print("  ⚠  Neo4j push failed — continuing without it")
        print("     Is Neo4j Desktop running?")

    # ── Step 6: Generate All Incident Reports ─────────────────────────────────

    step(6, "Generating LLM incident reports for all services")

    # Import and run directly instead of subprocess
    from llm_incident_report import run_pipeline as run_report

    with open("dependency_graph.json") as f:
        graph_data = json.load(f)
    services = graph_data["service_names"]

    reports_generated = 0
    for svc in services:
        try:
            result = run_report(svc)
            if result:
                reports_generated += 1
                print(f"  ✅ Report generated for {svc}")
            else:
                print(f"  ⚠  Report failed for {svc}")
        except Exception as e:
            print(f"  ⚠  Error for {svc}: {str(e)[:50]}")

    success(f"Generated {reports_generated}/{len(services)} incident reports")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = round(time.time() - start_time, 1)

    header("PIPELINE COMPLETE")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Total time: {elapsed} seconds")
    print()
    print("  Outputs:")
    print("    raw_spans.json           — collected trace data")
    print("    dependency_graph.json    — weighted service graph")
    print("    fault_scenarios.json     — 6 fault scenarios")
    print("    blast_radius_gnn.pt      — trained GNN model")
    print("    incident_report_*.json   — LLM reports per service")
    print()
    print("  Next steps:")
    print("    View Neo4j graph:   http://localhost:7474")
    print("    Launch dashboard:   streamlit run dashboard.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
