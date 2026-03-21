import json
import random
import copy
from collections import defaultdict

# ── Load real spans and graph ─────────────────────────────────────────────────
def load_data():
    with open("raw_spans.json") as f:
        spans = json.load(f)
    with open("dependency_graph.json") as f:
        graph = json.load(f)
    print(f"Loaded {len(spans)} spans and graph with {graph['num_nodes']} nodes")
    return spans, graph

# ── Build dependency map from graph ──────────────────────────────────────────
def build_dependency_map(graph):
    depends_on = defaultdict(set)
    affected_by = defaultdict(set)
    for edge in graph["edge_list"]:
        caller = edge["source_name"]
        callee = edge["target_name"]
        depends_on[caller].add(callee)
        affected_by[callee].add(caller)
    return depends_on, affected_by

# ── Find blast radius from graph structure ────────────────────────────────────
def get_blast_radius(fault_service, affected_by, graph):
    blast = {fault_service: 1.0}
    queue = [fault_service]
    visited = {fault_service}
    while queue:
        current = queue.pop(0)
        for caller in affected_by.get(current, []):
            if caller not in visited:
                visited.add(caller)
                queue.append(caller)
                # impact decreases with distance from fault
                distance = len(visited)
                impact = round(max(0.3, 1.0 - (distance * 0.2)), 4)
                blast[caller] = impact
    # services not in blast radius have zero impact
    all_services = graph["service_names"]
    for svc in all_services:
        if svc not in blast:
            blast[svc] = 0.0
    return blast

# ── Simulate one fault scenario ───────────────────────────────────────────────
def simulate_scenario(spans, fault_service, blast_radius, scenario_id):
    simulated_spans = []
    for span in spans:
        new_span = copy.deepcopy(span)
        svc = span["serviceName"]
        impact = blast_radius.get(svc, 0.0)
        if impact > 0:
            # inject errors proportional to impact score
            if random.random() < impact:
                new_span["error"] = True
                new_span["error_type"] = f"ServiceUnavailable: {fault_service} failed"
                # increase latency to simulate cascading slowdown
                new_span["duration_us"] = int(span["duration_us"] * (1 + impact * 4))
            else:
                new_span["error"] = False
        else:
            new_span["error"] = False
        simulated_spans.append(new_span)
    return {
        "scenario_id":   scenario_id,
        "fault_service": fault_service,
        "blast_radius":  blast_radius,
        "spans":         simulated_spans,
        "total_spans":   len(simulated_spans),
        "error_spans":   sum(1 for s in simulated_spans if s.get("error"))
    }

# ── Build GNN training labels ─────────────────────────────────────────────────
def build_gnn_labels(scenarios, graph):
    training_data = []
    service_names = graph["service_names"]
    for scenario in scenarios:
        blast = scenario["blast_radius"]
        label_vector = [blast.get(svc, 0.0) for svc in service_names]
        fault_index  = service_names.index(scenario["fault_service"])
        training_data.append({
            "scenario_id":    scenario["scenario_id"],
            "fault_service":  scenario["fault_service"],
            "fault_node_idx": fault_index,
            "label_vector":   label_vector,
            "service_names":  service_names
        })
    return training_data

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    print("=" * 55)
    print("SYNTHETIC FAULT INJECTION PIPELINE")
    print("=" * 55)

    spans, graph = load_data()
    depends_on, affected_by = build_dependency_map(graph)

    services = graph["service_names"]
    print(f"\nGenerating fault scenarios for: {services}")

    scenarios = []
    for i, fault_service in enumerate(services, 1):
        print(f"\n[Scenario {i}] Fault source: {fault_service}")
        blast = get_blast_radius(fault_service, affected_by, graph)
        print(f"  Blast radius:")
        for svc, impact in sorted(blast.items(), key=lambda x: -x[1]):
            bar = "█" * int(impact * 10)
            print(f"    {svc:<16} {impact:.2f}  {bar}")
        scenario = simulate_scenario(spans, fault_service, blast, i)
        print(f"  Total spans: {scenario['total_spans']}  |  Error spans: {scenario['error_spans']}")
        scenarios.append(scenario)

    # Save full scenarios
    scenarios_export = [{
        "scenario_id":   s["scenario_id"],
        "fault_service": s["fault_service"],
        "blast_radius":  s["blast_radius"],
        "total_spans":   s["total_spans"],
        "error_spans":   s["error_spans"]
    } for s in scenarios]

    with open("fault_scenarios.json", "w") as f:
        json.dump(scenarios_export, f, indent=2)

    # Build and save GNN training labels
    training_data = build_gnn_labels(scenarios, graph)
    with open("gnn_training_labels.json", "w") as f:
        json.dump(training_data, f, indent=2)

    print("\n" + "=" * 55)
    print("DAY 3 COMPLETE")
    print("=" * 55)
    print(f"  fault_scenarios.json     — {len(scenarios)} scenarios")
    print(f"  gnn_training_labels.json — {len(training_data)} labelled training samples")
    print("\nBlast radius summary:")
    print(f"  {'Scenario':<12} {'Fault Service':<16} {'Affected Services'}")
    print(f"  {'-'*55}")
    for s in scenarios_export:
        affected = [sv for sv, imp in s["blast_radius"].items()
                   if imp > 0.0 and sv != s["fault_service"]]
        print(f"  S{s['scenario_id']:<11} {s['fault_service']:<16} {', '.join(affected)}")

if __name__ == "__main__":
    main()