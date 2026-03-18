import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from collections import defaultdict

def load_spans(filepath="raw_spans.json"):
    with open(filepath, "r") as f:
        spans = json.load(f)
    print(f"Loaded {len(spans)} spans from {filepath}")
    return spans

def build_span_lookup(spans):
    lookup = {}
    for span in spans:
        lookup[span["spanID"]] = span
    return lookup

def extract_edges(spans, span_lookup):
    edge_calls = defaultdict(int)
    edge_durations = defaultdict(list)

    for span in spans:
        parent_id = span.get("parentSpanID")
        if not parent_id:
            continue
        parent_span = span_lookup.get(parent_id)
        if not parent_span:
            continue
        caller = parent_span["serviceName"]
        callee = span["serviceName"]

        # skip self calls
        if caller == callee:
            continue
        # skip unknown
        if caller == "unknown" or callee == "unknown":
            continue

        edge = (caller, callee)
        edge_calls[edge] += 1
        edge_durations[edge].append(span["duration_us"])

    return edge_calls, edge_durations

def build_networkx_graph(edge_calls, edge_durations):
    G = nx.DiGraph()

    total_calls = sum(edge_calls.values())

    for (caller, callee), count in edge_calls.items():
        avg_duration = sum(edge_durations[(caller, callee)]) / len(edge_durations[(caller, callee)])
        # normalized weight between 0 and 1
        weight = count / total_calls
        G.add_edge(
            caller,
            callee,
            weight=round(weight, 4),
            call_count=count,
            avg_duration_ms=round(avg_duration / 1000, 2)
        )

    return G

def print_graph_stats(G):
    print("\n" + "=" * 50)
    print("DEPENDENCY GRAPH STATISTICS")
    print("=" * 50)
    print(f"Total services (nodes):     {G.number_of_nodes()}")
    print(f"Total dependencies (edges): {G.number_of_edges()}")

    print("\nDependency edges (caller → callee):")
    print(f"{'Caller':<20} {'Callee':<20} {'Calls':>8} {'Weight':>8} {'Avg Latency':>12}")
    print("-" * 72)

    edges = sorted(G.edges(data=True), key=lambda x: x[2]["call_count"], reverse=True)
    for caller, callee, data in edges:
        print(f"{caller:<20} {callee:<20} {data['call_count']:>8} {data['weight']:>8.4f} {data['avg_duration_ms']:>10.2f}ms")

def compute_centrality(G):
    print("\n" + "=" * 50)
    print("CRITICAL SERVICE RANKING")
    print("=" * 50)

    in_degree   = nx.in_degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    pagerank    = nx.pagerank(G)

    services = list(G.nodes())
    scores = {}
    for s in services:
        score = (
            in_degree.get(s, 0) * 0.4 +
            betweenness.get(s, 0) * 0.3 +
            pagerank.get(s, 0) * 0.3
        )
        scores[s] = round(score, 4)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Service':<20} {'Score':>8} {'In-Degree':>10} {'Betweenness':>12} {'PageRank':>10}")
    print("-" * 70)
    for i, (service, score) in enumerate(ranked, 1):
        print(f"{i:<6} {service:<20} {score:>8.4f} {in_degree.get(service,0):>10.4f} {betweenness.get(service,0):>12.4f} {pagerank.get(service,0):>10.4f}")

    return scores, ranked

def visualize_graph(G, centrality_scores):
    plt.figure(figsize=(14, 10))
    plt.title("Microservice Dependency Graph\nNode size = criticality | Edge thickness = call frequency",
              fontsize=14, fontweight="bold", pad=20)

    pos = nx.spring_layout(G, seed=42, k=2)

    # node sizes based on criticality
    max_score = max(centrality_scores.values()) if centrality_scores else 1
    node_sizes = [
        3000 + (centrality_scores.get(node, 0) / max_score) * 4000
        for node in G.nodes()
    ]

    # node colors
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]
    node_colors = colors[:len(G.nodes())]

    # edge widths based on weight
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [1 + (w / max_weight) * 5 for w in edge_weights]

    # draw everything
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                           node_color=node_colors, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=11,
                            font_weight="bold", font_color="white")
    nx.draw_networkx_edges(G, pos, width=edge_widths,
                           edge_color="#2C3E50", arrows=True,
                           arrowsize=25, arrowstyle="-|>",
                           connectionstyle="arc3,rad=0.1")

    # edge labels showing call count
    edge_labels = {(u, v): f"{d['call_count']} calls"
                   for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels,
                                 font_size=8, font_color="#E74C3C")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig("dependency_graph.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\ndependency_graph.png saved")

def export_for_gnn(G, centrality_scores):
    # node features
    node_features = []
    node_list = list(G.nodes())
    for i, node in enumerate(node_list):
        node_features.append({
            "node_id": i,
            "service_name": node,
            "in_degree": G.in_degree(node),
            "out_degree": G.out_degree(node),
            "criticality_score": centrality_scores.get(node, 0),
            "total_incoming_calls": sum(
                G[u][node]["call_count"]
                for u in G.predecessors(node)
            )
        })

    # edge list
    edge_list = []
    for u, v, data in G.edges(data=True):
        edge_list.append({
            "source": node_list.index(u),
            "target": node_list.index(v),
            "source_name": u,
            "target_name": v,
            "weight": data["weight"],
            "call_count": data["call_count"],
            "avg_duration_ms": data["avg_duration_ms"]
        })

    gnn_data = {
        "num_nodes": len(node_list),
        "num_edges": len(edge_list),
        "node_features": node_features,
        "edge_list": edge_list,
        "service_names": node_list
    }

    with open("dependency_graph.json", "w") as f:
        json.dump(gnn_data, f, indent=2)

    print("dependency_graph.json saved — ready for GNN input")
    return gnn_data

def main():
    print("=" * 50)
    print("DEPENDENCY GRAPH BUILDER")
    print("=" * 50)

    spans        = load_spans()
    span_lookup  = build_span_lookup(spans)
    edge_calls, edge_durations = extract_edges(spans, span_lookup)

    print(f"\nUnique dependency edges found: {len(edge_calls)}")

    G = build_networkx_graph(edge_calls, edge_durations)
    print_graph_stats(G)

    centrality_scores, ranked = compute_centrality(G)
    visualize_graph(G, centrality_scores)
    gnn_data = export_for_gnn(G, centrality_scores)

    print("\n" + "=" * 50)
    print("DAY 2 COMPLETE")
    print("=" * 50)
    print(f"  raw_spans.json          — {len(spans)} spans")
    print(f"  dependency_graph.png    — graph visualisation")
    print(f"  dependency_graph.json   — {gnn_data['num_nodes']} nodes, {gnn_data['num_edges']} edges")
    print("=" * 50)

if __name__ == "__main__":
    main()