import json
import random
import copy
from itertools import combinations

random.seed(42)

def load_data():
    with open("dependency_graph.json") as f:
        graph = json.load(f)
    with open("gnn_training_labels.json") as f:
        labels = json.load(f)
    return graph, labels

def build_affected_by(graph):
    affected_by = {}
    for edge in graph["edge_list"]:
        callee = edge["target_name"]
        caller = edge["source_name"]
        if callee not in affected_by:
            affected_by[callee] = []
        affected_by[callee].append(caller)
    return affected_by

def get_blast_radius(fault_services, affected_by, all_services):
    blast = {}
    for fs in fault_services:
        blast[fs] = 1.0
    queue   = list(fault_services)
    visited = set(fault_services)
    depth   = 1
    while queue:
        next_queue = []
        for current in queue:
            for caller in affected_by.get(current, []):
                if caller not in visited:
                    visited.add(caller)
                    impact = round(max(0.2, 1.0 - depth * 0.2), 4)
                    blast[caller] = max(blast.get(caller, 0), impact)
                    next_queue.append(caller)
        queue = next_queue
        depth += 1
    for svc in all_services:
        if svc not in blast:
            blast[svc] = 0.0
    return blast

def augment(graph, labels):
    services    = graph["service_names"]
    affected_by = build_affected_by(graph)
    augmented   = []

    # Original 6 single-fault scenarios
    for label in labels:
        augmented.append(label)

    scenario_id = len(labels) + 1

    # Two-service simultaneous failures — 15 combinations
    for svc_a, svc_b in combinations(services, 2):
        blast  = get_blast_radius([svc_a, svc_b], affected_by, services)
        labels_vec = [blast.get(s, 0.0) for s in services]
        augmented.append({
            "scenario_id":    scenario_id,
            "fault_service":  f"{svc_a}+{svc_b}",
            "fault_node_idx": services.index(svc_a),
            "label_vector":   labels_vec,
            "service_names":  services
        })
        scenario_id += 1

    # Impact variation scenarios — vary weights slightly
    for label in labels[:6]:
        for variation in [0.8, 1.2]:
            varied = [min(1.0, round(v * variation + random.uniform(-0.05, 0.05), 4))
                     for v in label["label_vector"]]
            # fault node always stays 1.0
            varied[label["fault_node_idx"]] = 1.0
            augmented.append({
                "scenario_id":    scenario_id,
                "fault_service":  label["fault_service"] + f"_v{variation}",
                "fault_node_idx": label["fault_node_idx"],
                "label_vector":   varied,
                "service_names":  services
            })
            scenario_id += 1

    return augmented

def main():
    print("=" * 50)
    print("TRAINING DATA AUGMENTATION")
    print("=" * 50)

    graph, labels = load_data()
    augmented     = augment(graph, labels)

    with open("gnn_training_labels.json", "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"Original scenarios:  6")
    print(f"After augmentation:  {len(augmented)}")
    print(f"\nBreakdown:")
    print(f"  Single fault scenarios:      6")
    print(f"  Two-fault combinations:      15")
    print(f"  Impact variation scenarios:  12")
    print(f"  Total:                       {len(augmented)}")
    print(f"\ngnn_training_labels.json updated")
    print("=" * 50)

if __name__ == "__main__":
    main()


