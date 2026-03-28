import json
from neo4j import GraphDatabase

NEO4J_URI      = "bolt://localhost:7687"
NEO4J_USER     = "neo4j"
NEO4J_PASSWORD = "password123"

def load_graph():
    with open("dependency_graph.json") as f:
        return json.load(f)

def load_fault_scenarios():
    with open("fault_scenarios.json") as f:
        return json.load(f)

class Neo4jGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("  Cleared existing graph")

    def create_service_nodes(self, graph):
        with self.driver.session() as session:
            for node in graph["node_features"]:
                session.run("""
                    CREATE (s:Service {
                        name:              $name,
                        node_id:           $node_id,
                        in_degree:         $in_degree,
                        out_degree:        $out_degree,
                        criticality_score: $criticality_score,
                        total_incoming_calls: $total_incoming_calls
                    })
                """, {
                    "name":                 node["service_name"],
                    "node_id":              node["node_id"],
                    "in_degree":            node["in_degree"],
                    "out_degree":           node["out_degree"],
                    "criticality_score":    node["criticality_score"],
                    "total_incoming_calls": node["total_incoming_calls"]
                })
        print(f"  Created {len(graph['node_features'])} service nodes")

    def create_dependency_edges(self, graph):
        with self.driver.session() as session:
            for edge in graph["edge_list"]:
                session.run("""
                    MATCH (a:Service {name: $source})
                    MATCH (b:Service {name: $target})
                    CREATE (a)-[:DEPENDS_ON {
                        weight:          $weight,
                        call_count:      $call_count,
                        avg_duration_ms: $avg_duration_ms
                    }]->(b)
                """, {
                    "source":          edge["source_name"],
                    "target":          edge["target_name"],
                    "weight":          edge["weight"],
                    "call_count":      edge["call_count"],
                    "avg_duration_ms": edge["avg_duration_ms"]
                })
        print(f"  Created {len(graph['edge_list'])} dependency edges")

    def create_fault_scenarios(self, scenarios):
        with self.driver.session() as session:
            for scenario in scenarios:
                # Create fault scenario node
                session.run("""
                    CREATE (f:FaultScenario {
                        scenario_id:   $scenario_id,
                        fault_service: $fault_service,
                        total_spans:   $total_spans,
                        error_spans:   $error_spans
                    })
                """, {
                    "scenario_id":   scenario["scenario_id"],
                    "fault_service": scenario["fault_service"],
                    "total_spans":   scenario["total_spans"],
                    "error_spans":   scenario["error_spans"]
                })

                # Link fault scenario to affected services
                for svc, impact in scenario["blast_radius"].items():
                    if impact > 0:
                        session.run("""
                            MATCH (f:FaultScenario {scenario_id: $sid})
                            MATCH (s:Service {name: $svc})
                            CREATE (f)-[:AFFECTS {impact_score: $impact}]->(s)
                        """, {
                            "sid":    scenario["scenario_id"],
                            "svc":    svc,
                            "impact": impact
                        })

        print(f"  Created {len(scenarios)} fault scenario nodes")

    def get_stats(self):
        with self.driver.session() as session:
            nodes    = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
            edges    = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            services = session.run("MATCH (s:Service) RETURN count(s) AS c").single()["c"]
            faults   = session.run("MATCH (f:FaultScenario) RETURN count(f) AS c").single()["c"]
        return nodes, edges, services, faults

    def query_blast_radius(self, fault_service):
        with self.driver.session() as session:
            result = session.run("""
                MATCH (f:FaultScenario {fault_service: $svc})-[a:AFFECTS]->(s:Service)
                RETURN s.name AS service, a.impact_score AS impact
                ORDER BY a.impact_score DESC
            """, {"svc": fault_service})
            return [(r["service"], r["impact"]) for r in result]

    def query_critical_path(self, fault_service):
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (s:Service {name: $svc})-[:DEPENDS_ON*1..3]->(t:Service)
                RETURN [node in nodes(path) | node.name] AS path,
                       length(path) AS hops
                ORDER BY hops
            """, {"svc": fault_service})
            return [(r["path"], r["hops"]) for r in result]

def main():
    print("=" * 55)
    print("NEO4J GRAPH DATABASE INTEGRATION")
    print("=" * 55)

    graph     = load_graph()
    scenarios = load_fault_scenarios()
    neo4j     = Neo4jGraph()

    print("\n[1] Clearing existing data...")
    neo4j.clear_graph()

    print("\n[2] Creating service nodes...")
    neo4j.create_service_nodes(graph)

    print("\n[3] Creating dependency edges...")
    neo4j.create_dependency_edges(graph)

    print("\n[4] Creating fault scenarios...")
    neo4j.create_fault_scenarios(scenarios)

    print("\n[5] Verifying graph...")
    nodes, edges, services, faults = neo4j.get_stats()
    print(f"  Total nodes:     {nodes}")
    print(f"  Total edges:     {edges}")
    print(f"  Service nodes:   {services}")
    print(f"  Fault scenarios: {faults}")

    print("\n[6] Sample query — blast radius for mysql failure:")
    blast = neo4j.query_blast_radius("mysql")
    for svc, impact in blast:
        bar = "█" * int(impact * 10)
        print(f"  {svc:<16} {impact:.2f}  {bar}")

    print("\n[7] Sample query — dependency paths from frontend:")
    paths = neo4j.query_critical_path("frontend")
    for path, hops in paths:
        print(f"  {' → '.join(path)}  ({hops} hops)")

    neo4j.close()

    print("\n" + "=" * 55)
    print("NEO4J INTEGRATION COMPLETE")
    print("=" * 55)
    print("Open Neo4j Browser at: http://localhost:7474")
    print("Run this query to see your graph:")
    print("  MATCH (n)-[r]->(m) RETURN n,r,m")
    print("=" * 55)

if __name__ == "__main__":
    main()
