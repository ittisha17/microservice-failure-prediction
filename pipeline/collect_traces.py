import requests
import json
import time
from datetime import datetime

JAEGER_URL = "http://localhost:16686"

def get_all_services():
    url = f"{JAEGER_URL}/api/services"
    response = requests.get(url)
    data = response.json()
    services = data["data"]
    # remove jaeger itself from the list
    services = [s for s in services if "jaeger" not in s.lower()]
    print(f"Found {len(services)} services: {services}")
    return services

def get_traces_for_service(service_name, limit=200):
    url = f"{JAEGER_URL}/api/traces"
    params = {
        "service": service_name,
        "limit": limit,
        "lookback": "1h"
    }
    response = requests.get(url, params=params)
    data = response.json()
    traces = data.get("data", [])
    print(f"  {service_name}: {len(traces)} traces collected")
    return traces

def parse_spans(traces):
    all_spans = []
    for trace in traces:
        spans = trace.get("spans", [])
        processes = trace.get("processes", {})
        for span in spans:
            process_id = span.get("processID", "")
            process = processes.get(process_id, {})
            service_name = process.get("serviceName", "unknown")
            span_info = {
                "traceID":       span.get("traceID"),
                "spanID":        span.get("spanID"),
                "parentSpanID":  span.get("references", [{}])[0].get("spanID", None)
                                 if span.get("references") else None,
                "operationName": span.get("operationName"),
                "serviceName":   service_name,
                "duration_us":   span.get("duration", 0),
                "startTime":     span.get("startTime", 0)
            }
            all_spans.append(span_info)
    return all_spans

def main():
    print("=" * 50)
    print("TRACE COLLECTOR STARTING")
    print("=" * 50)

    # Step 1 - get all services
    services = get_all_services()

    # Step 2 - collect traces for each service
    all_traces = []
    for service in services:
        traces = get_traces_for_service(service)
        all_traces.extend(traces)

    print(f"\nTotal traces collected: {len(all_traces)}")

    # Step 3 - parse all spans
    all_spans = parse_spans(all_traces)
    print(f"Total spans parsed: {len(all_spans)}")

    # Step 4 - save raw spans to file
    with open("raw_spans.json", "w") as f:
        json.dump(all_spans, f, indent=2)

    print("\nraw_spans.json saved successfully")
    print("=" * 50)
    print("TRACE COLLECTION COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    main()