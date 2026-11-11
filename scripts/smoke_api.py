"""Exercise the running local service and save actual responses, not benchmark results."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Smoke record already exists")
    with httpx.Client(base_url=args.url, timeout=120, trust_env=False) as client:
        health = client.get("/health")
        health.raise_for_status()
        responses = {}
        for system in ["bm25", "dense", "hybrid", "rerank", "trained"]:
            result = client.post(
                "/search",
                json={"system": system, "top_k": 10, "query": "cell division and mitosis"},
            )
            result.raise_for_status()
            body = result.json()
            assert body["results"] and body["run_type"] == "interactive_search_not_benchmark"
            responses[system] = body
        assert client.post("/search", json={"query": " "}).status_code == 422
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "run_type": "live_local_api_smoke_not_benchmark",
                "health": health.json(),
                "results": responses,
                "blank_query_status": 422,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Verified five real search systems and request validation: {args.output}")


if __name__ == "__main__":
    main()
