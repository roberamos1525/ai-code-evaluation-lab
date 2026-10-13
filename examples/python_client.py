"""Drive the REST API from Python with httpx.

Start the server (``codeeval serve``) and run:
    python examples/python_client.py --base http://127.0.0.1:8000 --key <api key or empty>
"""

from __future__ import annotations

import argparse
import time

import httpx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--key", default="")
    args = parser.parse_args()
    headers = {"X-API-Key": args.key} if args.key else {}
    with httpx.Client(base_url=args.base, headers=headers, timeout=30) as client:
        runs = {}
        for adapter in ("oracle", "mutation:bug_rate=0.5,seed=3"):
            response = client.post("/v1/runs", json={"adapter": adapter, "limit": 15})
            response.raise_for_status()
            runs[adapter] = response.json()["run_id"]
        for adapter, run_id in runs.items():
            while (run := client.get(f"/v1/runs/{run_id}").json())["status"] in (
                "pending",
                "running",
            ):
                time.sleep(0.3)
            print(
                f"{adapter:32} {run['status']:10} pass@1={run['summary']['pass_at_k']['pass@1']:.3f}"
            )
        gate = client.post(
            "/v1/regression",
            json={"baseline": runs["oracle"], "candidate": runs["mutation:bug_rate=0.5,seed=3"]},
        ).json()
        print("regression verdict:", gate["verdict"])
        for check in gate["checks"]:
            print(
                f"  {check['name']:16} {'pass' if check['passed'] else 'FAIL'}  {check['message']}"
            )


if __name__ == "__main__":
    main()
