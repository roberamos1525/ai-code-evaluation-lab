"""Use codeeval as a library: run an evaluation and compare it with a baseline in-process.

python examples/library_usage.py
"""

from __future__ import annotations

import tempfile

from codeeval.analysis import RegressionPolicy
from codeeval.config import load_settings
from codeeval.context import AppContext
from codeeval.reporting import build_comparison_report, render_report


def main() -> None:
    with tempfile.TemporaryDirectory() as home:
        ctx = AppContext.from_settings(load_settings(overrides={"home": home}))
        try:
            baseline = ctx.engine.run(ctx.default_run_config(adapter="oracle", limit=10))
            candidate = ctx.engine.run(
                ctx.default_run_config(
                    adapter="mutation:bug_rate=0.6,seed=2", limit=10, samples=3, ks=(1, 3)
                )
            )
            print("baseline pass@1:", baseline.summary["pass_at_k"]["pass@1"])
            print("candidate pass@k:", candidate.summary["pass_at_k"])
            report = build_comparison_report(
                ctx.store,
                baseline.run_id,
                candidate.run_id,
                policy=RegressionPolicy(max_pass_drop=0.1),
            )
            print(render_report(report, "md").split("## Task changes")[0])
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
