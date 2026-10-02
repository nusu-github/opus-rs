#!/usr/bin/env python3
"""Summarize saved measurements and optionally render the core comparison."""
import argparse
import json
from pathlib import Path
import statistics


PROFILES = ("float", "fixed", "fixed-res24")


def load(directory, name):
    return json.loads((directory / f"{name}.json").read_text())


def aggregate(record):
    ratios = [case["rust_over_c"] for case in record["cases"]]
    return {
        "cases": len(ratios),
        "geometric_mean_rust_over_c": statistics.geometric_mean(ratios),
        "at_least_ten_percent_slower": sum(ratio >= 1.10 for ratio in ratios),
        "maximum_rust_over_c": max(ratios),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--plot", type=Path, help="Optional SVG output; requires matplotlib")
    args = parser.parse_args()
    records = {}
    summary = {"core": {}, "additional": {}}
    lines = [
        "| Profile | Before Rust/C | After Rust/C | Relative cost reduction | Cases >=1.10 before / after |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for profile in PROFILES:
        before, after = (load(args.directory, f"{profile}-{stage}") for stage in ("before", "after"))
        before_cases = {(case["name"], case["operation"]): case for case in before["cases"]}
        assert len(before_cases) == len(after["cases"]) == 12
        for case in after["cases"]:
            previous = before_cases[(case["name"], case["operation"])]
            assert previous["verified_output_sha256"] == case["verified_output_sha256"]
            assert previous["input_sha256"] == case["input_sha256"]
        old, new = aggregate(before), aggregate(after)
        reduction = 1 - new["geometric_mean_rust_over_c"] / old["geometric_mean_rust_over_c"]
        summary["core"][profile] = {"before": old, "after": new, "relative_cost_reduction": reduction}
        records[profile] = before, after
        lines.append(f"| {profile} | {old['geometric_mean_rust_over_c']:.3f} | {new['geometric_mean_rust_over_c']:.3f} | {reduction:.1%} | {old['at_least_ten_percent_slower']} / {new['at_least_ten_percent_slower']} |")
    lines += ["", "| Additional suite | Cases | Before Rust/C | After Rust/C | Relative cost reduction | Cases >=1.10 before / after |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("float-expanded", "float-plc", "pfa", "qext"):
        before = load(args.directory, f"{name}-before")
        record = load(args.directory, name)
        assert [(c["name"], c["operation"], c["verified_output_sha256"]) for c in before["cases"]] == [(c["name"], c["operation"], c["verified_output_sha256"]) for c in record["cases"]]
        previous = aggregate(before)
        result = aggregate(record)
        reduction = 1 - result["geometric_mean_rust_over_c"] / previous["geometric_mean_rust_over_c"]
        summary["additional"][name] = {"before": previous, "after": result, "relative_cost_reduction": reduction}
        lines.append(f"| {name} | {result['cases']} | {previous['geometric_mean_rust_over_c']:.3f} | {result['geometric_mean_rust_over_c']:.3f} | {reduction:.1%} | {previous['at_least_ten_percent_slower']} / {result['at_least_ten_percent_slower']} |")
    lines += ["", "| Profile / suite | Case | Operation | Rust/C | C microseconds/frame | Rust microseconds/frame |", "| --- | --- | --- | ---: | ---: | ---: |"]
    for name in (*PROFILES, "float-expanded", "float-plc", "pfa", "qext"):
        record = records[name][1] if name in records else load(args.directory, name)
        for case in record["cases"]:
            if case["rust_over_c"] >= 1.10:
                lines.append(f"| {name} | {case['name']} | {case['operation']} | {case['rust_over_c']:.3f} | {case['c_ns_per_frame'] / 1000:.1f} | {case['rust_ns_per_frame'] / 1000:.1f} |")
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.directory / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        matplotlib.rcParams["svg.hashsalt"] = "opus-performance-2026-10-02"
        figure, axes = plt.subplots(1, 3, figsize=(14, 6.5), sharex=True)
        for axis, profile in zip(axes, PROFILES):
            before, after = records[profile]
            positions = list(range(12))
            labels = [f"{case['name']} {case['operation']}" for case in after["cases"]]
            axis.barh([p - 0.18 for p in positions], [c["rust_over_c"] for c in before["cases"]], height=0.34, color="#b8bec8", label="Before")
            axis.barh([p + 0.18 for p in positions], [c["rust_over_c"] for c in after["cases"]], height=0.34, color="#187d8d", label="After")
            axis.axvline(1.0, color="#24292f", linewidth=0.8, label="C reference")
            axis.axvline(1.1, color="#ae423f", linewidth=1, linestyle="--", label="10% slower")
            axis.set_yticks(positions, labels, fontsize=8)
            axis.invert_yaxis()
            axis.set_title(profile)
            axis.set_xlabel("Thread CPU time / C thread CPU time")
            axis.grid(axis="x", alpha=0.18)
            axis.set_axisbelow(True)
        axes[0].legend(loc="lower right", fontsize=8)
        figure.suptitle("Opus scalar codec: equal outputs, before and after optimization", fontsize=13)
        figure.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(args.plot, metadata={"Date": None})
        plt.close(figure)


if __name__ == "__main__":
    main()
