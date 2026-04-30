#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


def _parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Invalid --run spec '{spec}'. Expected METHOD=/path/to/run")
    method, path_str = spec.split("=", 1)
    return method.strip(), Path(path_str).expanduser().resolve()


def _resolve_events_path(run_path: Path) -> Path:
    if run_path.is_file():
        return run_path
    candidates = [
        run_path / "wallclock_study" / "events.jsonl",
        run_path / "events.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find wallclock events under {run_path}")


def _resolve_meta_path(events_path: Path) -> Path | None:
    candidate = events_path.parent / "meta.json"
    return candidate if candidate.exists() else None


def _load_run(run_path: Path) -> dict:
    events_path = _resolve_events_path(run_path)
    meta_path = _resolve_meta_path(events_path)
    with events_path.open("r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]
    if not events:
        raise ValueError(f"No events found in {events_path}")
    meta = {}
    if meta_path is not None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return {"events_path": events_path, "meta": meta, "events": events}


def _safe_metric(event: dict, metric_key: str):
    return event.get("metrics", {}).get(metric_key, None)


def _best_score(events: list[dict], metric_key: str, budget_hours: float | None = None):
    best_value = None
    best_event = None
    for event in events:
        if not event.get("is_effective_update", False):
            continue
        if budget_hours is not None and float(event.get("elapsed_hours", math.inf)) > budget_hours:
            continue
        value = _safe_metric(event, metric_key)
        if value is None:
            continue
        if best_value is None or float(value) > float(best_value):
            best_value = float(value)
            best_event = event
    return best_value, best_event


def _time_to_target(events: list[dict], metric_key: str, target: float):
    for event in sorted(events, key=lambda x: float(x.get("elapsed_time_s", 0.0))):
        if not event.get("is_effective_update", False):
            continue
        value = _safe_metric(event, metric_key)
        if value is None:
            continue
        if float(value) >= float(target):
            return float(event["elapsed_hours"]), event
    return None, None


def _median_or_nan(values: list[float]) -> float:
    return float(median(values)) if values else math.nan


def _flatten_validation_rows(method: str, events: list[dict], metric_key: str) -> list[dict]:
    rows = []
    for event in events:
        if not event.get("is_effective_update", False):
            continue
        value = _safe_metric(event, metric_key)
        if value is None:
            continue
        rows.append(
            {
                "method": method,
                "global_step": int(event["global_step"]),
                "effective_update_index": int(event["effective_update_index"]),
                "elapsed_hours": float(event["elapsed_hours"]),
                "metric_key": metric_key,
                "metric_value": float(value),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize 14B wall-clock efficiency study runs.")
    parser.add_argument("--run", action="append", required=True, help="METHOD=/path/to/run_dir_or_events.jsonl")
    parser.add_argument("--metric-key", default="val-core/aime_combined_score")
    parser.add_argument("--budget-hours", type=float, default=12.0)
    parser.add_argument("--target-reference", default="DAPO")
    parser.add_argument("--target-fraction", type=float, default=0.95)
    parser.add_argument("--target-value", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = {}
    for spec in args.run:
        method, path = _parse_run(spec)
        runs[method] = _load_run(path)

    if args.target_value is not None:
        target_value = float(args.target_value)
        target_source = "absolute"
    else:
        if args.target_reference not in runs:
            raise ValueError(f"Target reference method '{args.target_reference}' not found in supplied runs")
        ref_best, ref_best_event = _best_score(
            runs[args.target_reference]["events"],
            args.metric_key,
            budget_hours=args.budget_hours,
        )
        if ref_best is None:
            raise ValueError(
                f"Reference run '{args.target_reference}' does not contain validation metric '{args.metric_key}' "
                f"within budget_hours={args.budget_hours}"
            )
        target_value = float(ref_best) * float(args.target_fraction)
        target_source = f"{args.target_fraction:.3f}x_{args.target_reference}_best_within_{args.budget_hours:g}h"

    runtime_rows = []
    fixed_budget_rows = []
    time_to_target_rows = []
    validation_curve_rows = []

    for method, payload in runs.items():
        events = payload["events"]
        meta = payload["meta"]
        raw_events = events
        effective_events = [event for event in events if event.get("is_effective_update", False)]

        raw_cycle_times = [float(event["cycle_time_s"]) for event in raw_events]
        effective_update_times = [
            float(event["effective_update_time_s"])
            for event in effective_events
            if event.get("effective_update_time_s") is not None
        ]

        effective_stage_names = sorted(
            {
                name
                for event in effective_events
                for name in event.get("effective_timing_s", {}).keys()
            }
        )

        runtime_row = {
            "method": method,
            "events_path": str(payload["events_path"]),
            "raw_cycle_count": len(raw_events),
            "effective_update_count": len(effective_events),
            "raw_per_effective_ratio": (len(raw_events) / len(effective_events)) if effective_events else math.nan,
            "median_raw_cycle_time_s": _median_or_nan(raw_cycle_times),
            "median_effective_update_time_s": _median_or_nan(effective_update_times),
            "total_gpus": meta.get("total_gpus", math.nan),
        }

        for stage_name in effective_stage_names:
            runtime_row[f"median_effective_timing_s/{stage_name}"] = _median_or_nan(
                [
                    float(event["effective_timing_s"][stage_name])
                    for event in effective_events
                    if stage_name in event.get("effective_timing_s", {})
                ]
            )
        runtime_rows.append(runtime_row)

        best_budget_value, best_budget_event = _best_score(events, args.metric_key, budget_hours=args.budget_hours)
        fixed_budget_rows.append(
            {
                "method": method,
                "metric_key": args.metric_key,
                "budget_hours": args.budget_hours,
                "best_metric_within_budget": best_budget_value if best_budget_value is not None else math.nan,
                "best_metric_step": best_budget_event["global_step"] if best_budget_event is not None else math.nan,
                "best_metric_elapsed_hours": (
                    float(best_budget_event["elapsed_hours"]) if best_budget_event is not None else math.nan
                ),
            }
        )

        ttt_hours, ttt_event = _time_to_target(events, args.metric_key, target_value)
        total_gpus = meta.get("total_gpus", math.nan)
        time_to_target_rows.append(
            {
                "method": method,
                "metric_key": args.metric_key,
                "target_value": target_value,
                "target_source": target_source,
                "time_to_target_hours": ttt_hours if ttt_hours is not None else math.nan,
                "gpu_hours_to_target": (
                    ttt_hours * float(total_gpus)
                    if ttt_hours is not None and not math.isnan(float(total_gpus))
                    else math.nan
                ),
                "target_reached": ttt_event is not None,
                "target_step": ttt_event["global_step"] if ttt_event is not None else math.nan,
            }
        )

        validation_curve_rows.extend(_flatten_validation_rows(method, events, args.metric_key))

    files_and_rows = [
        ("runtime_summary.csv", runtime_rows),
        ("fixed_budget_scores.csv", fixed_budget_rows),
        ("time_to_target.csv", time_to_target_rows),
        ("validation_over_time.csv", validation_curve_rows),
    ]

    for filename, rows in files_and_rows:
        path = output_dir / filename
        if not rows:
            path.write_text("", encoding="utf-8")
            continue
        fieldnames = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "metric_key": args.metric_key,
        "budget_hours": args.budget_hours,
        "target_value": target_value,
        "target_source": target_source,
        "methods": sorted(runs.keys()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
