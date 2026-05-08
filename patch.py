import re

with open("tests/benchmark_seclust_full.py", "r") as f:
    content = f.read()

# 1. Modify run_benchmark return
old_return = "    return rows\n"
new_return = """    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"seclust_full_benchmark_{timestamp}"
    import numpy as np
    for row in rows:
        case = cases[row["dataset"]]
        row["experiment_id"] = experiment_id
        row["dataset_source"] = case.source
        row["n_nodes"] = case.adjacency.shape[0] if case.adjacency is not None else None
        row["n_edges"] = int(np.sum(case.adjacency > 0) / 2) if case.adjacency is not None else None
        
        algo = str(row.get("algorithm", ""))
        if "SEClust" in algo or algo in ["Louvain", "Infomap", "Glass-Mod (JAX)", "Glass-Map (JAX)"]:
            row["seed"] = SECLUST_SEED
        else:
            row["seed"] = None
            
        status_orig = str(row.get("status", "ok"))
        if status_orig.startswith("skipped") or status_orig.startswith("unavailable"):
            row["skip_reason"] = status_orig
            row["status"] = "skipped"
        elif status_orig == "baseline_executed":
            row["status"] = "ok"
            row["skip_reason"] = None
        elif status_orig == "baseline_imported":
            row["status"] = "baseline_imported"
            row["skip_reason"] = None
        else:
            row["status"] = "ok"
            row["skip_reason"] = None
            
        row["estimated_runtime_seconds"] = row.pop("estimated_time", None)
        row["runtime_seconds"] = row.pop("time", None)
        row["labels_path"] = None
        row.pop("se", None)
        
    return rows
"""
content = content.replace(old_return, new_return)

# 2. Modify write_report
old_write_report = """def write_report(rows: list[dict[str, object]]) -> Path:
    out_dir = Path("docs/experimental_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "seclust_full_benchmark_20260507.json"
    report_path = out_dir / "seclust_full_benchmark_20260507.md\""""
new_write_report = """def write_report(rows: list[dict[str, object]]) -> Path:
    out_dir = Path("docs/experimental_reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_id = rows[0]["experiment_id"] if rows else "seclust_full_benchmark"
    json_path = out_dir / f"{exp_id}.json"
    report_path = out_dir / f"{exp_id}.md\""""
content = content.replace(old_write_report, new_write_report)

# 3. Modify synthetic_table and real_world_table
content = content.replace('fmt(row.get("time"), 4)', 'fmt(row.get("runtime_seconds"), 4)')
content = content.replace('fmt(row.get("estimated_time"), 1)', 'fmt(row.get("estimated_runtime_seconds"), 1)')

with open("tests/benchmark_seclust_full.py", "w") as f:
    f.write(content)
