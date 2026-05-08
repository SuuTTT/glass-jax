with open("tests/benchmark_seclust_full.py", "r") as f:
    content = f.read()

# 1. Add true_k to rows
old_loop = """    for row in rows:
        case = cases[row["dataset"]]
        row["experiment_id"] = experiment_id"""
new_loop = """    for row in rows:
        case = cases[row["dataset"]]
        row["experiment_id"] = experiment_id
        row["true_k"] = case.k"""
content = content.replace(old_loop, new_loop)

# 2. Update synthetic_table headers
content = content.replace(
    "| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Estimate (s) | Status |",
    "| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Estimate (s) | Input (N, E, K) |"
)

# 3. Update synthetic_table row values
old_syn_row = """                        maybe_bold(fmt(row.get("map_equation")), i in map_best),
                        fmt(row.get("runtime_seconds"), 4),
                        fmt(row.get("estimated_runtime_seconds"), 1),
                        str(row.get("status", "ok")),
                    ]"""
new_syn_row = """                        maybe_bold(fmt(row.get("map_equation")), i in map_best),
                        fmt(row.get("runtime_seconds"), 4),
                        fmt(row.get("estimated_runtime_seconds"), 1),
                        f"{row.get('n_nodes')}, {row.get('n_edges')}, {row.get('true_k')}" if row.get("n_nodes") else "skip",
                    ]"""
content = content.replace(old_syn_row, new_syn_row)

# 4. Update real_world_table headers
content = content.replace(
    "| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Status |",
    "| Dataset | Algorithm | ACC | NMI | ARI | K | Modularity | StructuralEntropy | MapEquation | Time (s) | Input (N, E, K) |"
)

# 5. Update real_world_table row values
old_rw_row = """                        maybe_bold(fmt(row.get("map_equation")), i in map_best),
                        fmt(row.get("runtime_seconds"), 4),
                        str(row.get("status", "ok")),
                    ]"""
new_rw_row = """                        maybe_bold(fmt(row.get("map_equation")), i in map_best),
                        fmt(row.get("runtime_seconds"), 4),
                        f"{row.get('n_nodes')}, {row.get('n_edges')}, {row.get('true_k')}" if row.get("n_nodes") else "skip",
                    ]"""
content = content.replace(old_rw_row, new_rw_row)

with open("tests/benchmark_seclust_full.py", "w") as f:
    f.write(content)
