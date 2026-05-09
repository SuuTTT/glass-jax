"""Convert an aggregated benchmark JSON into the LaTeX table bodies
(``sections/table_synthetic.tex`` and ``sections/table_realworld.tex``)
used by the SEClust paper.

Usage:
    PYTHONPATH=src python3 tests/aggregated_to_latex.py \\
        docs/experimental_reports/seclust_full_benchmark_<ts>_aggregated.json \\
        /workspace/SEClust-paper/sections/

The script writes ``table_synthetic.tex`` and ``table_realworld.tex``
in the target directory. Each cell is rendered with a ``\\meanstd``
macro that the paper's preamble defines as
``\\newcommand{\\meanstd}[2]{$#1{\\scriptstyle \\pm #2}$}``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


SYNTHETIC_DATASETS = ["Karate", "Caveman (10x20)", "SBM (N=100)", "SBM (N=500)", "SBM (N=1000)"]
REAL_WORLD_DATASETS = ["Cora", "Citeseer", "Photo"]

ALGORITHM_ORDER_SYNTHETIC = [
    "Louvain", "Leiden", "Infomap",
    "Glass-Mod (JAX)", "Glass-Map (JAX)",
    "HCSE",
    "SEClust-Auto", "SEClust-Tree", "SEClust-TargetK", "SEClust-MultiLevel",
]
ALGORITHM_ORDER_REAL_WORLD = [
    "Louvain", "Leiden", "Infomap",
    "HCSE", "LSEnet", "Glass-SE GNN",
    "SEClust-Auto", "SEClust-Tree", "SEClust-TargetK", "SEClust-MultiLevel",
]

DATASET_LABEL_TEX = {
    "Karate": "Karate",
    "Caveman (10x20)": "Caveman (10$\\times$20)",
    "SBM (N=100)": "SBM ($N{=}100$)",
    "SBM (N=500)": "SBM ($N{=}500$)",
    "SBM (N=1000)": "SBM ($N{=}1000$)",
}


def _fmt(mean, std, digits: int = 3) -> str:
    if mean is None:
        return "---"
    if not isinstance(mean, (int, float)):
        return str(mean)
    if std is None or not isinstance(std, (int, float)) or std < 1e-6:
        return f"{float(mean):.{digits}f}"
    return f"\\meanstd{{{float(mean):.{digits}f}}}{{{float(std):.{digits}f}}}"


def _bold_if(value_str: str, is_best: bool) -> str:
    return f"\\textbf{{{value_str}}}" if is_best else value_str


def _best_indices(group: list[dict], field: str, lower_is_better: bool = False) -> set[int]:
    pairs = [
        (i, row.get(f"{field}_mean") if row.get(f"{field}_mean") is not None else row.get(field))
        for i, row in enumerate(group)
    ]
    pairs = [(i, v) for i, v in pairs if isinstance(v, (int, float))]
    if not pairs:
        return set()
    best = min(v for _, v in pairs) if lower_is_better else max(v for _, v in pairs)
    return {i for i, v in pairs if abs(v - best) < 1e-12}


def _group_by_algorithm(group: list[dict], order: list[str]) -> list[dict]:
    by_algo = {row["algorithm"]: row for row in group}
    return [by_algo[a] for a in order if a in by_algo]


def _row_for(group: list[dict], dataset_label: str, dataset_first: bool) -> list[str]:
    """Render the rows for one dataset, returning a list of LaTeX lines."""
    acc_best = _best_indices(group, "acc")
    nmi_best = _best_indices(group, "nmi")
    ari_best = _best_indices(group, "ari")
    mod_best = _best_indices(group, "modularity")
    se_best = _best_indices(group, "structural_entropy", lower_is_better=True)
    map_best = _best_indices(group, "map_equation", lower_is_better=True)

    out = []
    for i, row in enumerate(group):
        ds_cell = dataset_label if i == 0 else ""
        if str(row.get("status")) == "skipped" and row.get("acc_mean") is None and row.get("acc") is None:
            cells = [
                ds_cell,
                row["algorithm"],
                "\\multicolumn{8}{c}{\\emph{" + str(row.get("skip_reason", "skipped")) + "}}",
            ]
            out.append(" & ".join(cells) + " \\\\")
            continue
        acc = _bold_if(_fmt(row.get("acc_mean"), row.get("acc_std")), i in acc_best)
        nmi = _bold_if(_fmt(row.get("nmi_mean"), row.get("nmi_std")), i in nmi_best)
        ari = _bold_if(_fmt(row.get("ari_mean"), row.get("ari_std")), i in ari_best)
        k_str = f"{int(row['k'])}" if row.get("k") is not None else "---"
        mod = _bold_if(_fmt(row.get("modularity_mean"), row.get("modularity_std")), i in mod_best)
        se = _bold_if(_fmt(row.get("structural_entropy_mean"), row.get("structural_entropy_std")), i in se_best)
        meq = _bold_if(_fmt(row.get("map_equation_mean"), row.get("map_equation_std")), i in map_best)
        t = _fmt(row.get("runtime_seconds_mean"), row.get("runtime_seconds_std"), digits=2)
        cells = [ds_cell, row["algorithm"], acc, nmi, ari, k_str, mod, se, meq, t]
        out.append(" & ".join(cells) + " \\\\")
    return out


def synthetic_latex(rows: list[dict]) -> str:
    body = []
    for ds in SYNTHETIC_DATASETS:
        group = _group_by_algorithm([r for r in rows if r["dataset"] == ds], ALGORITHM_ORDER_SYNTHETIC)
        if not group:
            continue
        body.extend(_row_for(group, DATASET_LABEL_TEX.get(ds, ds), dataset_first=True))
        if ds != SYNTHETIC_DATASETS[-1]:
            body.append("\\midrule")
    return "\n".join(body)


def real_world_latex(rows: list[dict]) -> str:
    body = []
    for ds in REAL_WORLD_DATASETS:
        group = _group_by_algorithm([r for r in rows if r["dataset"] == ds], ALGORITHM_ORDER_REAL_WORLD)
        if not group:
            continue
        # Add (N, E, K*) annotation in dataset cell
        n = group[0].get("n_nodes") or "?"
        e = group[0].get("n_edges") or "?"
        kstar = group[0].get("true_k") or "?"
        ds_label = f"{ds} ($N{{=}}{n}$, $E{{=}}{e}$, $K^*{{=}}{kstar}$)"
        # Override DATASET_LABEL_TEX for this row
        rendered = _row_for(group, ds_label, dataset_first=True)
        body.extend(rendered)
        if ds != REAL_WORLD_DATASETS[-1]:
            body.append("\\midrule")
    return "\n".join(body)


SYNTHETIC_HEADER = r"""\begin{table*}[t]
\centering
\caption{Synthetic results, mean $\pm$ std over five seeds
($\sigma\!\in\!\{0, 7, 17, 23, 42\}$). ACC, NMI, ARI, modularity ($Q$),
structural entropy ($H_2$, lower is better), map equation ($L$, lower
is better), recovered $K$ (median across seeds), wall-clock runtime in
seconds. Bold marks the best per (dataset, metric) cell; $K$ is
diagnostic and is not bolded.}
\label{tab:synthetic}
\renewcommand{\arraystretch}{1.10}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccccr}
\toprule
Dataset & Algorithm & ACC & NMI & ARI & K & Q & $H_2$ & L & Time (s) \\
\midrule
"""

SYNTHETIC_FOOTER = r"""\bottomrule
\end{tabular}
\end{table*}
"""


REAL_WORLD_HEADER = r"""\begin{table*}[t]
\centering
\caption{Real-world results on Cora / Citeseer / Photo (loaded sparsely
from PyG), mean $\pm$ std over five seeds ($\sigma\!\in\!\{0, 7, 17, 23, 42\}$).
Bold marks the best per (dataset, metric) cell. \HCSE is skipped on
Photo because its dense $(N,N)$ adjacency materialisation is
infeasible above $N=5{,}000$. \LSEnet here is the linear-projection
proxy; \textsc{Glass-SE GNN} is the GCN-encoder feature-aware variant
that consumes the same input as \LSEnet. The published \LSEnet
PyTorch~+~Lorentz numbers are imported in \Sec{sec:comparison}.}
\label{tab:realworld}
\renewcommand{\arraystretch}{1.10}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccccccr}
\toprule
Dataset & Algorithm & ACC & NMI & ARI & K & Q & $H_2$ & L & Time (s) \\
\midrule
"""

REAL_WORLD_FOOTER = r"""\bottomrule
\end{tabular}
\end{table*}
"""


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        sys.exit(1)
    aggregated_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(aggregated_path.read_text())

    syn = SYNTHETIC_HEADER + synthetic_latex(rows) + "\n" + SYNTHETIC_FOOTER
    rw = REAL_WORLD_HEADER + real_world_latex(rows) + "\n" + REAL_WORLD_FOOTER

    (out_dir / "table_synthetic.tex").write_text(syn, encoding="utf-8")
    (out_dir / "table_realworld.tex").write_text(rw, encoding="utf-8")

    print(f"Wrote {out_dir/'table_synthetic.tex'} ({len(syn)} bytes)")
    print(f"Wrote {out_dir/'table_realworld.tex'} ({len(rw)} bytes)")


if __name__ == "__main__":
    main()
