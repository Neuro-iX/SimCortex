import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import gaussian_kde
# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
models = ["SimCortex", "SimCortex_M"]
metrics    = ["Chamfer", "ASSD", "HD", "SIF"]
surfaces   = ["lh_white", "rh_white", "lh_pial", "rh_pial"]
input_dir  = "/home/at83760/workspace/fifth_semester/Topology-Project/ckpts/Reconstraction_Surface_Model/exp1/finetune_manual/compare/both_eval"
output_dir = os.path.join(input_dir, "surface_table_format")
plot_dir   = os.path.join(output_dir, "plots_boxplots")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# ─── LOAD ALL DATA ─────────────────────────────────────────────────────────────
def load_all_metrics() -> pd.DataFrame:
    dfs = []
    for m in models:
        path = os.path.join(input_dir, f"surface_metrics_{m}.xlsx")
        df = pd.read_excel(path)
        df["model"] = m
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# ─── DRAW P-VALUE LINE ─────────────────────────────────────────────────────────
def draw_p_value_line(ax, x1, x2, y, pval, ymax):
    ptxt = f"p = {pval:.3f}" + ("*" if pval < 0.05 else "")
    ax.plot([x1, x2], [y, y], color='black', linewidth=1.2)
    ax.text((x1 + x2) / 2, y + 0.01 * ymax, ptxt, ha="center", fontsize=9)

# ─── CREATE TABLE I-LIKE FORMAT ────────────────────────────────────────────────
def compute_table(df_long: pd.DataFrame, surface: str) -> pd.DataFrame:
    result_rows = []
    for metric in metrics:
        metric_df = df_long[
            (df_long["surface"] == surface) & 
            (df_long["metric"] == metric)
        ].pivot(index="subject", columns="model", values="value")

        base = "SimCortex"
        for model in models:
            vals = metric_df[model].dropna()
            mean = vals.mean()
            std  = vals.std()
            row  = {
                "Metric": metric,
                "Model": model,
                "Mean": round(mean, 3),
                "SD": round(std, 3),
                "p": ""
            }
            if model != base and model in metric_df.columns and base in metric_df.columns:
                pair = metric_df[[base, model]].dropna()
                if len(pair) >= 2:
                    _, pval = ttest_rel(pair[model], pair[base])
                    row["p"] = f"{pval:.3f}" + ("*" if pval < 0.05 else "")
            result_rows.append(row)

    table = pd.DataFrame(result_rows)

    def mark_best(group):
        best_idx = group["Mean"].idxmin()
        group.loc[best_idx, "Mean"] = f"**{group.loc[best_idx, 'Mean']}**"
        return group

    return table.groupby("Metric", group_keys=False).apply(mark_best)

# ─── PER-SURFACE BOXPLOTS ──────────────────────────────────────────────────────
def plot_boxplot_with_p(df_long, surface, metric):
    subset = df_long[
        (df_long["surface"] == surface) & 
        (df_long["metric"] == metric)
    ]
    pivot = subset.pivot(index="subject", columns="model", values="value")
    present_models = [m for m in models if m in pivot.columns and pivot[m].notna().sum() >= 2]

    if len(present_models) < 2:
        return

    data = [pivot[m].dropna().values for m in present_models]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(data, labels=present_models, patch_artist=True)
    ax.set_title(f"{metric} - {surface}", fontsize=14)
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.5)

    ymax = max(np.max(d) for d in data)
    offset = ymax * 0.05
    base_idx = present_models.index("SimCortex")

    for idx, model in enumerate(present_models):
        if model != "SimCortex":
            pair = pivot[["SimCortex", model]].dropna()
            if len(pair) >= 2:
                _, pval = ttest_rel(pair[model], pair["SimCortex"])
                x1, x2 = base_idx + 1, idx + 1
                y = ymax + offset * (idx + 1)
                draw_p_value_line(ax, x1, x2, y, pval, ymax)

    ax.set_ylim(top=ymax + offset * (len(models) + 2))
    fig.tight_layout()

    out_path = os.path.join(plot_dir, f"{surface}_{metric}_boxplot.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[📊] Saved per-surface boxplot → {out_path}")

# ─── VIOLIN PLOTS ──────────────────────────────────────────

def plot_violin_with_p(df_long, surface, metric):
    """
    Draws a violin plot (mirrored KDE) for `metric` on `surface` across all models,
    and annotates paired p‐values (vs SimCortex).
    """
    # 1) Subset & pivot
    subset = df_long[
        (df_long["surface"] == surface) &
        (df_long["metric"]  == metric)
    ]
    pivot = subset.pivot(index="subject", columns="model", values="value")

    # 2) Which models have ≥2 points?
    present = [m for m in models if m in pivot.columns and pivot[m].dropna().shape[0] >= 2]
    if len(present) < 2:
        return

    # 3) Gather data arrays in order of `models`
    data = [pivot[m].dropna().values for m in present]
    positions = list(range(1, len(present) + 1))

    # 4) Create figure + violin
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.8,
        showmeans=False,
        showmedians=True
    )
    # style
    for pc in parts['bodies']:
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    if 'cmedians' in parts:
        parts['cmedians'].set_color('firebrick')

    # 5) Labels & grid
    ax.set_xticks(positions)
    ax.set_xticklabels(present)
    ax.set_title(f"{metric} — {surface}", fontsize=14)
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.4)

    # 6) Annotate p-values vs SimCortex
    ymax = max(np.max(arr) for arr in data)
    offset = ymax * 0.05
    base_idx = present.index("SimCortex") + 1

    for idx, model in enumerate(present, start=1):
        if model == "SimCortex":
            continue
        pair = pivot[["SimCortex", model]].dropna()
        if len(pair) >= 2:
            _, pval = ttest_rel(pair[model], pair["SimCortex"])
            y = ymax + offset * (idx)
            # draw line
            ax.plot([base_idx, idx], [y, y], color='black', lw=1)
            # text
            txt = f"p={pval:.3f}" + ("*" if pval < 0.05 else "")
            ax.text((base_idx + idx)/2, y + 0.01*ymax, txt,
                    ha='center', va='bottom', fontsize=9)

    # 7) Save
    out_path = os.path.join(plot_dir, f"{surface}_{metric}_violin.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[🎻] Saved violin plot → {out_path}")


def plot_box_and_kde(df_long, surface, metric):
    """
    Two‐panel plot: left = boxplot across models (with p‐values vs SimCortex),
    right = KDE curves for each model on its own scale.
    """
    # ─── Prepare data ─────────────────────────────────────────────────────────
    subset = df_long[
        (df_long["surface"] == surface) &
        (df_long["metric"]  == metric)
    ]
    pivot = subset.pivot(index="subject", columns="model", values="value")
    present = [m for m in models if m in pivot.columns and pivot[m].dropna().size >= 2]
    if len(present) < 2:
        return

    # gather arrays
    data = [pivot[m].dropna().values for m in present]

    # ─── Make the figure ──────────────────────────────────────────────────────
    fig, (ax_box, ax_kde) = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Left: boxplot with p‐values ----
    ax_box.boxplot(data, labels=present, patch_artist=True)
    ax_box.set_title(f"{metric} – {surface} (boxplot)", fontsize=12)
    ax_box.set_ylabel(metric)
    ax_box.grid(True, linestyle='--', alpha=0.3)

    # annotate p‐values vs SimCortex
    ymax = max(v.max() for v in data)
    offset = ymax * 0.05
    base_idx = present.index("SimCortex") + 1
    for idx, model in enumerate(present, start=1):
        if model == "SimCortex":
            continue
        pair = pivot[["SimCortex", model]].dropna()
        if len(pair) >= 2:
            _, pval = ttest_rel(pair[model], pair["SimCortex"])
            y = ymax + offset * idx
            ax_box.plot([base_idx, idx], [y, y], 'k-', lw=1)
            txt = f"p={pval:.3f}" + ("*" if pval<0.05 else "")
            ax_box.text((base_idx+idx)/2, y+0.01*ymax, txt,
                        ha='center', va='bottom', fontsize=9)

    # ---- Right: KDE overlay ----
    ax_kde.set_title(f"{metric} – {surface} (KDE)", fontsize=12)
    ax_kde.set_xlabel(metric)
    ax_kde.set_ylabel("Density")
    ax_kde.grid(True, linestyle='--', alpha=0.3)

    for m in present:
        vals = pivot[m].dropna().values
        kde = gaussian_kde(vals)
        xs = np.linspace(vals.min(), vals.max(), 200)
        ys = kde(xs)
        ax_kde.plot(xs, ys, lw=1.5, label=m)

    ax_kde.legend(fontsize=9)

    # ─── Save & close ────────────────────────────────────────────────────────
    fname = f"{surface}_{metric}_box_kde.png"
    out_path = os.path.join(plot_dir, fname)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[📊+📈] Saved combined plot → {out_path}")


# ─── OVERALL BOXPLOTS ACROSS SURFACES ──────────────────────────────────────────
def plot_overall_boxplot(df_long, metric):
    metric_df = df_long[df_long["metric"] == metric]
    pivot = metric_df.pivot(index=["subject", "surface"], columns="model", values="value")

    avg_data = {m: pivot[m].dropna().values for m in models}

    fig, ax = plt.subplots(figsize=(8, 5))
    vals = [avg_data[m] for m in models]
    ax.boxplot(vals, labels=models, patch_artist=True)
    ax.set_title(f"{metric} (All Surfaces Combined)", fontsize=14)
    ax.set_ylabel(metric)
    ax.grid(True, linestyle='--', alpha=0.5)

    ymax = max(np.max(v) for v in vals)
    offset = ymax * 0.05
    base_idx = models.index("SimCortex")

    for i, m in enumerate(models):
        if m != "SimCortex":
            pair = pd.DataFrame({"SimCortex": avg_data["SimCortex"], m: avg_data[m]}).dropna()
            if len(pair) >= 2:
                _, pval = ttest_rel(pair[m], pair["SimCortex"])
                x1, x2 = base_idx + 1, i + 1
                y = ymax + offset * (i + 1)
                draw_p_value_line(ax, x1, x2, y, pval, ymax)

    ax.set_ylim(top=ymax + offset * (len(models) + 2))
    fig.tight_layout()

    out_path = os.path.join(plot_dir, f"overall_{metric}_boxplot.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[📊] Saved overall boxplot → {out_path}")

# ─── MAIN SCRIPT ───────────────────────────────────────────────────────────────
df_all = load_all_metrics()

# Melt to long format
value_cols = [f"{s}_{m}" for s in surfaces for m in metrics]
df_long = df_all.melt(
    id_vars=["subject", "model"],
    value_vars=value_cols,
    var_name="surface_metric",
    value_name="value"
)
df_long[["surface", "metric"]] = df_long["surface_metric"].str.rsplit("_", n=1, expand=True)
df_long.drop(columns="surface_metric", inplace=True)

# Generate per-surface tables and plots
for surface in surfaces:
    table = compute_table(df_long, surface)
    out_path = os.path.join(output_dir, f"TableI_{surface}.xlsx")
    table.to_excel(out_path, index=False)
    print(f"[✔] Table saved → {out_path}")

    for metric in metrics:
        plot_boxplot_with_p(df_long, surface, metric)
        plot_violin_with_p(df_long, surface, metric)
        plot_box_and_kde(df_long, surface, metric)

# Generate overall boxplots
for metric in metrics:
    plot_overall_boxplot(df_long, metric)



# ─── BUILD A SINGLE SUMMARY EXCEL ─────────────────────────────────────────────
summary_rows = []
base = "SimCortex"
for surface in surfaces:
    for metric in metrics:
        # pivot for this surface/metric
        pivot = df_long[
            (df_long["surface"] == surface) &
            (df_long["metric"] == metric)
        ].pivot(index="subject", columns="model", values="value")

        if base not in pivot.columns:
            continue

        for model in models:
            if model == base or model not in pivot.columns:
                continue
            pair = pivot[[base, model]].dropna()
            if len(pair) < 2:
                continue
            stat, pval = ttest_rel(pair[model], pair[base])
            summary_rows.append({
                "Surface": surface,
                "Metric": metric,
                "Base": base,
                "Model": model,
                "t_stat": round(stat, 3),
                "p_value": round(pval, 4),
                "p_adj": round(pval * len(surfaces) * len(metrics) * (len(models)-1), 4),
                "Significant": (pval * len(surfaces) * len(metrics) * (len(models)-1)) < 0.05
            })

# assemble and save
df_summary = pd.DataFrame(summary_rows)
df_summary["Significant"] = df_summary["Significant"].astype(int)
surf_out = os.path.join(output_dir, "surface_ttests.xlsx")
df_summary.to_excel(surf_out, index=False)
print(f"[✔] Summary t-test file saved → {surf_out}")
