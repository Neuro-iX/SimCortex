import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
models    = ["SimCortex", "CFPP", "V2C"]
metrics   = ["HD", "ASSD"]
input_dir = "/home/at83760/workspace/fifth_semester/Topology-Project/ckpts/Reconstraction_Surface_Model/files_models"
plot_dir  = os.path.join(input_dir, "surface_table_format", "plots_violin_only")
os.makedirs(plot_dir, exist_ok=True)

# ─── LOAD & PREPARE DATA ───────────────────────────────────────────────────────
def load_all_metrics():
    dfs = []
    for m in models:
        path = os.path.join(input_dir, f"surface_metrics_{m}.xlsx")
        if not os.path.exists(path):
            print(f"⚠️  Missing file: {path}")
            continue
        df = pd.read_excel(path)
        df["model"] = m
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

df = load_all_metrics()
print("🔍 Loaded DataFrame shape:", df.shape)
print("🔍 Models present:", df["model"].unique())

# compute per-subject pial/white averages for each metric
for met in metrics:
    df[f"white_{met}"] = df[[f"lh_white_{met}", f"rh_white_{met}"]].mean(axis=1)
    df[f"pial_{met}"]  = df[[f"lh_pial_{met}",  f"rh_pial_{met}"]].mean(axis=1)

# ─── PLOTTING ─────────────────────────────────────────────────────────────────
# Offsets so violins for each model don't overlap
offsets = np.linspace(-0.2, 0.2, len(models))
colors  = {"SimCortex":"C0", "CFPP":"C1", "V2C":"C2"}

fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharey=False)

for ax, met in zip(axes, metrics):
    # draw a violin for each model at x=1 (Pial) and x=2 (White)
    for i, model in enumerate(models):
        # Pial
        vals_p = df.loc[df.model==model, f"pial_{met}"].dropna().values
        pos_p  = 1 + offsets[i]
        parts  = ax.violinplot([vals_p], positions=[pos_p],
                               widths=0.3, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[model])
            pc.set_edgecolor('k')
            pc.set_alpha(0.7)
        # White
        vals_w = df.loc[df.model==model, f"white_{met}"].dropna().values
        pos_w  = 2 + offsets[i]
        parts  = ax.violinplot([vals_w], positions=[pos_w],
                               widths=0.3, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(colors[model])
            pc.set_edgecolor('k')
            pc.set_alpha(0.7)
    # axis styling
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Pial Surface", "White Surface"], fontsize=12)
    ax.set_ylabel(f"{met} (mm)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.3)

# shared legend: one entry per model
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], color=colors[m], marker='o', linestyle='',
           markersize=8, label=m) for m in models
]
fig.legend(
    handles=handles,
    title="Model",
    loc="upper right",
    bbox_to_anchor=(1, 0.85),
    fontsize=10,
    title_fontsize=11
)

# main title
fig.suptitle(
    "Violin Plots of Pial vs White Averages for HD and ASSD",
    fontsize=16,
    x=0.48,
    y=0.95
)

fig.tight_layout(rect=[0, 0, 0.88, 0.93])

# save figure
out_path = os.path.join(plot_dir, "violin_HD_ASSD.png")
fig.savefig(out_path, bbox_inches="tight")
plt.close(fig)

print(f"▶ Saved violin‐only figure → {out_path}")
