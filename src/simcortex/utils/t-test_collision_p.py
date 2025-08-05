import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────
models = ["SimCortex", "SimCortex_M"]
surfaces   = ["pial_lr", "white_lr", "white_pial_left", "white_pial_right"]
input_dir  = "/home/at83760/workspace/fifth_semester/Topology-Project/ckpts/Reconstraction_Surface_Model/exp1/finetune_manual/compare/both_eval"
output_dir = os.path.join(input_dir, "collision_surface_percentage_table")
os.makedirs(output_dir, exist_ok=True)

# ─── HELPERS ───────────────────────────────────────────────────────────────────
def parse_pair(s):
    a, b = s.strip("()").split(",")
    return int(a), int(b)

def compute_collision_pct(df, key):
    total = df[f"{key}_total_faces"].map(parse_pair)
    inter = df[f"{key}_intersecting_faces"].map(parse_pair)
    tL, tR = zip(*total)
    iL, iR = zip(*inter)
    pctL = np.array(iL) / np.array(tL) * 100
    pctR = np.array(iR) / np.array(tR) * 100
    return (pctL + pctR) / 2  # average

# ─── MAIN ──────────────────────────────────────────────────────────────────────
all_model_dfs = {}

for model in models:
    path = os.path.join(input_dir, f"collision_metrics_{model}.xlsx")
    df = pd.read_excel(path)
    df["model"] = model
    all_model_dfs[model] = df

# ─── PROCESS EACH SURFACE ──────────────────────────────────────────────────────
for surface in surfaces:
    all_data = []

    for model in models:
        df = all_model_dfs[model].copy()
        df["collision_pct"] = compute_collision_pct(df, surface)
        for _, row in df.iterrows():
            all_data.append({
                "subject": row["subject"],
                "model": model,
                "collision_pct": row["collision_pct"]
            })

    df_long = pd.DataFrame(all_data)

    # pivot to wide for easier stats
    df_wide = df_long.pivot(index="subject", columns="model", values="collision_pct")

    result_rows = []

    for model in models:
        vals = df_wide[model].dropna()
        mean = vals.mean()
        std = vals.std()
        row = {
            "Model": model,
            "Mean": round(mean, 3),
            "SD": round(std, 3),
            "p": ""
        }

        if model != "SimCortex" and "SimCortex" in df_wide.columns:
            pair = df_wide[[model, "SimCortex"]].dropna()
            if len(pair) >= 2:
                _, pval = ttest_rel(pair[model], pair["SimCortex"])
                row["p"] = f"{pval:.3f}" + ("*" if pval < 0.05 else "")

        result_rows.append(row)

    # highlight best (lowest mean)
    df_out = pd.DataFrame(result_rows)
    best_idx = df_out["Mean"].idxmin()
    df_out.loc[best_idx, "Mean"] = f"**{df_out.loc[best_idx, 'Mean']}**"

    # Save table
    out_path = os.path.join(output_dir, f"collision_pct_{surface}.xlsx")
    df_out.to_excel(out_path, index=False)
    print(f"[✔] Saved: {out_path}")
