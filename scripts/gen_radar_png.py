"""Genera un PNG estático del radar multidimensional (Top-5 modelos) para el deck pptx.

Reproduce con matplotlib la gráfica Plotly de `slides/index.qmd` (RMSE normalizado por
dimensión IMU, 0-1, donde 1 = mejor) a partir de `notebooks/evaluation_results.pkl`.
Salida: `assets/radar_multidim.png` a 300 DPI. No requiere kaleido.
"""
from pathlib import Path
import pickle
from math import pi

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
PKL = ROOT / "notebooks" / "evaluation_results.pkl"
OUT = ROOT / "assets" / "radar_multidim.png"

with open(PKL, "rb") as f:
    data = pickle.load(f)

all_results = data["results"]
df_metrics = data["df_metrics"]
IMU_NAMES = list(data["IMU_NAMES"])
MODEL_COLORS = data["MODEL_COLORS"]

# RMSE normalizado por dimensión (0-1); 1 = mejor (menor RMSE)
normalized_rmse = {}
for dim_name in IMU_NAMES:
    dim_rmses = [all_results[m]["metrics"]["per_dimension"][dim_name]["RMSE"]
                 for m in df_metrics["Modelo"]]
    max_rmse, min_rmse = max(dim_rmses), min(dim_rmses)
    for m in df_metrics["Modelo"]:
        rmse = all_results[m]["metrics"]["per_dimension"][dim_name]["RMSE"]
        norm_val = 1 - ((rmse - min_rmse) / (max_rmse - min_rmse + 1e-8))
        normalized_rmse.setdefault(m, []).append(norm_val)

top_5_models = df_metrics.head(5)["Modelo"].tolist()

# Ángulos del radar (cerrado)
n_vars = len(IMU_NAMES)
angles = [i / n_vars * 2 * pi for i in range(n_vars)]
angles += angles[:1]

plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
fig, ax = plt.subplots(figsize=(8, 7), subplot_kw=dict(polar=True))

for model_name in top_5_models:
    values = normalized_rmse[model_name] + normalized_rmse[model_name][:1]
    color = MODEL_COLORS.get(model_name, "#333333")
    ax.plot(angles, values, "o-", linewidth=2, markersize=5, color=color, label=model_name)
    ax.fill(angles, values, color=color, alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(IMU_NAMES, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8, color="#555")
ax.set_title("Comparación Multidimensional — Top 5 Modelos", fontsize=14, pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.10), fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig(OUT, bbox_inches="tight")
print(f"✅ Radar guardado en {OUT}")
