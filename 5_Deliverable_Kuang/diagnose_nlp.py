"""
diagnose_nlp.py
---------------
Reads all nlp_variables_{acId}.csv files from Output/ and produces
diagnostic plots that reveal the cause of NLP trajectory roughness.

Plots per flight:
  1. Heading (psi) vs node  — zigzag = alternating between two values
  2. Heading change (d_psi) per interval — spikes = sharp turns
  3. Cost breakdown per interval (fuel / heading / mach / vs smoothness)
  4. 2D trajectory (x_cart, y_cart) with node numbers labelled

Run from the 5_Deliverable_Kuang folder:
    python diagnose_nlp.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "Output"

csv_files = sorted(OUTPUT_DIR.glob("nlp_variables_*.csv"))
if not csv_files:
    print(f"No nlp_variables_*.csv files found in {OUTPUT_DIR}")
    raise SystemExit

for csv_path in csv_files:
    df = pd.read_csv(csv_path)
    acId = df["flight"].iloc[0]
    n_nodes = len(df)

    # ── Compute summary stats ─────────────────────────────────────────────
    d_psi   = df["d_psi_rad"].dropna()
    mean_dpsi = d_psi.abs().mean()
    max_dpsi  = d_psi.abs().max()
    total_cost_psi  = df["cost_psi"].sum(skipna=True)
    total_fuel_kg   = df["fuel_kg"].sum(skipna=True)
    total_cost_mach = df["cost_mach"].sum(skipna=True)
    total_cost_vs   = df["cost_vs"].sum(skipna=True)
    total_obj       = df["cost_total"].sum(skipna=True)

    print(f"\n{'='*60}")
    print(f"  {acId}")
    print(f"  Nodes           : {n_nodes}")
    print(f"  Mean |d_psi|    : {mean_dpsi:.3f} rad ({np.degrees(mean_dpsi):.1f}°)")
    print(f"  Max  |d_psi|    : {max_dpsi:.3f} rad ({np.degrees(max_dpsi):.1f}°)")
    print(f"  Total fuel      : {total_fuel_kg:.1f} kg")
    print(f"  Total cost_psi  : {total_cost_psi:.0f}  ({100*total_cost_psi/max(total_obj,1):.1f}% of obj)")
    print(f"  Total cost_mach : {total_cost_mach:.1f}")
    print(f"  Total cost_vs   : {total_cost_vs:.1f}")
    print(f"  Diagnosis: ", end="")
    if mean_dpsi > 1.0:
        print("⚠ ZIGZAG — heading changes >57°/interval on average (local minimum from poor init guess)")
    elif mean_dpsi > 0.3:
        print("⚠ ROUGH — moderate heading changes (waypoints pulling trajectory off-path)")
    else:
        print("✓ Smooth")

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(f"NLP Diagnostic — {acId}", fontsize=12, fontweight="bold")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Heading over time
    ax1 = fig.add_subplot(gs[0, 0])
    nodes_with_psi = df["node"][df["psi_deg"].notna()]
    ax1.plot(nodes_with_psi, df["psi_deg"].dropna(), "b-o", ms=3)
    ax1.axhline(0,   color="gray", lw=0.5, ls="--")
    ax1.axhline(180, color="gray", lw=0.5, ls="--")
    ax1.set_xlabel("Node")
    ax1.set_ylabel("Heading (°)")
    ax1.set_title("Heading vs Node\n(should vary smoothly, not alternate)")
    ax1.spines["right"].set_visible(False)
    ax1.spines["top"].set_visible(False)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Heading change per interval
    ax2 = fig.add_subplot(gs[0, 1])
    nodes_dpsi = df["node"][df["d_psi_rad"].notna()]
    dpsi_deg   = np.degrees(df["d_psi_rad"].dropna())
    colors_dpsi = ["red" if abs(v) > 30 else "steelblue" for v in dpsi_deg]
    ax2.bar(nodes_dpsi, dpsi_deg, color=colors_dpsi, alpha=0.8, width=0.8)
    ax2.axhline(0, color="black", lw=0.5)
    ax2.set_xlabel("Interval")
    ax2.set_ylabel("Δheading (°)")
    ax2.set_title("Heading Change per Interval\n(red bars = sharp turns >30°)")
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # 3. Cost breakdown stacked bar
    ax3 = fig.add_subplot(gs[0, 2])
    intervals_with_cost = df.dropna(subset=["cost_total"])
    nodes_c = intervals_with_cost["node"].values
    c_fuel  = intervals_with_cost["fuel_kg"].values
    c_psi   = intervals_with_cost["cost_psi"].values
    c_mach  = intervals_with_cost["cost_mach"].values
    c_vs    = intervals_with_cost["cost_vs"].values
    ax3.bar(nodes_c, c_fuel,  label="Fuel (kg)",           color="steelblue", alpha=0.85)
    ax3.bar(nodes_c, c_psi,   bottom=c_fuel,               label="Cost_psi",  color="tomato",    alpha=0.85)
    ax3.bar(nodes_c, c_mach,  bottom=c_fuel+c_psi,         label="Cost_mach", color="gold",      alpha=0.85)
    ax3.bar(nodes_c, c_vs,    bottom=c_fuel+c_psi+c_mach,  label="Cost_vs",   color="mediumseagreen", alpha=0.85)
    ax3.set_xlabel("Interval")
    ax3.set_ylabel("Cost")
    ax3.set_title("Cost Breakdown per Interval\n(heading penalty should NOT dominate)")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)

    # 4. 2D trajectory
    ax4 = fig.add_subplot(gs[1, :2])
    x_km = df["x_cart_m"] / 1000
    y_km = df["y_cart_m"] / 1000
    ax4.plot(x_km, y_km, "b-", lw=1.5, label="NLP trajectory")
    ax4.scatter(x_km, y_km, c=df["node"], cmap="viridis", s=30, zorder=5)
    # annotate every 5th node
    for _, row in df.iterrows():
        if int(row["node"]) % 5 == 0:
            ax4.text(row["x_cart_m"]/1000, row["y_cart_m"]/1000,
                     f"n{int(row['node'])}", fontsize=7, color="navy",
                     ha="left", va="bottom")
    ax4.plot(x_km.iloc[0],  y_km.iloc[0],  "g^", ms=10, label="Entry")
    ax4.plot(x_km.iloc[-1], y_km.iloc[-1], "rs", ms=10, label="STAR fix")
    ax4.set_xlabel("x (km)")
    ax4.set_ylabel("y (km)")
    ax4.set_title("2D NLP Trajectory (NLP internal frame)\nSmooth = flowing curve; Zigzag = alternating spikes")
    ax4.legend(fontsize=8)
    ax4.set_aspect("equal")
    ax4.grid(True, linestyle=":", alpha=0.4)
    ax4.spines["right"].set_visible(False)
    ax4.spines["top"].set_visible(False)

    # 5. Mach and vertical speed vs node
    ax5 = fig.add_subplot(gs[1, 2])
    ax5_twin = ax5.twinx()
    nodes_ctrl = df["node"][df["mach"].notna()]
    ax5.plot(nodes_ctrl, df["mach"].dropna(),   "b-o", ms=3, label="Mach")
    ax5_twin.plot(nodes_ctrl, df["vs_fpm"].dropna(), "r--o", ms=3, label="VS (fpm)")
    ax5.set_xlabel("Node")
    ax5.set_ylabel("Mach", color="b")
    ax5_twin.set_ylabel("VS (fpm)", color="r")
    ax5.set_title("Mach & Vertical Speed vs Node\n(should be smooth, descent = negative VS)")
    ax5.legend(loc="upper left", fontsize=7)
    ax5_twin.legend(loc="upper right", fontsize=7)
    ax5.spines["top"].set_visible(False)

    fig.tight_layout()
    save_path = OUTPUT_DIR / f"diagnostic_{acId}.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    print(f"  Saved → {save_path}")
    plt.close(fig)

print("\nAll diagnostics saved to Output/")
