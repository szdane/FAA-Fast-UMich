#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Top-5 airlines by flight count for DTW arrivals, aggregated over all
RD_D21_* files in a folder (e.g. RD_D21_20231201_050005_86300_LADDfiltered.csv).

Created on Sat Jul 11 11:50:59 2026

@author: anomi
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FOLDER = Path("/Users/anomi/Documents/LATTICE/FAST/CSV/SHERLOCK_data/2023_TRACON_datasets/C90_Dec_2023/RD_2023_12_01_2024_01_01")
FILE_PATTERN = "RD_C90_*.csv"
OUT_DIR = Path("/Users/anomi/Documents/LATTICE/FAST/Figures/TRACON_airline_distribution/C90")
OUT_PNG = OUT_DIR / "airline_distribution_c90.png"

def load_rd(path):
    """Read an RD csv, with or without a header row."""
    first = pd.read_csv(path, header=None, nrows=1, low_memory=False)
    if str(first.iloc[0, 0]).strip() == "lKey":
        rd = pd.read_csv(path, low_memory=False)
        rd.columns = rd.columns.str.strip()
    else:
        rd = pd.read_csv(path, header=None, names=range(70), low_memory=False)
        rd = rd.rename(columns={12: "AcId", 20: "Airline", 25: "Dest"})
    for col in ("AcId", "Airline", "Dest"):
        rd[col] = rd[col].astype(str).str.strip()
    return rd[["AcId", "Airline", "Dest"]]

# ---- Iterate over all matching files ----
files = sorted(FOLDER.glob(FILE_PATTERN))
if not files:
    raise SystemExit(f"No files matching {FILE_PATTERN} in {FOLDER}")
print(f"Found {len(files)} files matching {FILE_PATTERN}\n")

frames = []
for f in files:
    rd = load_rd(f)
    n_raw = len(rd)
    rd = rd[~rd["Airline"].str.startswith("GA.")]  # drop GA. airlines
    rd = rd[rd["Airline"] != "?"]                  # drop unknown airlines
    rd = rd[rd["Dest"].isin(["MDW", "ORD"])]       # ATL arrivals only
    print(f"{f.name}: {n_raw:,} rows -> {len(rd):,} after filters")
    frames.append(rd)

combined = pd.concat(frames, ignore_index=True)

# ---- Metrics (combined across all files) ----
flights_per_airline = combined["Airline"].value_counts()  # flight counts
total_flights = len(combined)
total_unique_callsigns = combined["AcId"].nunique()
top5 = flights_per_airline.head(5)

print(f"\n=== Combined across {len(files)} files ===")
print(f"Total DTW arrival flights:    {total_flights:,}")
print(f"Total unique callsigns:       {total_unique_callsigns:,}")
print(f"Unique airlines:              {len(flights_per_airline):,}")
print("\nFlight counts per airline (all):")
print(flights_per_airline.to_string())
print("\nTop 5 airlines by flight count:")
print(top5.to_string())

# ---- Bar chart: top 5 only, totals shown as text ----
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(top5.index, top5.values, color="steelblue", edgecolor="black")
ax.bar_label(bars, fmt="%d", padding=3)

ax.set_xlabel("Airline")
ax.set_ylabel("Number of flights")
ax.set_title(
    f"Top 5 airlines by flight count\n"
    f"({len(files)} days combined)"
)
ax.text(
    0.98, 0.95,
    f"Total unique callsigns: {total_unique_callsigns:,}\n"
    f"Total flights: {total_flights:,}",
    transform=ax.transAxes, ha="right", va="top", fontsize=10,
    bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray"),
)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"\nSaved chart to: {OUT_PNG}")
plt.show()
