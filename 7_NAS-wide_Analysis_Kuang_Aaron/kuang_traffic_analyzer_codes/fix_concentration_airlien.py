# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 13:02:42 2026

@author: anomi
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================

root_folder = Path(
    r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\SHERLOCK_data\2023_TRACON_datasets\D21_Dec_2023\RD_2023_12_01_2023_12_31")

output_folder = Path(
    r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\SHERLOCK_data\2023_TRACON_datasets\D21_Dec_2023")
output_folder.mkdir(parents=True, exist_ok=True)

actype_file = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Misc\part_121_acft_list.txt"

COL_AIRLINE = "Airline"
COL_ACTYPE = "AcType"
COL_DEST = "Dest"
COL_ORIG = "Orig"

ORIG_EXCLUDE = {"DA0", "DA1", "DA2", "DA3", "DA4",
                "DA5", "DA6", "DA7", "DA8", "DA9", "YIP", "DTW"}


# =========================================================
# STEP 1: LOAD AIRCRAFT TYPES
# =========================================================

with open(actype_file, "r") as f:
    ACTYPE_KEEP = {
        line.strip().upper()
        for line in f
        if line.strip() and not line.startswith("#")
    }

print(f"Loaded {len(ACTYPE_KEEP)} aircraft types.")


# =========================================================
# STEP 2: LOAD AND COMBINE CSV FILES
# =========================================================

csv_files = list(root_folder.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found.")

dfs = []
for file in csv_files:
    try:
        dfs.append(pd.read_csv(file))
    except Exception as e:
        print(f"Failed to read {file}: {e}")

if not dfs:
    raise ValueError("No CSV files could be read successfully.")

combined_df = pd.concat(dfs, ignore_index=True)
print("Combined dataset shape:", combined_df.shape)


# =========================================================
# STEP 3: CLEAN DATA
# =========================================================

for col in [COL_AIRLINE, COL_ACTYPE, COL_DEST, COL_ORIG]:
    combined_df[col] = combined_df[col].astype(str).str.strip().str.upper()


# =========================================================
# STEP 4: APPLY FILTERS
# =========================================================

filtered_df = combined_df.copy()

filtered_df = filtered_df[filtered_df[COL_AIRLINE] != "?"]
filtered_df = filtered_df[~filtered_df[COL_AIRLINE].str.startswith(
    "GA.", na=False)]
filtered_df = filtered_df[filtered_df[COL_ACTYPE].isin(ACTYPE_KEEP)]
filtered_df = filtered_df[filtered_df[COL_DEST] == "DTW"]
filtered_df = filtered_df[~filtered_df[COL_ORIG].isin(ORIG_EXCLUDE)]

print("Rows after filtering:", len(filtered_df))


# =========================================================
# STEP 5: GROUP BY ORIG + AIRLINE
# =========================================================

fix_airline_pivot = (
    filtered_df
    .groupby([COL_ORIG, COL_AIRLINE])
    .size()
    .unstack(fill_value=0)
)

# Sort fixes by total traffic
fix_airline_pivot = fix_airline_pivot.loc[
    fix_airline_pivot.sum(axis=1).sort_values(ascending=False).index
]

# Keep only top 10 airlines overall
top_n = 10
top_airlines = fix_airline_pivot.sum(axis=0).nlargest(top_n).index

top_df = fix_airline_pivot[top_airlines]
other_df = fix_airline_pivot.drop(columns=top_airlines)

top_df = top_df.copy()
top_df["OTHER"] = other_df.sum(axis=1)

fix_airline_pivot = top_df

# Save counts table
fix_airline_pivot.to_csv(output_folder / "fix_airline_counts.csv")


# =========================================================
# STEP 6: CONVERT TO PERCENT
# =========================================================

fix_airline_percent = fix_airline_pivot.div(
    fix_airline_pivot.sum(axis=1).replace(0, 1),
    axis=0
) * 100

fix_airline_percent.to_csv(output_folder / "fix_airline_percent.csv")


# =========================================================
# STEP 7: STACKED BAR CHART (COUNTS)
# =========================================================

fig, ax = plt.subplots(figsize=(22, 10))

fix_airline_pivot.plot(
    kind="bar",
    stacked=True,
    width=0.85,
    ax=ax
)

ax.set_xlabel("TRACON Entry Fix Name", fontsize=14)
ax.set_ylabel("Number of Arrivals", fontsize=14)
ax.set_title("Airline Usage by Entry Fix (D21 TRACON)", fontsize=16)

ax.tick_params(axis="x", rotation=45)

ax.legend(
    title="Airline",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()
plt.savefig(output_folder / "fix_airline_counts.png",
            dpi=300, bbox_inches="tight")
plt.show()


# =========================================================
# STEP 8: STACKED BAR CHART (PERCENT)
# =========================================================

fig, ax = plt.subplots(figsize=(22, 10))

fix_airline_percent.plot(
    kind="bar",
    stacked=True,
    width=0.85,
    ax=ax
)

ax.set_xlabel("TRACON Entry Fix Name", fontsize=14)
ax.set_ylabel("Percent of Arrivals", fontsize=14)
ax.set_title(
    "Airline Usage % Distribution by Entry Fix (D21 TRACON)", fontsize=16)

ax.tick_params(axis="x", rotation=45)

ax.legend(
    title="Airline",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()
plt.savefig(output_folder / "fix_airline_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()
