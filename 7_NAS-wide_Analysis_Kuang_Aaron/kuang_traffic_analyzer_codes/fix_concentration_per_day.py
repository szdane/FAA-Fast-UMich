# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:06:57 2026

@author: anomi
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# USER SETTINGS
# =========================================================

root_folder = Path(
    r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\SHERLOCK_data\2023_TRACON_datasets\C90_Dec_2023\RD_2023_12_01_2023_12_31")

output_folder = Path(
    r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\CSV\SHERLOCK_data\2023_TRACON_datasets\C90_Dec_2023")
output_folder.mkdir(parents=True, exist_ok=True)

actype_file = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Misc\part_121_acft_list.txt"

COL_AIRLINE = "Airline"
COL_ACTYPE = "AcType"
COL_DEST = "Dest"
COL_ORIG = "Orig"
COL_START_DATE = "StartDate"
COL_START_TIME = "StartTime"

ORIG_EXCLUDE = {"BDF", "BEH", "BMI", "CGT", "ERE", "HUL", "JER",
                "MSN", "MUS", "NDE", "OBK", "ORD", "PMM", "RPJ", "UNASSIGNED", "VYS"}


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
filtered_df = filtered_df[filtered_df[COL_DEST] == "ORD"]
filtered_df = filtered_df[~filtered_df[COL_ORIG].isin(ORIG_EXCLUDE)]

print("Rows after filtering:", len(filtered_df))


# =========================================================
# STEP 5: CREATE DATETIME + HOUR
# =========================================================

filtered_df["StartDateTime"] = pd.to_datetime(
    filtered_df[COL_START_DATE] + " " + filtered_df[COL_START_TIME],
    format="%m/%d/%Y %H:%M:%S",
    errors="coerce"
)

filtered_df = filtered_df.dropna(subset=["StartDateTime"])
filtered_df["Hour"] = filtered_df["StartDateTime"].dt.hour


# =========================================================
# STEP 6: GROUP BY HOUR + ORIG
# =========================================================

hourly_orig_pivot = (
    filtered_df
    .groupby(["Hour", COL_ORIG])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

# Ensure all 24 hours are present
hourly_orig_pivot = hourly_orig_pivot.reindex(range(24), fill_value=0)

hourly_orig_pivot.to_csv(output_folder / "orig_counts_by_hour.csv")


# =========================================================
# STACKED BAR CHART (COUNTS) — IMPROVED READABILITY
# =========================================================

# Increase figure width significantly
fig, ax = plt.subplots(figsize=(19, 10))

hourly_orig_pivot.plot(
    kind="bar",
    stacked=True,
    width=0.5,
    ax=ax
)

ax.set_xlabel("Hour of Day (UTC Time), Local Time is UTC-6", fontsize=14)
ax.set_ylabel("Number of Arrivals", fontsize=14)
ax.set_title("C90 TRACON Entry Fix Distribution by Hour", fontsize=16)

# Improve x-axis readability
ax.set_xticks(range(24))
ax.set_xticklabels(range(24), rotation=0)

# Move legend OUTSIDE plot
ax.legend(
    title="Orig",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()

plt.savefig(output_folder / "stacked_counts_by_hour.png",
            dpi=300, bbox_inches="tight")
plt.show()
# =========================================================
# STACKED BAR CHART (PERCENT) — IMPROVED READABILITY
# =========================================================

# Convert counts → percent by hour
hourly_percent = hourly_orig_pivot.div(
    hourly_orig_pivot.sum(axis=1).replace(0, 1),
    axis=0
) * 100

fig, ax = plt.subplots(figsize=(19, 10))

hourly_percent.plot(
    kind="bar",
    stacked=True,
    width=0.5,
    ax=ax
)

ax.set_xlabel("Hour of Day (UTC Time), Local Time is UTC-6", fontsize=14)
ax.set_ylabel("Percent of Arrivals", fontsize=14)
ax.set_title("C90 TRACON Entry Fix Distribution by Hour (Percent)", fontsize=16)

ax.set_xticks(range(24))
ax.set_xticklabels(range(24), rotation=0)

ax.legend(
    title="Orig",
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=9,
    title_fontsize=10
)

plt.tight_layout()

plt.savefig(output_folder / "stacked_percent_by_hour.png",
            dpi=300, bbox_inches="tight")
plt.show()
