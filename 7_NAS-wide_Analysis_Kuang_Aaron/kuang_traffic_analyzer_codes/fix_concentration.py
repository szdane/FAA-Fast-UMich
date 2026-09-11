# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 11:01:57 2026

@author: anomi

Full workflow:
1. Read all CSVs from subfolders
2. Combine into one dataset
3. Generate aircraft list from dataset (optional first run)
4. Load aircraft list from external text file
5. Apply filters
6. Count Orig occurrences
7. Plot results
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

# Path to your ICAO aircraft type list (text file)
actype_file = r"C:\Users\anomi\Documents\Michigan\AERO590\LATTICE\FAST\Misc\part_121_acft_list.txt"

COL_AIRLINE = "Airline"
COL_ACTYPE = "AcType"
COL_DEST = "Dest"
COL_ORIG = "Orig"

ORIG_EXCLUDE = {"BDF", "BEH", "BMI", "CGT", "ERE", "HUL", "JER",
                "MSN", "MUS", "NDE", "OBK", "ORD", "PMM", "RPJ", "UNASSIGNED", "VYS"}

# =========================================================
# STEP 1: LOAD AIRCRAFT TYPES FROM TEXT FILE
# =========================================================

with open(actype_file, "r") as f:
    ACTYPE_KEEP = {
        line.strip().upper()
        for line in f
        if line.strip() and not line.startswith("#")
    }

print(f"Loaded {len(ACTYPE_KEEP)} aircraft types from file.")


# =========================================================
# STEP 2: READ AND COMBINE CSV FILES
# =========================================================

csv_files = list(root_folder.rglob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV files found.")

print(f"Found {len(csv_files)} CSV files.")

dfs = []
for file in csv_files:
    try:
        df = pd.read_csv(file)
        dfs.append(df)
    except Exception as e:
        print(f"Failed to read {file}: {e}")

combined_df = pd.concat(dfs, ignore_index=True)
print("Combined dataset shape:", combined_df.shape)

# Save combined dataset (optional)
combined_df.to_csv(output_folder / "combined_all_csvs.csv", index=False)


# =========================================================
# STEP 3: CLEAN STRING COLUMNS
# =========================================================

for col in [COL_AIRLINE, COL_ACTYPE, COL_DEST, COL_ORIG]:
    combined_df[col] = combined_df[col].astype(str).str.strip().str.upper()


# =========================================================
# STEP 4: APPLY FILTERS
# =========================================================

filtered_df = combined_df.copy()

print("\nInitial rows:", len(filtered_df))

# Remove Airline == "?"
filtered_df = filtered_df[filtered_df[COL_AIRLINE] != "?"]
print("After removing Airline == '?':", len(filtered_df))

# Remove Airline starting with "GA."
filtered_df = filtered_df[
    ~filtered_df[COL_AIRLINE].str.startswith("GA.", na=False)
]
print("After removing Airline starting with 'GA.':", len(filtered_df))

# Filter by aircraft type (from text file)
filtered_df = filtered_df[
    filtered_df[COL_ACTYPE].isin(ACTYPE_KEEP)
]
print("After AcType filtering:", len(filtered_df))

# Keep only Dest == "DTW"
filtered_df = filtered_df[
    filtered_df[COL_DEST] == "ORD"
]
print("After filtering Dest == 'ORD':", len(filtered_df))

# Remove excluded Orig values
filtered_df = filtered_df[
    ~filtered_df[COL_ORIG].isin(ORIG_EXCLUDE)
]
print("After removing excluded Orig values:", len(filtered_df))

# =========================================================
# STEP 5: COUNT UNIQUE ORIG VALUES
# =========================================================

orig_counts = (
    filtered_df[COL_ORIG]
    .value_counts()
    .rename_axis("Orig")
    .reset_index(name="Count")
)

print("\nCounts by Orig:\n")
print(orig_counts)

# Save outputs
orig_counts.to_csv(output_folder / "orig_counts.csv", index=False)
filtered_df.to_csv(output_folder / "filtered_dataset.csv", index=False)


# =========================================================
# STEP 6: PLOT RESULTS
# =========================================================

plt.figure(figsize=(12, 6))
plt.bar(orig_counts["Orig"], orig_counts["Count"])

plt.xlabel("TRACON Entry Fix")
plt.ylabel("Number of Arrivals")
plt.title("Aircraft Arrivals into C90 TRACON across different Entry Fixes")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plot_path = output_folder / "orig_counts_bar_chart.png"
plt.savefig(plot_path, dpi=300)
plt.show()

print(f"\nPlot saved to: {plot_path}")


# =========================================================
# OPTIONAL DEBUG
# =========================================================

missing = set(filtered_df[COL_ACTYPE].unique()) - ACTYPE_KEEP
print("\nAircraft types NOT in your list:", missing)
