import pandas as pd
import openap
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import dates
import matplotlib

# Minimal changes: keep original style/structure, only map new CSV columns

matplotlib.rc("font", size=11)
matplotlib.rc("font", family="Ubuntu")
matplotlib.rc("lines", linewidth=2, markersize=8)
matplotlib.rc("grid", color="darkgray", linestyle=":")


def format_ax(ax):
    ax.xaxis.set_major_formatter(dates.DateFormatter("%H:%M"))
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_label_coords(-0.1, 1.03)
    ax.yaxis.label.set_rotation(0)
    ax.yaxis.label.set_ha("left")
    ax.grid()


# Load the CSV exported from main (or your provided CSV)
csv_path = "FAST_Extended_Abstract/data/DAL8952_KPHLtoKDTW_pretracon_optimized.csv"
df = pd.read_csv(csv_path)

# Parse time and alias columns to match previous code
if "recTime" in df.columns:
    df["recTime"] = pd.to_datetime(df["recTime"], format="mixed", errors="coerce")

df_plot = df.rename(
    columns={
        "recTime": "timestamp",
        "alt": "altitude",
        "groundSpeed": "groundspeed",
        "rateOfClimb": "vertical_rate",
    }
)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), sharex=True)

ax1.plot(df_plot.timestamp, df_plot.altitude)
ax2.plot(df_plot.timestamp, df_plot.groundspeed)
ax3.plot(df_plot.timestamp, df_plot.vertical_rate)

ax1.set_ylabel("altitude (ft)")
ax2.set_ylabel("groundspeed (kts)")
ax3.set_ylabel("vertical rate (ft/min)")

for ax in (ax1, ax2, ax3):
    format_ax(ax)

# plt.tight_layout()
# plt.show()

mass_takeoff_assumed = 66300  # kg

fuelflow = openap.FuelFlow("B737")

mass_current = mass_takeoff_assumed

fuelflow_every_step = []
fuel_every_step = []

for i, row in df_plot.iterrows():
    ff = fuelflow.enroute(
        mass=mass_current,
        tas=row.groundspeed,
        alt=row.altitude,
        vs=row.vertical_rate,
    )
    fuel = ff * row.d_ts
    fuelflow_every_step.append(ff)
    fuel_every_step.append(ff * row.d_ts)
    mass_current -= fuel

df_plot = df_plot.assign(fuel_flow=fuelflow_every_step, fuel=fuel_every_step)

plt.figure(figsize=(7, 2))
plt.plot(df_plot.timestamp, df_plot.fuel_flow, color="tab:red")
plt.ylabel("fuel flow (kg/s)")
format_ax(plt.gca())


plt.show()

