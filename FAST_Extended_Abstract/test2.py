from openap import FuelFlow, Emission
import numpy as np

fuelflow = FuelFlow(ac="A320")
emission = Emission(ac="A320")

TAS = 350
ALT = 30000

FF = fuelflow.enroute(mass=60000, tas=TAS, alt=ALT, vs=0)  # kg/s

CO2 = emission.co2(FF)  # g/s
H2O = emission.h2o(FF)  # g/s
NOx = emission.nox(FF, tas=TAS, alt=ALT)  # g/s
CO = emission.co(FF, tas=TAS, alt=ALT)  # g/s
HC = emission.hc(FF, tas=TAS, alt=ALT)  # g/s

import pandas as pd
import openap
import matplotlib.pyplot as plt

mass_takeoff_assumed = 66300  # kg

fuelflow = openap.FuelFlow("b752")

# Load the data
df = pd.read_csv(
    "FAST_Extended_Abstract/data/flight_a319_opensky.csv",
    parse_dates=["timestamp"],
    dtype={"icao24": str},
)

# Calculate seconds between each timestamp
df = df.assign(d_ts=lambda d: d.timestamp.diff().dt.total_seconds().bfill())

from matplotlib import dates

import matplotlib

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


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5, 5), sharex=True)

ax1.plot(df.timestamp, df.altitude)
ax2.plot(df.timestamp, df.groundspeed)
ax3.plot(df.timestamp, df.vertical_rate)

ax1.set_ylabel("altitude (ft)")
ax2.set_ylabel("groundspeed (kts)")
ax3.set_ylabel("vertical rate (ft/min)")

for ax in (ax1, ax2, ax3):
    format_ax(ax)

mass_current = mass_takeoff_assumed

fuelflow_every_step = []
fuel_every_step = []
gamma_every_step = []
thrust_every_step = []
ratio_every_step = []

for i, row in df.iterrows():
    ff,  gamma, T, ratio = fuelflow.enroute(
        mass=mass_current,
        tas=row.groundspeed,
        alt=row.altitude,
        vs=row.vertical_rate,
    )
    fuel = ff * row.d_ts
    fuelflow_every_step.append(ff)
    fuel_every_step.append(ff * row.d_ts)
    mass_current -= fuel
    gamma_every_step.append(np.degrees(gamma))
    thrust_every_step.append(T)
    # Thrust-to-weight ratio using current mass before update (N / (kg*m/s^2))
    ratio_every_step.append(ratio)

df = df.assign(
    fuel_flow=fuelflow_every_step,
    fuel=fuel_every_step,
    gamma=gamma_every_step,
    thrust=thrust_every_step,
    ratio=ratio_every_step,
)

# Fuel flow figure
fig1 = plt.figure(figsize=(7, 2))
ax_fuel = fig1.add_subplot(111)
ax_fuel.plot(df.timestamp, df.fuel_flow, color="tab:red")
ax_fuel.set_ylabel("fuel flow (kg/s)")
format_ax(ax_fuel)

# Gamma and thrust in a separate figure
fig2, (ax_g, ax_t) = plt.subplots(2, 1, figsize=(7, 4), sharex=True)
ax_g.plot(df.timestamp, df.gamma, color="tab:blue")
ax_g.set_ylabel("gamma (deg)")
format_ax(ax_g)

ax_t.plot(df.timestamp, df.thrust, color="tab:orange")
ax_t.set_ylabel("thrust (N)")
format_ax(ax_t)

# Ratio (T/W) in a separate figure
fig3 = plt.figure(figsize=(7, 2))
ax_ratio = fig3.add_subplot(111)
ax_ratio.plot(df.timestamp, df["ratio"], color="tab:brown")
ax_ratio.set_ylabel("Throttle ratio")
format_ax(ax_ratio)

plt.show()