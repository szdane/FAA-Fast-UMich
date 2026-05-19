import pandas as pd
import numpy as np

# Load the data from the CSV file
df = pd.read_csv('ac_1sec.csv')

# Define the conversion factor for latitude, which is constant
FEET_PER_DEG_LAT = 69.172 * 5280

# To over-approximate the longitude distance based on current latitudes,
# we find the latitude closest to the equator for each time step.
# This results in the largest possible longitude conversion factor.
df['min_abs_lat'] = np.minimum(np.abs(df['f1_lat']), np.abs(df['f2_lat']))

# Calculate the longitude conversion factor based on the min absolute latitude
df['feet_per_deg_lon'] = FEET_PER_DEG_LAT * np.cos(np.deg2rad(df['min_abs_lat']))

# Calculate the relative trajectories in feet
df['relative_lat_ft'] = (df['f1_lat'] - df['f2_lat']) * FEET_PER_DEG_LAT
df['relative_lon_ft'] = (df['f1_lon'] - df['f2_lon']) * df['feet_per_deg_lon']
df['relative_alt_ft'] = df['f1_alt_ft'] - df['f2_alt_ft']

# Save the results to CSV files
df[['relative_lat_ft', 'relative_lon_ft']].to_csv('relative_lat_lon.csv', index=False)
df[['relative_lat_ft', 'relative_alt_ft']].to_csv('relative_lat_alt.csv', index=False)
df[['relative_lon_ft', 'relative_alt_ft']].to_csv('relative_lon_alt.csv', index=False)