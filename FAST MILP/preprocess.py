"""
Filter the historical trajectory of an intruder flight in every 30s
"""
import pandas as pd

df = pd.read_csv("filtered_rows.csv")  

df['recTime'] = pd.to_datetime(df['recTime'], format='mixed', errors='raise')  

# Filter by acID
df = df[df['acId'] == 'SWA3255_KDENtoKDTW']

# Define time window
start_time = pd.Timestamp('2023-12-22 00:30:01.000')
end_time = pd.Timestamp('2023-12-22 00:45:28.000')

# Filter rows within the time range
df_filtered = df[(df['recTime'] >= start_time) & (df['recTime'] <= end_time)]

# Resample to every 30 seconds using nearest timestamp
df_resampled = df_filtered.set_index('recTime').resample('30S').nearest().reset_index()

# df_resampled = df_resampled[(df_resampled['recTime'] >= start_time) & (df_resampled['recTime'] <= end_time)]
df_result = pd.DataFrame()
df_result['t'] = range(0, len(df_resampled)*30, 30)
df_result['x'] = df_resampled['coord1']
df_result['y'] = df_resampled['coord2']
df_result['z'] = df_resampled['alt']

# Keep only t, x, y, z
final_df = df_result[['t', 'x', 'y', 'z']]
# Output
print(df_result)
df_result.to_csv("filtered_flight_den.csv", index=False)
