import pandas as pd
import numpy as np

def generate_piecewise_function(df, x_col, y_col):
    """
    Generates a string representation of a piecewise linear function.
    Handles zero division by treating vertical segments as constant functions.
    Merges segments with similar slopes.
    """
    if len(df) < 2:
        return "Not enough data points to create a piecewise function."

    # Remove consecutive duplicate points to simplify the data
    df_filtered = df.drop_duplicates(subset=[x_col, y_col]).reset_index(drop=True)
    # print(df_filtered)
    df_filtered = df_filtered.sort_values(by = x_col, ignore_index = True)
    # print(df_filtered)
    merged_segments = []

    # Handle the very first point
    x1, y1 = df_filtered.loc[0, x_col], df_filtered.loc[0, y_col]
    print(x1,y1)

    for i in range(1, len(df_filtered)):
        x2, y2 = df_filtered.loc[i, x_col], df_filtered.loc[i, y_col]

        dx = x2 - x1
        dy = y2 - y1

        # Check for vertical segments (x is constant) and handle as a special case
        if np.isclose(dx, 0, atol=1e-9):
            slope = 0.0
            intercept = y1
        else:
            slope = dy / dx
            intercept = y1 - slope * x1

        current_segment = {
            'slope': slope,
            'intercept': intercept,
            'start_x': x1,
            'end_x': x2
        }

        if not merged_segments:
            merged_segments.append(current_segment)
        else:
            last_segment = merged_segments[-1]
            # Check if slopes and intercepts are approximately the same for merging
            if np.isclose(current_segment['slope'], last_segment['slope'], atol=1e-6) and np.isclose(current_segment['intercept'], last_segment['intercept'], atol=1.0):
                last_segment['end_x'] = x2
            else:
                merged_segments.append(current_segment)

        x1, y1 = x2, y2

    # Format the output string
    pieces_str = []
    for i, seg in enumerate(merged_segments):
        start_x = seg['start_x']
        end_x = seg['end_x']
        slope = seg['slope']
        intercept = seg['intercept']
        
        func_str = f"{slope}*x + {intercept}"
        if i == 0:
            condition = f"x <= {end_x}"
        else:
            condition = f"(x > {start_x}) & (x <= {end_x})"
        
        pieces_str.append(f"({func_str}, {condition})")
    
    return f"Piecewise({', '.join(pieces_str)})"

# Load data from the CSV files
df_lat_lon = pd.read_csv('relative_lat_lon.csv')
df_lat_alt = pd.read_csv('relative_lat_alt.csv')
df_lon_alt = pd.read_csv('relative_lon_alt.csv')

# Generate and print the piecewise functions
print("--- Relative Latitude vs. Longitude ---")
print(generate_piecewise_function(df_lat_lon, 'relative_lon_ft', 'relative_lat_ft'))
print("\n--- Relative Latitude vs. Altitude ---")
print(generate_piecewise_function(df_lat_alt, 'relative_lat_ft', 'relative_alt_ft'))
print("\n--- Relative Longitude vs. Altitude ---")
print(generate_piecewise_function(df_lon_alt, 'relative_lon_ft', 'relative_alt_ft'))