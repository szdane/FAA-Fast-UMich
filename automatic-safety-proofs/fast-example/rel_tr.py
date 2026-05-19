import pandas as pd
import numpy as np


FEET_PER_DEG_LAT = 364000

def calculate_relative_trajectory(input_file, output_file):
    """
    Calculates the relative trajectory between two objects from a CSV file.

    Args:
        input_file (str): The path to the input CSV file.
        output_file (str): The path to the output CSV file for the relative trajectory.
    """
    try:
        # 1. Read the CSV file into a pandas DataFrame.
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")
        return

    # Ensure the required columns exist
    required_columns = ['t', 'f1_lat', 'f1_lon', 'f1_alt_ft', 'f2_lat', 'f2_lon', 'f2_alt_ft']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The CSV file must contain the following columns: {required_columns}")
        return
    df['min_abs_lat'] = np.minimum(np.abs(df['f1_lat']), np.abs(df['f2_lat']))
    df['feet_per_deg_lon'] = FEET_PER_DEG_LAT * np.cos(np.deg2rad(df['min_abs_lat']))
    # 2. Select the coordinate columns for each object.
    f1_coords = df[['f1_lat', 'f1_lon', 'f1_alt_ft']].values
    f2_coords = df[['f2_lat', 'f2_lon', 'f2_alt_ft']].values
    time_points = df['t']
    relative_df = pd.DataFrame()
    relative_df['relative_lat_ft'] = ((df['f1_lat'] - df['f2_lat']) * FEET_PER_DEG_LAT)/500
    relative_df['relative_lon_ft'] = ((df['f1_lon'] - df['f2_lon']) * df['feet_per_deg_lon'])/500
    relative_df['relative_alt_ft'] = (df['f1_alt_ft'] - df['f2_alt_ft'])/500

    # 3. Calculate the relative position by subtracting the coordinates.
    # The result is a new array where each row represents the relative position (f2 - f1).
    relative_trajectory_coords = f2_coords - f1_coords

    # Create a new DataFrame for the relative trajectory
    # relative_df = pd.DataFrame(relative_trajectory_coords, columns=['relative_lat', 'relative_lon', 'relative_alt_ft'])
    relative_df.insert(0, 't', time_points)

    # 4. Save the relative trajectory to a new CSV file.
    relative_df.to_csv(output_file, index=False)
    print(f"Relative trajectory calculated and saved to '{output_file}'")

# --- Example Usage ---
if __name__ == "__main__":
    # Create a dummy input CSV file for demonstration
    # dummy_data = {
    #     't': [0, 1, 2, 3],
    #     'f1_lat': [34.05, 34.06, 34.07, 34.08],
    #     'f1_lon': [-118.25, -118.26, -118.27, -118.28],
    #     'f1_alt_ft': [1000, 1050, 1100, 1150],
    #     'f2_lat': [34.051, 34.062, 34.073, 34.084],
    #     'f2_lon': [-118.252, -118.264, -118.276, -118.288],
    #     'f2_alt_ft': [1100, 1150, 1200, 1250]
    # }
    # dummy_df = pd.DataFrame(dummy_data)
    # dummy_df.to_csv('input_trajectories.csv', index=False)

    # print("Created dummy input file: 'input_trajectories.csv'")



    # Run the function
    calculate_relative_trajectory('ac_2.csv', 'relative_trajectories.csv')
