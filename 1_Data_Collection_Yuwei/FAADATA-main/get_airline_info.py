import pandas as pd
import os
import sys

def get_airline_info(airline_id, input_dir):
    # Define file paths
    merged_data_path = os.path.join(input_dir, 'merged_unique_data.csv')

    # Read the merged data
    merged_data = pd.read_csv(merged_data_path)

    # Filter the data based on the airline ID
    airline_info = merged_data[merged_data['acId'] == airline_id]

    # Check if the airline info is found
    if not airline_info.empty:
        departure = airline_info['ORIGIN'].values[0]
        arrival = airline_info['DEST'].values[0]
    else:
        departure = 'NaN'
        arrival = 'NaN'

    return departure, arrival

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python get_airline_info.py <airline_id> <input_dir>")
    else:
        airline_id = sys.argv[1]
        input_dir = sys.argv[2]
        departure, arrival = get_airline_info(airline_id, input_dir)
        print(f"Departure: {departure}")
        print(f"Arrival: {arrival}")
