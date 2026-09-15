import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

# Read data from files
def read_data(input_dir, output_dir):
    file_path_ua = os.path.join(input_dir, 'T_ONTIME_MARKETING-8.csv')
    data_ua = pd.read_csv(file_path_ua)
    
    filtered_rows_path = os.path.join(output_dir, 'filtered_rows.csv')
    filtered_rows = pd.read_csv(filtered_rows_path)
    
    icao_iata_mapping_path = os.path.join(input_dir, 'icao_iata_codes.csv')
    icao_iata_mapping = pd.read_csv(icao_iata_mapping_path)
    
    return data_ua, filtered_rows, icao_iata_mapping

# Merge data
def merge_data(filtered_rows, data_ua, icao_iata_mapping):
    # Extract unique acId
    unique_acid = filtered_rows['acId'].unique()
    
    # Create a new DataFrame with unique acId
    unique_rows = pd.DataFrame(unique_acid, columns=['acId'])
    
    # Extract Carrier and FlightNumber from acId
    unique_rows['Carrier'] = unique_rows['acId'].apply(lambda x: ''.join(filter(str.isalpha, x)))
    unique_rows['FlightNumber'] = unique_rows['acId'].apply(lambda x: ''.join(filter(str.isdigit, x)))
    
    # Ensure FlightNumber column is of string type
    unique_rows['FlightNumber'] = unique_rows['FlightNumber'].astype(str)
    
    # Create a dictionary for ICAO to IATA mapping from CSV file
    icao_to_iata_dict_csv = icao_iata_mapping.set_index('ICAO')['IATA'].to_dict()
    
    # Function to get IATA code from Carrier
    def get_iata_code(carrier):
        if carrier in icao_to_iata_dict_csv:
            return icao_to_iata_dict_csv[carrier]
        else:
            return carrier[:-1]

    unique_rows['IATA'] = unique_rows['Carrier'].apply(get_iata_code)
    
    # Save unique_rows to a new CSV file
    unique_csv_path = os.path.join(output_dir, 'unique.csv')
    unique_rows.to_csv(unique_csv_path, index=False)
    
    # Select required columns from data_ua
    data_ua_selected = data_ua[['MKT_CARRIER', 'MKT_CARRIER_FL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME', 'DEST', 'DEST_CITY_NAME']]
    data_ua_selected = data_ua_selected.rename(columns={
        'MKT_CARRIER': 'IATA',
        'MKT_CARRIER_FL_NUM': 'FlightNumber'
    })
    
    # Ensure FlightNumber column is of string type
    data_ua_selected['FlightNumber'] = data_ua_selected['FlightNumber'].astype(str)
    
    # Merge unique_rows with data_ua_selected
    merged_data = pd.merge(unique_rows, data_ua_selected, on=['IATA', 'FlightNumber'], how='left')
    
    # Remove duplicate rows
    merged_data = merged_data.drop_duplicates()
    
    return merged_data

# Update filtered data with merged data
def update_filtered_rows(filtered_rows, merged_data):
    merged_filtered_rows = pd.merge(filtered_rows, merged_data[['acId', 'DEST']], on='acId', how='left')
    return merged_filtered_rows

# Plot flight tracks
def plot_flight_tracks(df, output_dir):
    unique_ac_ids = df['acId'].unique()
    
    # Calculate the occurrence of each destination
    top_dests = df['DEST'].value_counts().nlargest(5).index.tolist()
    
    # Assign colors to each destination
    vibrant_colors = [
        '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231'
    ]
    
    dest_colors = {dest: color for dest, color in zip(top_dests, vibrant_colors)}
    light_gray = '#d3d3d3'
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot tracks without top destinations first
    for ac_id in unique_ac_ids:
        ac_df = df[df['acId'] == ac_id]
        if pd.isna(ac_df['DEST'].iloc[0]) or ac_df['DEST'].iloc[0] not in top_dests:
            longitude = ac_df['coord2'].astype(float)
            latitude = ac_df['coord1'].astype(float)
            ax.plot(longitude, latitude, color=light_gray, linewidth=0.5, alpha=0.7, zorder=1)
    
    # Plot tracks with top destinations
    for ac_id in unique_ac_ids:
        ac_df = df[df['acId'] == ac_id]
        if not pd.isna(ac_df['DEST'].iloc[0]) and ac_df['DEST'].iloc[0] in top_dests:
            longitude = ac_df['coord2'].astype(float)
            latitude = ac_df['coord1'].astype(float)
            color = dest_colors.get(ac_df['DEST'].iloc[0], light_gray)
            ax.plot(longitude, latitude, color=color, linewidth=0.5, alpha=0.7, zorder=2)
    
    # Add city coordinates
    city_coordinates = {
        "KFNT": (42.9654, -83.7436),  # Flint, MI
        "KDTW": (42.2125, -83.3534),  # Detroit, MI
        "KTOL": (41.5868, -83.8078),  # Toledo, OH
        "KCLE": (41.4118, -81.8498),  # Cleveland, OH
        "KPIT": (40.4915, -80.2329),  # Pittsburgh, PA
        "KBUF": (42.9405, -78.7322),  # Buffalo, NY
        "KROC": (43.1189, -77.6724),  # Rochester, NY
        "KCMH": (39.9612, -82.9988)   # Columbus, OH
    }
    
    for city, (lat, lon) in city_coordinates.items():
        ax.plot(lon, lat, marker='o', color='red', markersize=5)
        ax.text(lon, lat, city, fontsize=9)
    
    # Add legend for top destinations
    handles = [plt.Line2D([0], [0], color=color, linewidth=3) for color in dest_colors.values()]
    labels = [dest for dest in dest_colors.keys() if dest_colors[dest] != light_gray]
    ax.legend(handles, labels, title='Top Destinations', loc='upper right')
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', zorder=3)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.title('Flight Tracks')

    # Save the plot
    output_image_path = os.path.join(output_dir, 'arrival_airports.png')
    plt.savefig(output_image_path)

def main(input_dir, output_dir):
    data_ua, filtered_rows, icao_iata_mapping = read_data(input_dir, output_dir)
    merged_data = merge_data(filtered_rows, data_ua, icao_iata_mapping)
    merged_filtered_rows = update_filtered_rows(filtered_rows, merged_data)
    plot_flight_tracks(merged_filtered_rows, output_dir)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python plot_arrival_airports.py <input_dir> <output_dir>")
    else:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        main(input_dir, output_dir)
