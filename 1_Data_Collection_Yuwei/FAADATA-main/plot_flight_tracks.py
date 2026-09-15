import os
import pandas as pd
import matplotlib.pyplot as plt

# Predefined city coordinates
city_coordinates = {
    "KFNT": (42.9654, -83.7436),  # Flint, MI
    "KDTW": (42.2125, -83.3534),  # Detroit, MI
    "KTOL": (41.5868, -83.8078),  # Toledo, OH
    "KCLE": (41.4118, -81.8498),  # Cleveland, OH
    "KPIT": (40.4915, -80.2329),  # Pittsburgh, PA
    "KBUF": (42.9405, -78.7322),  # Buffalo, NY
    "KROC": (43.1189, -77.6724),  # Rochester, NY
    "KCMH": (39.9612, -82.9988)  # Columbus, OH
}

# Function to plot flight tracks
def plot_flight_tracks(df, output_path):
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Get unique aircraft IDs
    unique_ac_ids = df['acId'].unique()
    
    # Plot flight tracks for each aircraft
    for ac_id in unique_ac_ids:
        ac_df = df[df['acId'] == ac_id]
        longitude = ac_df['coord2'].astype(float)  # Assuming coord2 is longitude
        latitude = ac_df['coord1'].astype(float)   # Assuming coord1 is latitude
        ax.plot(longitude, latitude, linewidth=0.5, alpha=0.7)

    # Mark city locations
    for city, (lat, lon) in city_coordinates.items():
        ax.plot(lon, lat, marker='o', color='red', markersize=5)
        ax.text(lon, lat, city, fontsize=9)
    
    # Add grid lines
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', zorder=3)
    
    # Set axis labels
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Set title
    plt.title('Flight Tracks')

    # Save the plot to the specified output path
    plt.savefig(output_path)

# Main function
if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python plot_flight_tracks.py <filtered_rows_path> <output_image_path>")
    else:
        filtered_rows_path = sys.argv[1]
        output_image_path = sys.argv[2]

        # Read filtered_rows.csv
        filtered_rows = pd.read_csv(filtered_rows_path)
        
        # Plot flight tracks and save to the output path
        plot_flight_tracks(filtered_rows, output_image_path)
        print(f"Flight tracks saved to: {output_image_path}")
