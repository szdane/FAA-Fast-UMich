import os
import pandas as pd
from shapely.geometry import Polygon as ShapelyPolygon, Point
from collections import defaultdict

# Define regions with their coordinates
regions_coordinates = [
    {  # Region: PECK
        "name": "PECK",
        "coords": [(-85.03859, 43.58688), (-82.22609, 43.97555), (-82.1382, 43.59484), (-82.44582, 42.99513), (-83.37966, 42.6971), (-85.07155, 42.76166)]
    },
    {  # Region: DETROIT
        "name": "DETROIT",
        "coords": [(-85.07155, 42.76166), (-83.37966, 42.6971), (-82.7864, 41.87624), (-84.66506, 41.85169), (-84.98366, 42.03147)]
    },
    {  # Region: BLUFFTON
        "name": "BLUFFTON",
        "coords": [(-84.66506, 41.85169), (-82.7864, 41.87624), (-82.66555, 41.17713), (-83.43459, 40.7374), (-84.73098, 40.94519)]
    },
    {  # Region: HUDSON
        "name": "HUDSON",
        "coords": [(-82.44582, 42.99513), (-83.37966, 42.6971), (-82.7864, 41.87624), (-81.28127, 41.83532), (-80.90773, 42.51111), (-80.49025, 42.83421)]
    },
    {  # Region: RAVENNA
        "name": "RAVENNA",
        "coords": [(-82.7864, 41.87624), (-81.28127, 41.83532), (-80.55617, 41.69602), (-80.7759, 41.10267), (-80.27053, 40.66244), (-80.76491, 40.69577), (-81.63283, 40.58739), (-83.43459, 40.7374), (-82.66555, 41.17713)]
    },
    {  # Region: NIAGARA
        "name": "NIAGARA",
        "coords": [(-80.90773, 42.51111), (-80.49025, 42.83421), (-79.35866, 43.05938), (-78.79836, 42.9469), (-77.04055, 42.49491), (-77.3152, 42.04779), (-79.14992, 42.4544)]
    },
    {  # Region: JAMESTOWN
        "name": "JAMESTOWN",
        "coords": [(-80.90773, 42.51111), (-81.28127, 41.83532), (-80.55617, 41.69602), (-77.67775, 41.46591), (-77.3152, 42.04779), (-79.14992, 42.4544)]
    },
    {  # Region: CLARKSBURG
        "name": "CLARKSBURG",
        "coords": [(-80.55617, 41.69602), (-77.67775, 41.46591), (-77.93044, 41.02812), (-77.76564, 40.64577), (-80.27053, 40.66244), (-80.7759, 41.10267)]
    },
    {  # Region: FRANKLIN
        "name": "FRANKLIN",
        "coords": [(-81.63283, 40.58739), (-80.76491, 40.69577), (-78.40285, 39.67181), (-80.2925, 39.11997), (-81.38015, 39.82386)]
    },
    {  # Region: ALLEGHENY
        "name": "ALLEGHENY",
        "coords": [(-80.76491, 40.69577), (-80.27053, 40.66244), (-77.76564, 40.64577), (-77.58986, 40.05971), (-78.40285, 39.67181)]
    },
    {  # Region: GENESEO
        "name": "GENESEO",
        "coords": [(-79.35866, 43.05938), (-78.79836, 42.9469), (-77.04055, 42.49491), (-76.15065, 42.65672), (-76.87575, 43.84099), (-78.77639, 43.7617)]
    }
]

# Create polygons for each region
regions_polygons = []
for region in regions_coordinates:
    coords = region["coords"]
    name = region["name"]
    polygon = ShapelyPolygon(coords)
    regions_polygons.append((name, polygon))

def analyze_flight_regions(filtered_rows_path, regions_coordinates, output_dir):
    # Load filtered rows
    filtered_rows = pd.read_csv(filtered_rows_path)
    filtered_rows['recTime'] = pd.to_datetime(filtered_rows['recTime'])

    # Dictionary to store flight region times
    flight_region_times = defaultdict(list)

    # Get unique aircraft IDs
    unique_ac_ids = filtered_rows['acId'].unique()
    for ac_id in unique_ac_ids:
        ac_df = filtered_rows[filtered_rows['acId'] == ac_id]
        ac_df = ac_df.sort_values(by='recTime')  # Sort by time
        longitude = ac_df['coord2'].astype(float)  # Assuming coord2 is longitude
        latitude = ac_df['coord1'].astype(float)   # Assuming coord1 is latitude
        times = ac_df['recTime']

        for i in range(len(ac_df)):
            point = Point(longitude.iloc[i], latitude.iloc[i])
            for region_name, polygon in regions_polygons:
                if polygon.contains(point):
                    flight_region_times[ac_id].append((region_name, times.iloc[i]))

    # Prepare records for DataFrame
    records = []
    for ac_id, regions_times in flight_region_times.items():
        for i in range(len(regions_times) - 1):
            region_name, entry_time = regions_times[i]
            _, exit_time = regions_times[i + 1]
            duration = exit_time - entry_time
            records.append((ac_id, region_name, duration))

    # Create DataFrame from records
    df = pd.DataFrame(records, columns=['Aircraft ID', 'Location', 'Duration'])

    # Summarize data by Aircraft ID and Location
    summary_df = df.groupby(['Aircraft ID', 'Location'], as_index=False).sum()

    # Create organized flights DataFrame
    result = pd.DataFrame(columns=['Aircraft ID', 'Locations', 'Durations'])
    for ac_id in summary_df['Aircraft ID'].unique():
        ac_data = summary_df[summary_df['Aircraft ID'] == ac_id]
        locations = ac_data['Location'].tolist()
        durations = ac_data['Duration'].astype(str).tolist()
        result = result.append({'Aircraft ID': ac_id, 'Locations': locations, 'Durations': durations}, ignore_index=True)

    # Save the organized flights DataFrame to a CSV file
    organized_flights_path = os.path.join(output_dir, 'organized_flights.csv')
    result.to_csv(organized_flights_path, index=False)

    return organized_flights_path

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python flight_region.py <filtered_rows_path> <output_dir>")
    else:
        filtered_rows_path = sys.argv[1]
        output_dir = sys.argv[2]

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Analyze flight regions and save the results
        organized_flights_path = analyze_flight_regions(filtered_rows_path, regions_coordinates, output_dir)
        print(f"Organized flights saved to: {organized_flights_path}")
