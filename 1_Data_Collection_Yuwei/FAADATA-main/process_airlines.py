import pandas as pd
import os

def process_airlines_data(file_path_ua, filtered_rows_path, icao_iata_mapping_path, output_dir):
    # Read CSV files
    data_ua = pd.read_csv(file_path_ua)
    filtered_rows = pd.read_csv(filtered_rows_path)
    icao_iata_mapping = pd.read_csv(icao_iata_mapping_path)

    # Create a dictionary for ICAO to IATA mapping
    icao_iata_dict = icao_iata_mapping.set_index('ICAO')['IATA'].to_dict()

    # Extract unique acId
    unique_acid = filtered_rows['acId'].unique()

    # Create a new DataFrame with unique acId
    unique_rows = pd.DataFrame(unique_acid, columns=['acId'])

    # Extract Carrier and FlightNumber from acId
    unique_rows['Carrier'] = unique_rows['acId'].apply(lambda x: ''.join(filter(str.isalpha, x)))
    unique_rows['FlightNumber'] = unique_rows['acId'].apply(lambda x: ''.join(filter(str.isdigit, x)))
    unique_rows['FlightNumber'] = unique_rows['FlightNumber'].astype(str)

    # Define a function to get IATA code from Carrier
    def get_iata_code(carrier):
        if carrier in icao_iata_dict:
            return icao_iata_dict[carrier]
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
    data_ua_selected['FlightNumber'] = data_ua_selected['FlightNumber'].astype(str)

    # Merge unique_rows with data_ua_selected
    merged_data = pd.merge(unique_rows, data_ua_selected, on=['IATA', 'FlightNumber'], how='left')
    merged_data = merged_data.drop_duplicates()

    # Save merged_data to a new CSV file
    merged_csv_path = os.path.join(output_dir, 'merged_unique_data.csv')
    merged_data.to_csv(merged_csv_path, index=False)

    # Calculate the ratio of NaN values in the last column
    last_column_name = merged_data.columns[-1]
    nan_count = merged_data[last_column_name].isna().sum()
    total_count = len(merged_data)
    nan_ratio = nan_count / total_count

    # Print the results
    print(f"Total entries in merged_data: {total_count}")
    print(f"NaN entries in last column ({last_column_name}): {nan_count}")
    print(f"NaN ratio: {nan_ratio:.2%}")

    # Print the first few rows of merged_data
    print("Merged Data:")
    print(merged_data.head())

    # Check which IATA codes were not matched
    unmatched_iata = unique_rows[~unique_rows['IATA'].isin(merged_data['IATA'])]
    unmatched_csv_path = os.path.join(output_dir, 'unmatched_iata.csv')
    unmatched_iata.to_csv(unmatched_csv_path, index=False)

    print("Unmatched IATA codes:")
    print(unmatched_iata)

    return merged_csv_path, unmatched_csv_path

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python process_airlines.py <file_path_ua> <filtered_rows_path> <icao_iata_mapping_path> <output_dir>")
    else:
        file_path_ua = sys.argv[1]
        filtered_rows_path = sys.argv[2]
        icao_iata_mapping_path = sys.argv[3]
        output_dir = sys.argv[4]
        process_airlines_data(file_path_ua, filtered_rows_path, icao_iata_mapping_path, output_dir)
