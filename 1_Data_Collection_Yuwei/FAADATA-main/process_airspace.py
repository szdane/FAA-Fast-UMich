# process_airspace.py

import os
import pandas as pd

def process_airspace_file(input_file_path, output_dir):
    # Read data from the file
    data = []
    data_36 = []
    data_45 = []

    # Read the file and categorize lines based on their length
    with open(input_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 36:
                data_36.append(parts)
            elif len(parts) == 45:
                data_45.append(parts)
            elif len(parts) == 4:
                data.append(parts)

    # Create DataFrames for each type of data
    df = pd.DataFrame(data)
    df_36 = pd.DataFrame(data_36)
    df_45 = pd.DataFrame(data_45)

    # Define column names for each DataFrame
    columns_36 = [
        "recType", "recTime", "fltKey", "bcnCode", "cid", "Source", "msgType", "acId",
        "recTypeCat", "coord1", "coord2", "alt", "significance", "coord1Accur", "coord2Accur",
        "altAccur", "groundSpeed", "course", "rateOfClimb", "altQualifier", "altIndicator",
        "trackPtStatus", "leaderDir", "scratchPad", "msawInhibitInd", "assignedAltString",
        "controllingFac", "controllingSec", "receivingFac", "receivingSec", "activeContr",
        "primaryContr", "kybrdSubset", "kybrdSymbol", "adsCode", "opsType"
    ]

    columns_45 = columns_36 + [
        "airportCode", "trackNumber", "tptReturnType", "modeSCode", "sensorTrackNumberList",
        "spi", "dvs", "dupM3a", "tid"
    ]

    df.columns = ["Column1", "Column2", "Column3", "Column4"]
    df_36.columns = columns_36
    df_45.columns = columns_45

    df_36['recTime'] = pd.to_datetime(df_36['recTime'], unit='s', origin='unix')
    df_45['recTime'] = pd.to_datetime(df_45['recTime'], unit='s', origin='unix')
    df_36['alt'] = pd.to_numeric(df_36['alt'], errors='coerce')
    df_45['groundSpeed'] = pd.to_numeric(df_45['groundSpeed'], errors='coerce')
    df_36.fillna(method='ffill', inplace=True)
    df_45.fillna(method='ffill', inplace=True)
    filtered_combined_df = pd.concat([df_36, df_45])
    filtered_rows = filtered_combined_df[filtered_combined_df['recType'] == '3']

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save the filtered DataFrame to a CSV file in the output directory
    csv_file_path = os.path.join(output_dir, 'filtered_rows.csv')
    filtered_rows.to_csv(csv_file_path, index=False)

    # Generate a unique acId list
    unique_acid_list = filtered_rows['acId'].unique()

    # Save the unique acId list to a CSV file
    unique_acid_file_path = os.path.join(output_dir, 'unique_acid_list.csv')
    pd.DataFrame(unique_acid_list, columns=['acId']).to_csv(unique_acid_file_path, index=False)

    return csv_file_path, unique_acid_file_path

# Entry point for the script
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python process_airspace.py <input_file_path> <output_dir>")
    else:
        input_file_path = sys.argv[1]
        output_dir = sys.argv[2]
        csv_file_path, unique_acid_file_path = process_airspace_file(input_file_path, output_dir)
        print(f"Filtered rows saved to: {csv_file_path}")
        print(f"Unique acId list saved to: {unique_acid_file_path}")
