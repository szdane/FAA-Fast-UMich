import pandas as pd
import os
import sys

def process_flights(file_path_ua, tail_number, airports, output_dir):
    """
    Process flight data to filter based on a specified tail number and airports, and calculate turnaround times.

    Args:
        file_path_ua (str): Path to the input CSV file containing flight data.
        tail_number (str): Tail number of the aircraft to filter.
        airports (str): Comma-separated list of airport codes to filter.
        output_dir (str): Directory to save the output CSV file.

    Returns:
        str: Path to the output CSV file.
    """
    # Read the CSV file
    data_ua = pd.read_csv(file_path_ua)
    
    # Parse the list of airports
    airports_list = airports.split(',')
    
    # Filter rows where ORIGIN or DEST columns contain specified airports
    filtered_data = data_ua[data_ua['ORIGIN'].isin(airports_list) | data_ua['DEST'].isin(airports_list)]
    
    # Further filter rows based on the specified tail number
    tail_num_data = filtered_data[filtered_data['TAIL_NUM'] == tail_number]
    
    # Select required columns
    selected_columns = [
        'FL_DATE', 'TAIL_NUM', 'MKT_CARRIER', 'MKT_CARRIER_FL_NUM', 
        'ORIGIN_CITY_NAME', 'DEST_CITY_NAME', 'DEP_TIME', 'ARR_TIME'
    ]
    tail_num_data_selected = tail_num_data[selected_columns]
    
    # Convert time format from HHMM to HH:MM
    def convert_time(time):
        time = int(time)
        hours = time // 100
        minutes = time % 100
        return f"{hours:02d}:{minutes:02d}"

    tail_num_data_selected['DEP_TIME'] = tail_num_data_selected['DEP_TIME'].apply(convert_time)
    tail_num_data_selected['ARR_TIME'] = tail_num_data_selected['ARR_TIME'].apply(convert_time)
    
    # Convert FL_DATE to datetime format and sort by date
    tail_num_data_selected['FL_DATE'] = pd.to_datetime(tail_num_data_selected['FL_DATE'])
    
    # Group data by flight date and calculate turnaround times
    def process_group(group):
        group = group.sort_values(by='DEP_TIME')  # Sort by departure time
        flights = list(group['MKT_CARRIER'] + group['MKT_CARRIER_FL_NUM'].astype(str))
        turnaround_times = []
        for i in range(len(group) - 1):
            turnaround_time = calculate_turnaround_time(group.iloc[i]['ARR_TIME'], group.iloc[i+1]['DEP_TIME'])
            turnaround_times.append(turnaround_time)
        return pd.Series({
            'TAIL_NUM': group['TAIL_NUM'].iloc[0],
            'FLIGHTS': flights,
            'TURNAROUND_TIMES': turnaround_times
        })

    def calculate_turnaround_time(arr_time, dep_time):
        arr_hours, arr_minutes = map(int, arr_time.split(':'))
        dep_hours, dep_minutes = map(int, dep_time.split(':'))
        arr_total_minutes = arr_hours * 60 + arr_minutes
        dep_total_minutes = dep_hours * 60 + dep_minutes
        turnaround_minutes = dep_total_minutes - arr_total_minutes
        return f"{turnaround_minutes // 60}:{turnaround_minutes % 60:02d}"

    grouped_data = tail_num_data_selected.groupby('FL_DATE').apply(process_group).reset_index()

    # Format output
    formatted_data = grouped_data.apply(lambda row: f"{row['FL_DATE'].date()}, {row['TAIL_NUM']}, {row['FLIGHTS']}, {row['TURNAROUND_TIMES']}", axis=1)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the final result to a CSV file
    output_file_path = os.path.join(output_dir, f'{tail_number}_flights.csv')
    formatted_data.to_csv(output_file_path, index=False, header=False)
    
    # Return the output file path
    return output_file_path

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_flights.py <file_path_ua> <tail_number> <airports> <output_dir>")
    else:
        file_path_ua = sys.argv[1]
        tail_number = sys.argv[2]
        airports = sys.argv[3]
        output_dir = sys.argv[4]
        process_flights(file_path_ua, tail_number, airports, output_dir)
