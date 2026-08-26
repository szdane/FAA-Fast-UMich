# FAA-DATA

# FAA
This project integrates NASA and BTS data to visualize and analyze airline trajectories, focusing on landings at the same airport, regions traversed, and turnaround times in the ZOB area. 

process_airspace.py
Description: Processes ZOB area files downloaded from NASA and converts them into readable CSV files.


Usage: python process_airspace.py <input_nasa_file> <output_csv_file>


process_airline.py
Description: Integrates NASA ZOB area data with BTS data to find the departure and arrival airports for each airline ID.


Usage: python process_airline.py <input_zob_csv> <input_bts_data> <output_csv_file>


get_airline_info.py
Description: Retrieves departure and arrival airport information for a given airline ID.


Usage: python get_airline_info.py <airline_id>


flight_region.py
Description: Determines the sequence of regions traversed by each airline and the time spent in each region. It accounts for interruptions if the airline passes through an airport.


Usage: python flight_region.py <input_flight_data> <output_region_data>


process_flights.py
Description: Given a tail number, it identifies paired airline IDs within a day and calculates turnaround times at ZOB area airports.


Usage: python process_flights.py <input_flight_data> <tail_number> <output_paired_data>


plot_flight_tracks.py
Description: Plots all flight tracks contained in the NASA data files.


Usage: python plot_flight_tracks.py <input_flight_data> <output_directory>


plot_arrival_airports.py
Description: Visualizes flight tracks for flights landing at the same airport.


Usage: python plot_arrival_airports.py <input_flight_data> <output_directory>


plot_region_tracks.py
Description: Displays detailed flight routes of an airline within the ZOB area on a map.


Usage: python plot_region_tracks.py <input_flight_data> <airline_id> <output_directory>
