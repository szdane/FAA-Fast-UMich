# TRACON Plot Generation for Selected Airports
# Kuang Sun, June 2025

#Hello!:)

# input data processing package
import os # built-in Python module, provides a way to interact with the operating system, including file and directory operations
import pandas as pd

# input custom helper functions
from functions import computation_funcs, plot_funcs

# input plotting functions
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt

def main():
    # loop the same code for each csv file in the folder named "data"
    csv_file_list = [f for f in os.listdir("data") if f.endswith(".csv")]

    for selected_csv_file in csv_file_list:
        file_path = os.path.join("data", selected_csv_file) # "data_folder" contains folder name, "csv_file" contains csv file name

        ############
        # 1. Input #
        ############
        # 1.1 Input data from into dataframe "df_raw"
        df_raw = pd.read_csv(file_path, header=0)

        # 1.2 Extract star fix data from dataframe "df_raw" into dataframe "df_cleaned"
        df_cleaned = df_raw.iloc[:-1, :3].copy() #:-1 means without the last row; :3 means only copy the first three columns, because the last two columns are just NAN
        df_cleaned.columns = ["Star Fix Name", "Star Fix Lat", "Star Fix Lon"]

        df_cleaned["Star Fix Lat"] = df_cleaned["Star Fix Lat"].astype(float)
        df_cleaned["Star Fix Lon"] = df_cleaned["Star Fix Lon"].astype(float) # convert lat, lon into float format

        # 1.3 Extract airport data from dataframe "df_raw" into dataframe "df_cleaned"
        last_row = df_raw.iloc[-1] # Extract airport data from the last row

        tracon_name   = str(last_row.iloc[0]).strip()
        airport_full  = str(last_row.iloc[1]).strip()
        airport_abbr  = str(last_row.iloc[2]).strip()
        airport_lat   = float(last_row.iloc[3])
        airport_lon   = float(last_row.iloc[4]) # identify each data

        df_cleaned["TRACON"] = tracon_name
        df_cleaned["Airport"] = airport_full
        df_cleaned["Airport Abbr."] = airport_abbr
        df_cleaned["Airport Lat"] = airport_lat
        df_cleaned["Airport Lon"] = airport_lon # Put data into dataframe "df_cleaned"

        # 1.4 Reorder columns in dataframe "df_cleaned"
        df_cleaned = df_cleaned[[
            "TRACON", "Airport", "Airport Abbr.", "Airport Lat", "Airport Lon",
            "Star Fix Name", "Star Fix Lat", "Star Fix Lon"
        ]]

        # 1.5 find x-y coordinate of star fixes
        star_fix_x_list, star_fix_y_list = computation_funcs.proj_with_defined_origin(df_cleaned["Star Fix Lat"].tolist(), df_cleaned["Star Fix Lon"].tolist(), airport_lat, airport_lon)
        df_cleaned["Star fix x"] = star_fix_x_list
        df_cleaned["Star fix y"] = star_fix_y_list

        # 1.5 Change dataframe "df_cleaned" name into csv file's base name
        df_name = os.path.splitext(selected_csv_file)[0]  # Extract the csv file base name (without .csv)
        dic_dataframes = {df_name: df_cleaned} # Add to python dictionary "dic_dataframes"
        print(dic_dataframes[df_name])

        #############################################
        # 2. Find Star Fixes That Forms Convex Hull #
        #############################################
        # 2.1 Find outer star fixes lat and lon
        df_outer_star_fixes = computation_funcs.get_convex_hull_star_fixes(df_cleaned)
        print(df_outer_star_fixes, "\n")

        # 2.2 Find out star fixes x, y in plot
        outer_star_fix_x_list, outer_star_fix_y_list = computation_funcs.proj_with_defined_origin(df_outer_star_fixes["Lat"].tolist(), df_outer_star_fixes["Lon"].tolist(), airport_lat, airport_lon)
        df_outer_star_fixes["x"] = outer_star_fix_x_list
        df_outer_star_fixes["y"] = outer_star_fix_y_list

        # 2.3 Create Shapely Polygon from selected outer star fixes
        fix_points = list(zip(df_outer_star_fixes["x"], df_outer_star_fixes["y"]))
        TRACON_polygon = Polygon(fix_points) #filled polygon
        TRACON_polygon_x, TRACON_polygon_y = TRACON_polygon.exterior.xy


        ###########
        # 3. Plot #
        ###########
        plot_funcs.plot_2d_trajectory(df1=None, df2=None, label1=None, label2=None, plot_trajectory_endpoints = False,
                    tracon_polygon=(TRACON_polygon_x, TRACON_polygon_y), star_fixes = (star_fix_x_list, star_fix_y_list, df_cleaned["Star Fix Name"].tolist()), pretracon_circle=None, tracon_label="Star Fixes", pretracon_label=None,
                    lat0=airport_lat, lon0=airport_lon, airport_name = airport_abbr, lat_lon_grid=True,
                    waypoints=None, waypoints_plot = False,
                    preTRACON_entry=None, preTRACON_exit=None, pre_TRACON_endpoints = False,
                    title="2D TRACON for {}".format(df_name))


    # print(dataframes)
    plt.show()


if __name__ == "__main__":
    main()

