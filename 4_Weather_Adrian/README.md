This README primarily concerns wx_grid_creator_2.py and fake_wx_frame_generator.py; These 2 files are the predominant python code used to implement our weather processing pipeline and also generate simulated fake weather for MILP test scenarios. 

--------

The high-level steps followed in the **wx_grid_creator_2.py** code are:

1. Downloading of the IEM N0Q Raster images
2. Creation of a TRACON boundary in lat/long space and an associate pre-TRACON boundary
3. Removal of all pixels from the N0Q Raster Image, except for those present in the pre-TRACON region
4. Conversion of the pixel greyscale intensity values into a feasible/infeasible binary mask
5. The grouping of the individual feasible/infeasible pixels to form discrete regions
6. Overlaying the discrete regions over the pre-TRACON boundary and TRACON boundary (for visual diagnostics)
7. Overlaying the optimized MILP waypoints generated over the previous raster image (once again for visual diagnostics)

Each of these steps has been given it's own code cell in the wx_grid_creator_2.py Python file. This allows you to run each step sequentially to ensure that no errors occur that make you wonder what the heck is going on or where the problem is occuring. It allows for step-by-step analysis of the pipeline, in the event that any bugs pop up.

--------

The high-level steps followed in the **fake_wx_frame_generator.py** code are:

1. Loading of the infeasible_regions CSV file containing bounding box corners and centroid coordinates
2. Definition of the WGS84 geodesic Earth model using pyproj.Geod for accurate great-circle motion
3. Specification of weather advection parameters (wind speed in knots, time step interval in minutes, and number of time steps)
4. Conversion of wind speed from knots to meters per time step using nautical mile-to-meter conversion
5. Definition of a geodesic northward translation function that moves longitude/latitude coordinates using azimuth = 0°
6. Iterative generation of time-evolved frames by shifting all bounding box corners (SW and NE) and centroids northward according to elapsed time
7. Appending time metadata (t_minutes) and total displacement (shift_meters) to each generated frame
8. Creation of an output directory for storing time-stepped weather frames
9. Export of one CSV file per time step representing the northward-advected infeasible weather regions
