This README primarily concerns wx_grid_creator_2.py. This file is the predominant python code used to implement our weather processing pipeline. The high-level steps followed in the code are:

1. Downloading of the IEM N0Q Raster images
2. Creation of a TRACON boundary in lat/long space and an associate pre-TRACON boundary
3. Removal of all pixels from the N0Q Raster Image, except for those present in the pre-TRACON region
4. Conversion of the pixel greyscale intensity values into a feasible/infeasible binary mask
5. The grouping of the individual feasible/infeasible pixels to form discrete regions
6. Overlaying the discrete regions over the pre-TRACON boundary and TRACON boundary (for visual diagnostics)
7. Overlaying the optimized MILP waypoints generated over the previous raster image (once again for visual diagnostics)

Each of these steps has been given it's own code cell in the wx_grid_creator_2.py Python file. This allows you to run each step sequentially to ensure that no errors occur that make you wonder what the heck is going on or where the problem is occuring. It allows for step-by-step analysis of the pipeline, in the event that any bugs pop up.
