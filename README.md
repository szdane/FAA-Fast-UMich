# FAA-Fast-UMich

**Hi Max! To run the project (MILP + Fuel Estimation):**
    **step 1: download filtered_rows.csv from Slack Channel, and put it in 5_Deliverable_Kuang/Input**
    **step 2: run 5_Deliverable_Kuang/main.py**

To run the MILP:
    python -m pip install -r requirements.txt
    cd 2_MILP_Serra/2.1_MILP/2.1.2_MILP
    python milp_multiple_Debug.py

Note that the resulting waypoints will ve saved under "2_MILP_Serra/2.3_Outputs_and_Results"

* All the codes you need to run FAA FAST project MILP Optimizer:
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/milp_multiple_Debug.py (MILP)
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/main_ver4_gurobi_debug.py (Fuel Estimation)
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/entry_exit_points.csv (Pre-TRACON Entry)
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/plot_multiple.py (Plot trajectory waypoints)
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/post-process.py (Active Corner Methods preprocessing)
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/post-process.py (Active corner methods post processing)
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/stress_test.py (main loop for stress test)
    * https://github.com/szdane/FAA-Fast-UMich/blob/main/FAST%20MILP/milp_stress.py (main solver for stress test)

