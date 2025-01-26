#!/bin/bash
python experiment_gns.py --model 0010000.pt --t_min 0 --t_max 500 -acc -nw --csv_path all_models_2_intervals.csv 
python experiment_gns.py --model 0010000.pt --t_min 501 --t_max 1000 -acc -nw --csv_path all_models_2_intervals.csv 
python experiment_gns.py --model 0020000.pt --t_min 0 --t_max 500 -acc -nw --csv_path all_models_2_intervals.csv 
python experiment_gns.py --model 0020000.pt --t_min 501 --t_max 1000 -acc -nw --csv_path all_models_2_intervals.csv 
python experiment_gns.py --model 0700000.pt --t_min 0 --t_max 500 -acc -nw --csv_path all_models_2_intervals.csv 
python experiment_gns.py --model 0700000.pt --t_min 501 --t_max 1000 -acc -nw --csv_path all_models_2_intervals.csv 
python experiment_gns.py --model 0750000.pt --t_min 0 --t_max 500 -acc -nw --csv_path all_models_2_intervals.csv 
python experiment_gns.py --model 0750000.pt --t_min 501 --t_max 1000 -acc -nw --csv_path all_models_2_intervals.csv 
