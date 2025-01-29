#!/bin/bash
python compute_gns.py --model 0050000.pt --csv_path /home/c/cimsir/GenAI_Practical/gns_experiments/2_hyperparameters/results/model_0050000_Bb_100_reps_2.csv --vis_dir /home/c/cimsir/GenAI_Practical/gns_experiments/2_hyperparameters/visuals -acc -nw -b 50 -B 5000 -r 2 
python compute_gns.py --model 0050000.pt --csv_path /home/c/cimsir/GenAI_Practical/gns_experiments/2_hyperparameters/results/model_0050000_Bb_100_reps_2.csv --vis_dir /home/c/cimsir/GenAI_Practical/gns_experiments/2_hyperparameters/visuals -acc -nw -b 100 -B 10000 -r 2 
