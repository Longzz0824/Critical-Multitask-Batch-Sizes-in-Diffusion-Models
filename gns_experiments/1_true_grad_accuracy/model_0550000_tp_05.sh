#!/bin/bash
python compute_gns.py --model 0550000.pt --csv_path /home/c/cimsir/GenAI_Practical/gns_experiments/1_true_grad_accuracy/results/model_0550000_tp_05.csv --vis_dir /home/c/cimsir/GenAI_Practical/gns_experiments/1_true_grad_accuracy/visuals -acc -nw -p 0.5 -r 5 -b 50 -B 5000 
python compute_gns.py --model 0550000.pt --csv_path /home/c/cimsir/GenAI_Practical/gns_experiments/1_true_grad_accuracy/results/model_0550000_tp_05.csv --vis_dir /home/c/cimsir/GenAI_Practical/gns_experiments/1_true_grad_accuracy/visuals -acc -nw -p 0.5 -r 5 -b 50 -B 5000 
