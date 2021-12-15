import csv, os
import numpy as np
import pandas as pd
import argparse
import re

if __name__ == "__main__":

  # Parse arguments for the script
  parser = argparse.ArgumentParser(description='Run the conformal map with options.')

  parser.add_argument("-i", "--input",  help="input folder that stores log files")
  parser.add_argument("-o", "--output",  help="output folder for the combind output file")
  parser.add_argument("-n", "--name",  help="name of the combined output file")
  parser.add_argument("--use_mpf",     action="store_true", help="True for multiprecision logs")
  args = parser.parse_args()

  # Initialize column headers
  final_max_grads = [['name', 'type', 'n_flips', 'time'],]

  # Add ConformalIdealDelaunayMapping results with label type 0
  log_path = os.path.join(args.input, 'flips_stats.csv')
  with open(log_path, newline='') as csvfile:
    grad_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for row in grad_reader:
      final_max_grads.append([])
      final_max_grads[-1].append(row['name'])
      final_max_grads[-1].append(0)
      final_max_grads[-1].append(row[' n_flips'])
      final_max_grads[-1].append(row[' time'])

  # Add old method results with label type 1
  log_path = os.path.join(args.input, 'summary_similarity.csv')
  with open(log_path, newline='') as csvfile:
    grad_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
    for row in grad_reader:
      final_max_grads.append([])
      final_max_grads[-1].append(row['name'])
      final_max_grads[-1].append(1)
      final_max_grads[-1].append(row[' n_flips'])
      final_max_grads[-1].append(row[' time'])
  
  # Save the max_grads as a csv file
  final_max_grads = np.array(final_max_grads,dtype=str)
  final_max_grads_df = pd.DataFrame(final_max_grads)
  os.makedirs(args.output, exist_ok=True)
  table_path = os.path.join(args.output, args.name)
  with open(table_path, 'bw') as f:
    final_max_grads_df.to_csv(table_path,header=False,index=False)
  
