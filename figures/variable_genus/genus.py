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

  # Get list of log files, sorted by name
  logs = os.listdir(args.input)
  if args.use_mpf:
    logs = np.array([f for f in logs if f.endswith("_mpf.csv")])
  else:
    logs = np.array([f for f in logs if f.endswith("_float.csv")])
  log_numbers = np.array([int(num) for log in logs for num in re.split('[_-]',log) if num.isdigit()])
  I = np.argsort(log_numbers)
  logs = logs[I]
  log_numbers = log_numbers[I]
  
  final_max_grads = []
  for i, f in enumerate(logs):
    final_max_grads.append([log_numbers[i],0])
    log_path = os.path.join(args.input, f)
    with open(log_path, newline='') as csvfile:
      grad_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
      # Get the final max error values (the first column is the iteration number)
      for row in grad_reader:
        final_max_grads[-1][1] = row[' max error']

  
  # Save the max_grads as a csv file
  final_max_grads = np.array(final_max_grads,dtype=float)
  final_max_grads_df = pd.DataFrame(final_max_grads)
  os.makedirs(args.output, exist_ok=True)
  table_path = os.path.join(args.output, args.name)
  with open(table_path, 'bw') as f:
    final_max_grads_df.to_csv(table_path,index=False)
  
