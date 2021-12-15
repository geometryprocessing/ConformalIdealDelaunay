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

  # Iterate over all prec directories
  final_max_grads = [['prec', 'max', 'oom'],]
  prec_dirs = os.listdir(args.input)
  for prec_dir in prec_dirs:
  # Get list of log files, sorted by name
    prec_path = os.path.join(args.input, prec_dir)
    prec = prec_dir[5:]
    logs = np.array(os.listdir(prec_path))
    if args.use_mpf:
      logs = np.array([f for f in logs if f.endswith("_mpf.csv")])
    else:
      logs = np.array([f for f in logs if f.endswith("_float.csv")])
    print(np.array([num for log in logs for num in re.split('[._-]',log)]))
    log_numbers = np.array([int(num) for log in logs for num in re.split('[._-]',log) if num.isdigit()])
    I = np.argsort(log_numbers)
    logs = logs[I]
    log_numbers = log_numbers[I]
    print(log_numbers)
    
    print(final_max_grads)
    for i, f in enumerate(logs):
      print(prec)
      final_max_grads.append([prec,0,0])
      log_path = os.path.join(prec_path, f)
      with open(log_path, newline='') as csvfile:
        grad_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
        # Get the final max error values and u differences at the final iteration up to iteration 50
        i = 0
        for row in grad_reader:
          final_max_grads[-1][1] = row[' max error']
          final_max_grads[-1][2] = float(row[' max_u']) - float(row[' min_u'])
          i += 1
          if i >= 50:
            break
  
    
  # Save the max_grads as a csv file
  final_max_grads = np.array(final_max_grads,dtype=str)
  final_max_grads_df = pd.DataFrame(final_max_grads)
  os.makedirs(args.output, exist_ok=True)
  table_path = os.path.join(args.output, args.name)
  with open(table_path, 'bw') as f:
    final_max_grads_df.to_csv(table_path,header=False,index=False)
    
