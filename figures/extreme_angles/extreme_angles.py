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
  final_max_grads = []
  # Get list of log files, sorted by name
  merged_tab_path = os.path.join(args.input, "merged_tab.csv")
  with open(merged_tab_path, newline='') as csvfile:
    grad_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in grad_reader:
      final_max_grads.append([row[1], row[2], row[3]])
    
  # Save the max_grads as a csv file
  final_max_grads = np.array(final_max_grads,dtype=float)
  I = final_max_grads[:,0].argsort()
  final_max_grads = final_max_grads[I]
  final_max_grads_df = pd.DataFrame(final_max_grads)
  os.makedirs(args.output, exist_ok=True)
  table_path = os.path.join(args.output, args.name)
  with open(table_path, 'bw') as f:
    final_max_grads_df.to_csv(table_path,header=False,index=False)
    
