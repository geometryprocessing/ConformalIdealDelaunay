import csv, os
import numpy as np
import pandas as pd
import argparse

if __name__ == "__main__":

  # Parse arguments for the script
  parser = argparse.ArgumentParser(description='Run the conformal map with options.')

  parser.add_argument("-i", "--input",  help="input folder that stores log files")
  parser.add_argument("-o", "--output",  help="output folder to write combined data")
  parser.add_argument("-n", "--name",  help="name of the combined output file")
  parser.add_argument("--use_mpf",     action="store_true", help="True for multiprecision logs")
  parser.add_argument("--max_itr", type=int, help="number of iterations for the combined logs", default=500)
  args = parser.parse_args()

  # Get list of log files
  logs = os.listdir(args.input)
  if args.use_mpf:
    logs = [f for f in logs if f.endswith("_mpf.csv")]
  else:
    logs = [f for f in logs if f.endswith("_float.csv")]

  
  max_grads = []
  for f in logs:
    max_grads.append([])
    log_path = os.path.join(args.input, f)
    with open(log_path, newline='') as csvfile:
      grad_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')

      # Get the max error values (the first column is the iteration number)
      for row in grad_reader:
        max_grads[-1].append(row[' max error'])    
        # Ensure each log is the requested number of iterations
        if(len(max_grads[-1])>=args.max_itr):
          break
      while(len(max_grads[-1])<args.max_itr):
        max_grads[-1].append(max_grads[-1][-1])
  
  # Save the max_grads as a csv file
  max_grads = np.array(max_grads,dtype=float).T
  max_grads_df = pd.DataFrame(max_grads)
  os.makedirs(args.output, exist_ok=True)
  table_path = os.path.join(args.output, args.name)
  with open(table_path, 'bw') as f:
    max_grads_df.to_csv(table_path,index=False)
  
