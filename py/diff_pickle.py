import numpy as np
import mpmath as mp
import pickle
import sys

import argparse

if __name__ == "__main__":

  # Parse arguments for the script
  parser = argparse.ArgumentParser(description='Script to check whether two conformal results are the same')
  parser.add_argument("-a", "--file_a",     help="file a to compare")
  parser.add_argument("-b", "--file_b",     help="file b to compare")

  args = parser.parse_args()
  file_a = args.file_a
  file_b = args.file_b
  with open(file_a, 'rb') as fp:
    n0, to0, f0, h0, out0, opp0, l0, R0, u0 = pickle.load(fp)
  with open(file_b, 'rb') as fp:
    n1, to1, f1, h1, out1, opp1, l1, R1, u1 = pickle.load(fp)
  
  n0 = np.array(n0)    ; n1 = np.array(n1)
  to0 = np.array(to0)  ; to1 = np.array(to1)
  f0 = np.array(f0)    ; f1 = np.array(f1)
  out0 = np.array(out0); out1 = np.array(out1)
  opp0 = np.array(opp0); opp1 = np.array(opp1)
  l0 = np.array(l0)    ; l1 = np.array(l1)
  R0 = np.array(R0)    ; R1 = np.array(R1)
  u0 = np.array(u0)    ; u1 = np.array(u1)

  if   (u0      != u1   ).any() or \
       (l0      != l0   ).any():
    print(1)

  # out comparison doesn't work with new tufted cover
  elif((n0  != n1).any()   or \
      (to0  != to1).any()  or \
      (f0   != f1).any()   or \
      (len(out0) != len(out1)) or \
      #(out0 != out1).any() or \
      (opp0 != opp1).any() or \
      (R0   != R1).any()) :
    print(2)

#  elif (R0 and not R1)                  or \
#       (not R0 and R1)                  or \
#       ((R0 and R1)                     and \
#        (  (R0.refl != R1.refl).any()       or \
#           (R0.vtype != R1.vtype).any()     or \
#           (R0.refl_f != R1.refl_f).any()   or \
#           (R0.refl_he != R1.refl_he).any() or \
#           (R0.ftype != R1.ftype).any()     or \
#           (R0.etype != R1.etype).any()
#        )):
#    print(3)
    
  else: # same file
    print(0)
