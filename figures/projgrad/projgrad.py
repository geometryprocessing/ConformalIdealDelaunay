import csv, sys, os, argparse, mpmath, igl
import numpy as np
from mpmath import mp
sys.path.append("py")
from conformal_py import *

if __name__ == '__main__':
    # Parse arguments for the script
    parser = argparse.ArgumentParser(description='Get projected gradient from mesh with phi.')

    parser.add_argument("-i", "--input",  help="input folder that stores log files")
    parser.add_argument("-o", "--output",  help="output folder for the combind output file")
    parser.add_argument("-n", "--name",  help="name of the combined output file")
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)

    # Load mesh from file
    float_type = mp.mpf
    mp.prec = 300
    m = 'raptor50K'
    v, f = igl.read_triangle_mesh(os.path.join(args.input, m + '.obj'))
    Th_hat = np.loadtxt(os.path.join(args.input, m + '_Th_hat'), dtype=float)
    Th_hat = np.array([float_type(d) for d in Th_hat])

    # Round Th_hat to multiprecision multiples of pi/2
    for i,angle in enumerate(Th_hat):
        n=round(2*angle/mp.pi)
        Th_hat[i] = n*mp.pi/2
        assert abs(2*angle/mp.pi - n) < 1e-12
        
    # Create a doubled conformal mesh object
    vnstr = np.vectorize(lambda a:str(repr(a))[5:-2]) 
    Th_hat = vnstr(Th_hat)
    m0 = fv_to_double_mpf(v, f, Th_hat, [], [], [], [], []);

    # Load saved phi from file
    u0 = np.loadtxt(os.path.join(args.input, m + '_phi'), dtype=str)

    # Get projected gradient data (in multiprecision)
    lm_min = '0.001187887528662357698033824604522123991046100854873657226563'
    lm_max = '0.001187887545987688538151605399662003037519752979278564453125'
    decr_path = os.path.join(args.output, args.name)
    set_mpf_prec(mp.prec)
    newton_decrement_samples(m0, u0, decr_path, lm_min, lm_max, 100)
