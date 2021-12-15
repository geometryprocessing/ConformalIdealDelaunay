import csv, sys, os, argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='assign input/output paths for newton decr sample data')
    parser.add_argument("-i", "--input",  help="newton decrement samples")
    parser.add_argument("-o", "--output",  help="shifted lambda data")
    
    args  = parser.parse_args()
    fname = args.input
    out   = args.output

    f = open(fname, "r"); fs = open(out, "w")
    lines = f.readlines()
    fs.write("lambda, newton_decrement\n")
    for line in lines[1:]:
        x = line.strip().split(",")
        fs.write(str(float(x[0])-0.015897601733878048) + "," + x[1] + "\n")
    f.close(); fs.close()