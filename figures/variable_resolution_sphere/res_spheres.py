import os, pickle, sys, csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    # Parse arguments for the script
    parser = argparse.ArgumentParser(description='Run the conformal map with options.')

    parser.add_argument("-i", "--input",  help="input folder that stores log files")
    parser.add_argument("-o", "--output",  help="output folder for the heatmap")
    args = parser.parse_args()

    n_vals = np.arange(6,21)
    r_vals = np.concatenate((np.arange(1,6),np.arange(10,100,10),np.arange(95,100)))
    sphere_test_results = []
    sphere_sizes = np.zeros(len(n_vals),dtype=int)
    ratios = np.zeros(len(r_vals))

    sphere_test_dirs = ['float', 'random', 'mpf']
    # Iterate through test result directories
    for sphere_test_dir in sphere_test_dirs:
        print("Adding heatmap for ", sphere_test_dir)
        sphere_test_results.append({n:[] for n in n_vals})

        # Iterate through r_vals  and n_vals
        for i,r in enumerate(r_vals):
            # Take complementary ratio so that it gives the ratio of fixed vertices
            # instead of the ratio of rebalanced vertices
            ratios[i] = str((100-r)/100)
            for j,n in enumerate(n_vals):
                # Compute the sphere size from the inclusion-exclusion principle
                sphere_sizes[j] = 6*n*n - 12*n + 8

                # Get the final error for the given result directory and r,n values
                result_file = 'sphere_x30_n' + str(n) + '_r' + str(r)
                if 'mpf' in sphere_test_dir:
                    result_file += '_mpf.csv'
                else:
                    result_file += '_float.csv'
                result_path = os.path.join(args.input, sphere_test_dir, result_file)
                try:
                    with open(result_path, newline='') as csvfile:
                        sphere_test_results[-1][n].append(0)
                        grad_reader = csv.DictReader(csvfile, delimiter=',', quotechar='|')
                        for row in grad_reader:
                            sphere_test_results[-1][n][-1] = np.log10(float(row[' max error']))
                # Default to result 2 (maximum color value for the heatmap) if no result data found
                except FileNotFoundError:
                    print("Missing file")
                    sphere_test_results[-1][n].append(2)
                    continue
                   

    # Initialize heatmap
    fig,axs = plt.subplots(1,len(sphere_test_results),figsize=(8,3),sharey=True)
    matplotlib.rc('font', size=16)
    matplotlib.rc('xtick', labelsize=10) 
    matplotlib.rc('ytick', labelsize=10)

    # Add heatmap for each test result directory
    for i,sphere_test_results_i in enumerate(sphere_test_results):
        sphere_test_results_arr = np.array([sphere_test_results_i[n] for n in sphere_test_results_i])
        im = axs[i].imshow(sphere_test_results_arr.T, cmap='seismic', interpolation='nearest')
        axs[i].set_xticks(np.arange(len(sphere_sizes))[1::4])
        axs[i].set_yticks(np.arange(len(ratios))[::2])
        axs[i].set_xticklabels(sphere_sizes[1::4])
        axs[i].set_yticklabels(ratios[::2])
        axs[i].set_aspect(1)
        im.set_clim([-22,2])
        
    # Add axis titles
    axs[0].set(ylabel='ratio of vertices')
    axs[1].set(xlabel='number of vertices in sphere')

    # Add color bar
    fig.tight_layout()
    fig.colorbar(im, ax=axs.tolist(),shrink=0.8)

    # Save heatmap as both png and eps file
    heatmap_path = os.path.join(args.output, 'heatmap.eps')
    fig.savefig(heatmap_path, format='eps')
    heatmap_path = os.path.join(args.output, 'heatmap.png')
    fig.savefig(heatmap_path)

  
