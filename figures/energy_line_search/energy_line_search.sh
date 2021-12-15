#! /bin/bash

input_dir="data/MPZ_cut"
output_dir="output/energy_line_search"
plot_dir="figures/energy_line_search"

# This scripts generate two data files: energy_sample.csv, newton_decrement.csv
python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f knot1.obj --energy_cond --energy_samples --eps 1e-7 --no_plot_result --bypass_overlay

join -o 1.1,1.2,2.2 -t ',' <(cat ${output_dir}/energy_sample.csv) <(cat ${output_dir}/newton_decrement.csv) > ${plot_dir}/merged.txt

pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{energy_line_search.tex}" &