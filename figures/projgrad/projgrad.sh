#! /bin/bash
input_dir="data/projgrad"
output_dir="output/projgrad"
plot_dir="figures/projgrad"
output_file="projgrad.txt"

python3 figures/projgrad/projgrad.py -i ${input_dir} -o ${output_dir} -n ${output_file}

wait
python3 ${plot_dir}/translate_lambda.py -i ${output_dir}/projgrad.txt -o ${output_dir}/projgrad_shift.txt

wait
pdflatex -output-directory ${plot_dir} "\def\data_dir{$output_dir} \input{projgrad.tex}" &

