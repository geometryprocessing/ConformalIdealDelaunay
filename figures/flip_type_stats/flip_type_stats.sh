#! /bin/bash

input_dir="data/boundary-disk5K-random"
output_dir="output/flip_type_stats"
plot_dir="figures/flip_type_stats"
output_file="stat_file_boundary_random_data.txt"

rm -rf ${output_dir}

for i in {0..999}
do
  python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f disk5K.obj --flip_count --error_log --suffix $i --no_plot_result --bypass_overlay&
done

wait
cp ${output_dir}/flips_stats.csv ${plot_dir}/${output_file}

wait
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{flip_type_stats.tex}"  &
