#! /bin/bash

input_dir="data/N-torus-single-cone"
output_dir="output/variable_genus"
plot_dir="figures/variable_genus"
output_file="stat_file_genus.txt"
model_list=($(ls ${input_dir}))

rm -rf ${output_dir}

for i in ${model_list[@]}
do
  if [[ "$i" = *".obj" ]]; then
    python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f $i --error_log --do_reduction --no_plot_result --bypass_overlay&
  fi
done

wait
python3 ${plot_dir}/genus.py -i ${output_dir} -o ${plot_dir} -n ${output_file} &

wait
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{genus.tex}" &
