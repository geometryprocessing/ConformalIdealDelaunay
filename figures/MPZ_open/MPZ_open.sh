#! /bin/bash

input_dir="data/MPZ_open"
output_dir="output/MPZ_open"
plot_dir="figures/MPZ_open"
output_file="stat_file_MPZ_boundary.txt"
model_list=($(ls ${input_dir}))

rm -rf ${output_dir}

for i in ${model_list[@]}
do
  if [[ "$i" = *".obj" ]]; then
    python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f $i --error_log --max_itr 50 --no_plot_result --bypass_overlay&
  fi
done

wait
python3 figures/combine_logs.py -i ${output_dir} -o ${plot_dir} -n ${output_file} --max_itr 50 &

wait
sed -i '1d' "${plot_dir}/${output_file}"
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{MPZ_open.tex}" &
