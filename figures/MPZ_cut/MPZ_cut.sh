#! /bin/bash

input_dir="data/MPZ_cut"
output_dir="output/MPZ_cut"
plot_dir="figures/MPZ_cut"
output_file_float="non_disjoint_cuts_float_max_grads.txt"
output_file_mpf="non_disjoint_cuts_mpf_max_grads.txt"
model_list=($(ls ${input_dir}))

rm -rf ${output_dir}

for i in ${model_list[@]}
do
  if [[ "$i" = *".obj" ]]; then
    # use float
    python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f $i --error_log --max_itr 200 --no_plot_result --bypass_overlay&
    # use mpf
    python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f $i --use_mpf --prec 100 --error_log --max_itr 200 --no_plot_result --bypass_overlay&
  fi
done

wait
python3 figures/combine_logs.py -i ${output_dir} -o ${plot_dir} -n ${output_file_float} --max_itr 200 &
python3 figures/combine_logs.py -i ${output_dir} -o ${plot_dir} -n ${output_file_mpf} --use_mpf --max_itr 200 &

wait
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{MPZ_cut.tex}" &
