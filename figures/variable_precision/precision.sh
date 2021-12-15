#! /bin/bash

input_dir="data/closed-sphere1K-random"
output_dir="output/variable_precision"
plot_dir="figures/variable_precision"
output_file="stat_file_oom_50.txt"

prec_collection=( 53 75 100 125 150 )

rm -rf ${output_dir}

for i in {0..9}
do
  for j in {0..99}
  do
    (( k = i*100 + j))
    for prec in ${prec_collection[@]}
    do
      python3 py/script_conformal.py -o ${output_dir}"/prec_"$prec -i ${input_dir} -f sphere1K.obj --use_mpf --error_log --suffix $k --prec $prec --max_itr 50 --no_round_Th_hat --no_plot_result --bypass_overlay&  
    done
  done
  wait
done

wait
python3 figures/variable_precision/precision.py -i ${output_dir} -o ${plot_dir} -n ${output_file} --use_mpf &

wait
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{precision.tex}" &

