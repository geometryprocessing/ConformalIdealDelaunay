#! /bin/bash

input_dir="data/closed-sphere10K-random"
output_dir="output/compare_efficiency"
plot_dir="figures/compare_efficiency"
output_file="stat_file_flips.txt"

rm -rf ${output_dir}

for i in {0..39} #39
do
  for j in {0..24}
  do
  (( k = i*25 + j))
  # conformal ideal delaunay map
  python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f sphere10K.obj --eps 1e-10 --flip_count --suffix $k --no_plot_result --bypass_overlay&
  # conformal similarity map
  build/CSM_bin --m sphere10K.obj --d ${input_dir} --t ${input_dir}"/sphere10K_"$k"_Th_hat" --o ${output_dir} --p $k &
  done
  wait
done

wait
python3 ${plot_dir}/compare_efficiency.py -i ${output_dir} -o ${plot_dir} -n ${output_file}

wait
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{compare_efficiency.tex}" &

