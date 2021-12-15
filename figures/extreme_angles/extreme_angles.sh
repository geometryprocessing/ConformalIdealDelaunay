#! /bin/bash

input_dir="data/extreme_Th_hat_min"
output_dir="output/extreme_angles/Th_hat_min"
plot_dir="figures/extreme_angles"
output_file="stat_file_extreme.txt"

rm -rf ${output_dir}

for i in {0..87}
do
  python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f sphere1K.obj --suffix $i --print_summary --do_reduction --no_plot_result --bypass_overlay &
done

wait

for i in {0..87}
do
    build/CSM_bin --m sphere1K.obj --d ${input_dir} --t ${input_dir}"/sphere1K_"$i"_Th_hat" --o ${output_dir} --p $i&
done

# merge the two tables
wait

join -t ',' -o 1.1,2.4,1.3,2.3 <(cat ${output_dir}/summary_delaunay.csv | tail -n 87 | sort -n) <(cat ${output_dir}/summary_similarity.csv | tail -n 87 | sort -n) > ${output_dir}/merged_tab.csv
python3 ${plot_dir}/extreme_angles.py -i ${output_dir} -o ${plot_dir} -n ${output_file}


input_dir="data/extreme_Th_hat_max"
output_dir="output/extreme_angles/Th_hat_max"
plot_dir="figures/extreme_angles"
output_file="stat_file_extreme_large.txt"

rm -rf ${output_dir}

for i in {0..42}
do
  python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f sphere1K.obj --suffix $i --print_summary --do_reduction --no_plot_result --bypass_overlay &
done

wait

for i in {0..42}
do
  build/CSM_bin --m sphere1K.obj --d ${input_dir} --t ${input_dir}"/sphere1K_"$i"_Th_hat" --o ${output_dir} --p $i&
done

# merge the two tables
wait
join -t ',' -o 1.1,2.4,1.3,2.3 <(cat ${output_dir}/summary_delaunay.csv | tail -n 43 | sort -n) <(cat ${output_dir}/summary_similarity.csv | tail -n 43 | sort -n) > ${output_dir}/merged_tab.csv
python3 ${plot_dir}/extreme_angles.py -i ${output_dir} -o ${plot_dir} -n ${output_file}

# wait
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{extreme_angles.tex}" &


