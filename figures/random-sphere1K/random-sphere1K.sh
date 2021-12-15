#! /bin/bash

input_dir="data/closed-sphere1K-random"
output_dir="output/closed-sphere1K-random"
plot_dir="figures/random-sphere1K"
output_file="stat_file_decay_random.txt"

rm -rf ${output_dir}

for i in {0..999}
do
  python3 py/script_conformal.py -o ${output_dir} -i ${input_dir} -f sphere1K.obj --error_log --suffix $i --max_itr 50 --no_plot_result --bypass_overlay &
done

wait
python3 figures/combine_logs.py -i ${output_dir} -o ${plot_dir} -n ${output_file} --max_itr 50 &

wait
sed -i '1d' "${plot_dir}/${output_file}"
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{decay0_499.tex}" &
pdflatex -output-directory ${plot_dir} "\def\data_dir{$plot_dir} \input{decay500_999.tex}" &
