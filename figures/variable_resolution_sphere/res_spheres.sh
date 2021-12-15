#! /bin/bash

# issue: 'missing files' data not in place

input_dir="data/variable_sphere_random"
output_dir="output/variable_resolution_sphere"
plot_dir="figures/variable_resolution_sphere"
model_list=($(ls ${input_dir}))

rm -rf ${output_dir}

for j in {0..2}
do
  for k in {0..4}
  do
    (( n = j*5 + k + 6 ))
    for i in ${model_list[@]}
    do
      if [[ "$i" = *"n$n"*".obj" ]]; then
        echo $i
        # use float
        python3 py/script_conformal.py -o ${output_dir}/random -i ${input_dir} -f $i --error_log --no_lm_reset --max_itr 1000000 --lambda0 1e-6 --bound_norm_thres 0 --do_reduction --no_plot_result --bypass_overlay &
      fi
    done
  done
  wait
done

input_dir="data/variable_sphere_clustered"
output_dir="output/variable_resolution_sphere"
plot_dir="figures/variable_resolution_sphere"
model_list=($(ls ${input_dir}))

for j in {0..2}
do
  for k in {0..4}
  do
    (( n = j*5 + k + 6 ))
    for i in ${model_list[@]}
    do
      if [[ "$i" = *"n$n"*".obj" ]]; then
        # use float
        echo $i
        python3 py/script_conformal.py -o ${output_dir}/float -i ${input_dir} -f $i --error_log --no_lm_reset --max_itr 1000000 --lambda0 1e-6 --bound_norm_thres 0 --do_reduction --no_plot_result --bypass_overlay &
      fi
    done
  done
  wait
done

for j in {0..2}
do
  for k in {0..4}
  do
    (( n = j*5 + k + 6 ))
    for i in ${model_list[@]}
    do
      if [[ "$i" = *"n$n"*".obj" ]]; then
        # use mpf
        python3 py/script_conformal.py -o ${output_dir}/mpf -i ${input_dir} -f $i --use_mpf --no_round_Th_hat --error_log --no_lm_reset --prec 150 --max_itr 1000000 --lambda0 1e-6 --bound_norm_thres 0 --do_reduction --no_plot_result --bypass_overlay &
      fi
    done
  done
  wait
done


# wait
python3 figures/variable_resolution_sphere/res_spheres.py -i ${output_dir} -o ${plot_dir}
