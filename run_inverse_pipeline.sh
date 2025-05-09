inverser_cfg=(
  # Input image, save name, model_name, use mask, opt_env_from, opt_src, opt_order
  "examples/99866.png 99866 pos_mlp no_mask 2 a rm a" # for real images we only use the predicted albedo for optimization, as predicted roughness and metallic are not accurate
  "examples/indoor2.png indoor2 pos_mlp no_mask 2 a rm a"
)

read -p "Select which to run, 0 for all images, 1 for the first image, 2 for the second image, etc.:" img_idx

if [[ $img_idx == "0" ]]; then
  inverser_cfg=("${inverser_cfg[@]}")
else
  inverser_cfg=("${inverser_cfg[$img_idx-1]}")
fi

echo "Selected image: ${inverser_cfg[@]}"

for cfg in "${inverser_cfg[@]}"; do
  read -r img_path save_name model_name use_mask opt_env_from opt_src opt_order <<< "$cfg"
  echo "img_path: $img_path, save_name: $save_name, model_name:$model_name, opt_src: $opt_src, opt_order: $opt_order, use_mask: $use_mask, opt_env_from: $opt_env_from"

  cmd="python inverse_img_w_mi.py --img_inverse_path=\"$img_path\" --save_name=\"$save_name\" --model_name=\"$model_name\" --opt_env_from=$opt_env_from" 
  
  [[ -n $opt_src ]] && cmd+=" --opt_src=\"$opt_src\""
  [[ -n $opt_order ]] && cmd+=" --opt_order $opt_order"
  [[ $use_mask == "use_mask" ]] && cmd+=" --use_mask"
  [[ $opt_env_from == "opt_env_from" ]] && cmd+=" --opt_env_from"
  echo "Executing command: $cmd"
  eval "$cmd"
done