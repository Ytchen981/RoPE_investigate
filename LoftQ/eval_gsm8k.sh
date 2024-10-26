
for ratio in -0.01 -0.05 -0.1 -0.2
do
  for threshold in 0.01 0.05 0.1 0.2
  do
      CUDA_VISIBLE_DEVICES=2 python test_gsm8k.py \
        --model_name_or_path /home_new/chenyiting/RoPE_angle/model/Mistral-7B-v0.1 \
        --adapter_name_or_path /home_new/chenyiting/RoPE_angle/LoftQ-main/exp_results/gsm8k_mistral_7b_base/Mistral-7B-v0.1/ep_6/lr_0.0003/seed_11 \
        --adjust_ratio $ratio \
        --mask_threshold $threshold \
        --batch_size 8
  done
done
        #--adapter_name_or_path /home_new/chenyiting/RoPE_angle/LoftQ/exp_results/gsm8k_TinyLlama_step1431k_mask_0.01/TinyLlama_1431k-3T/ep_6/lr_0.0003/seed_11 \