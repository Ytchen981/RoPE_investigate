# LoftQ: train 4-bit 64-rank llama-2-7b on wikitext-2 using 8 GPU
# global batch size = 64
for threshold in 0.01 0.005 0.001
do
  CUDA_VISIBLE_DEVICES=3  python train_clm.py \
      --model_name_or_path /home_new/chenyiting/RoPE_angle/model/Phi-2 \
      --output_dir exp_results/wikitext-2/Phi-2_mask-$threshold \
      --learning_rate 3e-4  \
      --seed 11 \
      --dataset_name wikitext \
      --dataset_config wikitext-2-raw-v1 \
      --num_train_epochs 3 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 8 \
      --save_strategy "epoch" \
      --weight_decay 0.1 \
      --warmup_ratio 0.03 \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --do_train --do_eval \
      --logging_steps 50 \
      --block_size 1024 \
      --lora_init True \
      --mask_threshold $threshold
done
