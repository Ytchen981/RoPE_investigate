for threshold in 0.01 0.005 0.001
do
  CUDA_VISIBLE_DEVICES=1 python train_gsm8k.py \
        --model_name_or_path /home_new/chenyiting/RoPE_angle/model/Phi-2 \
        --learning_rate 3e-4 \
        --seed 11 \
        --expt_name gsm8k_Phi_mask_$threshold \
        --output_dir exp_results/ \
        --num_train_epochs 6 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --weight_decay 0.1 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --do_train \
        --lora_init True \
        --mask_threshold $threshold
done