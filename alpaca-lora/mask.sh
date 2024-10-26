for threhold in 0.01
do
  CUDA_VISIBLE_DEVICES=0 python finetune_mask.py --base_model /home/chenyiting/RoPE_angle/model/Llama-2-7b-hf \
                      --data_path yahma/alpaca-cleaned\
                      --output_dir ./lora-alpaca-mask-new-$threhold \
                      --micro_batch_size 4 \
                      --cutoff_len 512\
                      --threhold $threhold
done