CUDA_VISIBLE_DEVICES=4 python finetune.py --base_model '/home/ubuntu/cyt/RoPE_investigate/model/Llama-2-7b-hf' \
                      --data_path 'yahma/alpaca-cleaned'\
                      --output_dir './lora-alpaca-base' \
                      --micro_batch_size 16 \
                      --cutoff_len 512