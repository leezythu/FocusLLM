torchrun --nproc_per_node 8 -m main.train \
--output_dir outputs \
--model_name_or_path ./Llama-2-7b-chat-hf \
--train_data ./data/redpajama-sample.json ./data/longalpaca.json \
--max_length 8192 \
--min_length 3072 \
--max_train_num_per_data 200000 \
--num_train_epochs 1 \
--enable_beacon True \
--local_window 3072 \
--memory_stride 64 128 256 512 1024 2048 \
--beacon_param q k v o \
--group_by_length \
--gradient_checkpointing \
--save_strategy steps \
--save_steps 2000 \
--logging_steps 1 \
--deepspeed ./src/deepspeed/stage2.json \
--dataset_cache_dir ./cache \
--model_cache_dir ./cache