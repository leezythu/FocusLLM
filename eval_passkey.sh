python eval_passkey.py \
--output_dir data/outputs/debug \
--model_name_or_path ./focusllm-checkpoint \
--max_length 8192 \
--enable_beacon True \
--local_window 2048 \
--memory_stride 2048 \
--beacon_param q k v o \