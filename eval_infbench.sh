python eval_infbench.py \
--model_name infmem \
--model_name_or_path ./focusllm-checkpoint \
--enable_beacon True \
--local_window 2048 \
--memory_stride 2048 \
--beacon_param q k v o \