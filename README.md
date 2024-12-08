# FocusLLM

This repository contains the official implementation of *FocusLLM: Scaling LLM’s Context by Parallel Decoding*.

### Architecture

![image](https://anonymous.4open.science/r/FocusLLM/assets/framework.png)

### Environment

```
conda create -n focusllm python=3.10.14

conda activate focusllm

conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers deepspeed accelerate datasets peft pandas seaborn rouge fuzzywuzzy jieba python-Levenshtein
pip install flash-attn --no-build-isolation
```
### Data
Download the data for training and evaluation on Longbench. For Infinite-Bench, you can download from [InfiniteBench](https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/tree/main). We will also release the checkpoint of FocusLLM.
```
https://huggingface.co/datasets/zhangyik21/focusllm_train_data/tree/main
```
### Train

```
bash train.sh
```

Hyper parameters

- `memory_stride` [64, 128, 256, 512, 1024, 2048]
- `local_window` 3072
- `add_params` [q, k, v, o]

### Inference

Hyper parameters

- `memory_stride` 2048
- `local_window` 2048
- `inference_batch_size` [TODO in the next version]
  - parallel level for parallel decoding

#### Evaluate Passkey Retrieval

```
bash eval_passkey.sh
```

#### LongBench

```
bash eval_longbench.sh
```

#### Inf-Bench

```
bash eval_infbench.sh
```

#### Additional Notes
- This project builds upon the codebase of [Activation Beacon](https://github.com/FlagOpen/FlagEmbedding/tree/master/research/Long_LLM/activation_beacon), and we sincerely thank the authors for their valuable contribution. However, please note that the "beacon token" mentioned in our code actually refers to the "candidate token" as described in our paper. While we reuse the term "beacon token," its function is fundamentally different from the beacon token in the Activation Beacon paper. For details on how the candidate token functions, please refer to our paper.

- Due to memory constraints, during training, we randomly select either the repetition loss or continuation loss for optimization at each step. If sufficient memory is available, you can modify the forward function in src/activation_beacon_llama/modeling_llama.py to optimize both losses simultaneously.

#### Citation
If you find this repository useful, please give us a star ⭐.

To cite our work:
```
@misc{li2024focusllmscalingllmscontext,
      title={FocusLLM: Scaling LLM's Context by Parallel Decoding}, 
      author={Zhenyu Li and Yike Zhang and Tengyu Pan and Yutao Sun and Zhichao Duan and Junjie Fang and Rong Han and Zixuan Wang and Jianyong Wang},
      year={2024},
      eprint={2408.11745},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.11745}, 
}
```