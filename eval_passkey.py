import os
import datasets
import time
import torch
from datetime import timedelta
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from accelerate import Accelerator, InitProcessGroupKwargs
from transformers import HfArgumentParser
from torch.utils.data import DataLoader

from src import ModelArgs, DatasetProcessFn, DefaultDataCollator, FileLogger, get_model_and_tokenizer, makedirs, split_file_dir_name_ext, evaluate_perplexity


@dataclass
class Args(ModelArgs):
    eval_data: str = field(
        default="activation-beacon:lm/pg19.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="data/results/lm/",
        metadata={'help': 'Output directory for results and logs.'}
    )

    retokenize: bool = field(
        default=False,
        metadata={'help': 'Retokenize the corpus?'}
    )
    tokenize_max_char: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of chars to truncate.'}
    )

    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'}
    )
    padding_side: str = field(
        default="right",
        metadata={'help': 'Which side to pad?'}
    )
    stride: int = field(
        default=2048,
        metadata={'help': 'Streaming stride when evaluating perplexity.'}
    )

    max_sample_num: int = field(
        default=100,
        metadata={'help': 'How many samples to evaluate in eval_data?'}
    )
    min_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Minimum length for input_ids.'}
    )



parser = HfArgumentParser([Args])
args: Args = parser.parse_args_into_dataclasses()[0]

# increase timeout to avoid error
accelerator = Accelerator(cpu=args.cpu, kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=100000))])
model, tokenizer = get_model_and_tokenizer(args, accelerator=accelerator)


device = torch.device("cuda")

model.eval()


from numpy import random


def generate_prompt_landmark(n_garbage, seed):
    """Generates a text file and inserts an passkey at a random position."""
    rnd_state = random.get_state()
    random.seed(seed)
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 50000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        garbage_prefix,
        information_line,
        garbage_suffix,
        task_description,
        final_question,
    ]
    random.set_state(rnd_state)
    return "\n".join(lines), str(pass_key)


def passkey_retrieval_test(n_garbage=60000, seed=555):

    #n_garbage=60000 results in ~16k tokens

    prompt, answer = generate_prompt_landmark(n_garbage, seed)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to(device)
    print(f"Prompt has {input_ids.shape[-1]} tokens")

    answer_ids = tokenizer(answer, return_tensors="pt").input_ids[:, 1:] # drop BOS
    if hasattr(model, "memory") and model.memory is not None:
        model.memory.reset()

    outputs = model.generate(
        input_ids,
        max_new_tokens=answer_ids.shape[-1],
        num_beams=1,
        do_sample=False,
    )

    # model_answer = outputs[0, -answer_ids.shape[-1]:].cpu()
    model_answer = outputs[0,input_ids.shape[1]:].cpu()
    

    # is_correct = (model_answer == answer_ids[0]).all().item()
    is_correct = tokenizer.decode(answer_ids[0].cpu())==tokenizer.decode(model_answer.cpu())
    print(f"The correct answer is {tokenizer.decode(answer_ids[0].cpu())}",flush=True)
    print(f"The model answer is ::{tokenizer.decode(model_answer.cpu())}, is_correct : {is_correct}",flush=True)
    return is_correct


num_tests = 10
lengths = [15000,30000,60000,375000,960000,1500000]
for length in lengths:
    passed_tests = 0
    for i in range(num_tests):
        passed_tests += passkey_retrieval_test(n_garbage=length, seed=i)

    print(f"Accuracy is {passed_tests/num_tests}",flush=True)