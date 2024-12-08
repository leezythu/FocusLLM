import json
from pathlib import Path
import time
from typing import List, Tuple, Any
from typing import Optional

import torch
from dataclasses import dataclass, field, asdict
from transformers import HfArgumentParser
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from infbench_src.eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)

from accelerate import Accelerator, InitProcessGroupKwargs
from src import get_model_and_tokenizer,ModelArgs
from datetime import timedelta

MAX_POSITION_ID = 256 * 1024  # Determined by the model
TRUNCATE_LEN = 256 * 1024
device = torch.device("cuda")

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

if __name__ == "__main__":
    model_name = "infmem"
    # data_names = ["passkey","number_string","math_find","code_debug","longbook_choice_eng","kv_retrieval"]
    data_names = ["longbook_choice_eng"]
    for data_name in data_names:
        # Model
        max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]

        # Data
        result_dir = Path(args.output_dir)
        result_dir.mkdir(exist_ok=True, parents=True)
        examples = load_data(data_name, data_dir="infbench_data")
        start_idx = 0
        stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Start index: {start_idx}")
        print(f"Stop index: {stop_idx}")
        print(f"Max tokens: {max_tokens}")
        for i in range(start_idx, stop_idx):
            eg = examples[i]
            input_text = create_prompt(eg, data_name, model_name, data_dir="infbench_data")
            print(f"====== Example {i} ======")
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            input_ids = input_ids.to(device)
            print(f"Prompt has {input_ids.shape[-1]} tokens")

            if hasattr(model, "memory") and model.memory is not None:
                model.memory.reset()

            outputs = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                num_beams=1,
                do_sample=False,
                temperature=1.,
            )
            model_answer = tokenizer.decode(outputs[0,input_ids.shape[1]:].cpu())
            preds.append(
                {
                    "id": i,
                    "prediction": model_answer,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            print(preds)
            dump_jsonl(preds, output_path)
