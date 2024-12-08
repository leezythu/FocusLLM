import json
from pathlib import Path
import time
from typing import List, Tuple, Any

import torch
from torch import Tensor
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPast

from eval_utils import (
    dump_jsonl,
    create_prompt,
    load_data,
    get_answer,
    DATA_NAME_TO_MAX_NEW_TOKENS,
)
from vllm import LLM, SamplingParams
from args import parse_args


MAX_POSITION_ID = 200000  # Determined by the model
TRUNCATE_LEN = 200000

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens
    return tok.decode(tokens, skip_special_tokens=True)


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_tokens: int,
    verbose: bool = False,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    print("Truncating...")
    input_text = truncate_by_tokens(input_text, tok, TRUNCATE_LEN)
    if verbose:
        print("# chars:", len(input_text))
        print("=============== Input ===============")
        print(input_text[:200])
        print("...")
        print(input_text[-200:])
        print("=====================================")
    outputs = model.generate([input_text], sampling_params)

    output = outputs[0].outputs[0].text
    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str = "../../../yarn-mistral-7b-128k",
    ngpu=8,
):
    print("Loading tokenizer")
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    print("Loading model")
    start_time = time.time()
    llm = LLM(model=model_name, tensor_parallel_size=ngpu)
    print("Time taken:", round(time.time() - start_time))
    return llm, tok  # type: ignore


if __name__ == "__main__":
    
    args = parse_args()
    model_name = args.model_name

    print(json.dumps(vars(args), indent=4))
    data_name = args.task

    # Model
    max_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
    model, tok = load_model(args.model_path)
    # sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
    # Data
    result_dir = Path(args.output_dir, model_name)
    result_dir.mkdir(exist_ok=True, parents=True)
    examples = load_data(data_name, data_dir=args.data_dir)

    if args.stop_idx is None:
        args.stop_idx = len(examples)
        output_path = (
            result_dir / f"preds_{data_name}.jsonl"
        )
    else:
        output_path = (
            result_dir / f"preds_{data_name}_{args.start_idx}-{args.stop_idx}.jsonl"  # noqa
        )

    preds = []
    print("==== Evaluation ====")
    print(f"# examples: {len(examples)}")
    print(f"Start index: {args.start_idx}")
    print(f"Stop index: {args.stop_idx}")
    print(f"Verbose: {args.verbose}")
    print(f"Max tokens: {max_tokens}")
    for i in range(args.start_idx, args.stop_idx):
        eg = examples[i]
        input_text = create_prompt(eg, data_name, model_name, args.data_dir)
        print(f"====== Example {i} ======")
        pred = get_pred(
            model, tok, input_text, max_tokens=max_tokens, verbose=args.verbose
        )
        if args.verbose:
            print(pred)
        preds.append(
            {
                "id": i,
                "prediction": pred,
                "ground_truth": get_answer(eg, data_name),
            }
        )
        dump_jsonl(preds, output_path)
