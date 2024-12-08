import torch
import numpy as np
import torch.distributed as dist
from transformers.utils import logging
from typing import List, Tuple, Optional
from .modeling_retrieval import BM25Retriever
import random

logger = logging.get_logger(__name__)


class Memory(torch.nn.Module):
    def __init__(self, model_config, local_window:int=1024, memory_stride:List[int]=[512], beacon_attn:str="step-expansion", beacon_attend_previous:bool=True, beacon_ratio:List[int]=[8], beacon_stride_mix:str="step-random", beacon_ratio_mix:str="step-random", beacon_param:List[str]=["q", "k", "v", "o"], k_seq_dim:int=2, v_seq_dim:int=2, retrieval_method:str=None, retrieval_topk:int=2) -> None:
        super().__init__()

        assert beacon_attn in ["segmentation", "step-expansion", "full-coverage"], f"beacon_attn {beacon_attn} not implemented!"
        assert beacon_stride_mix in ["instance-random", "step-random", "mix-random"], f"beacon_stride_mix {beacon_stride_mix} not implemented!"
        assert beacon_ratio_mix in ["instance-random", "step-random", "mix-random", "sequence"] or "adapt-" in beacon_ratio_mix, f"beacon_ratio_mix {beacon_ratio_mix} not implemented!"

        info = f"applying activation beacon on {beacon_param}, with window size {local_window}, stride {memory_stride} (mixed by {beacon_stride_mix}), {beacon_attn} attention ({'attending to previous beacons' if beacon_attend_previous else 'not attending to previous beacons'}), condensing ratio {beacon_ratio} (mixed by {beacon_ratio_mix}), {retrieval_method+' retrieval'+' top-'+str(retrieval_topk) if retrieval_method is not None else 'no retrieval'}, ..."
        logger.info(info)

        self.local_window = local_window
        self.memory_stride = memory_stride
        self.beacon_attn = beacon_attn
        self.beacon_ratio = beacon_ratio
        self.beacon_stride_mix = beacon_stride_mix
        self.beacon_ratio_mix = beacon_ratio_mix
        max_beacon_size = max([local_window // x for x in beacon_ratio if x > 0] + [1])
        self.beacon_tokens = torch.zeros(max_beacon_size, dtype=torch.long) + model_config.vocab_size
        
        # initialize necessary parameters
        self.k_seq_dim = k_seq_dim
        self.v_seq_dim = v_seq_dim
        self.num_layers = model_config.num_hidden_layers
        self.max_position_embeddings = model_config.max_position_embeddings

        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk

        self.rng = np.random.default_rng(42)
        self.reset()
        # self.neg_beacon_activations = [(None, None) for _ in range(self.num_layers)]
    
    @property
    def finish(self):
        return self.end_idx == self.sequence_length
    
    def get_memory_size(self):
        beacon_memory_size = 0
        raw_memory_size = 0
        if self.beacon_activations[0][0] is not None:
            beacon_memory_size += self.beacon_activations[0][0].shape[self.k_seq_dim]
        # if self.raw_activations[0][0] is not None:
        #     raw_memory_size += self.raw_activations[0][0].shape[self.k_seq_dim]
        memory_size = beacon_memory_size 
        return beacon_memory_size, raw_memory_size, memory_size

    def reset(self):
        # the length of current sequence
        self.sequence_length = 0
        # the length of all sequences until the memory is reset
        self.total_sequence_length = 0
        # the cursor pointing to the start of the current window
        self.start_idx = 0
        # the cursor pointing to the end of the current window
        self.end_idx = 0
        # the beacon sizes of all strides
        self._beacon_sizes = []
        # the step index
        self.step_idx = 0
        #lzy
        self.input_ids = None

        if self.beacon_ratio_mix != "step-random":
            self._stride = None
            self._ratio = None

        self.batch_loss = None
        self.valid_token_num = None

        # self.raw_activations = [(None, None) for _ in range(self.num_layers)]
        self.beacon_activations = [(None, None) for _ in range(self.num_layers)]
        # self.local_q_activations = [(None, None) for _ in range(self.num_layers)]
        self.raw_activations = []
        # self.neg_beacon_activations = [(None, None) for _ in range(self.num_layers)]
        self.last_input_ids = None
        # self.local_q_activations = []
        self.local_a_input_ids = None
        self.generated_id_cache = None

        # NOTE: when training, we strictly aligh the rng_state across processes
        if self.training and dist.is_initialized():
            rng_state = self.rng.__getstate__()
            if dist.get_rank() == 0:
                obj = [rng_state]
            else:
                obj = [None]
            dist.broadcast_object_list(obj, src=0)
            self.rng.__setstate__(obj[0])

    def prepare(self, input_ids, attention_mask, labels, autoencoder):

        assert input_ids.shape[0] == 1, f"Make sure batch_size is 1!"
        assert self.start_idx == 0 #has been reseted
        self.sequence_length = input_ids.shape[1]
        #lzy
        self.start_idx = self.end_idx = 0
        self.chunk_number = (self.sequence_length - self.sequence_length % self.local_window) // self.local_window


        if labels is not None:
            # rotate labels in advance so that the loss of the last token is not ignored in every window
            labels = torch.cat([labels[:, 1:], labels.new_zeros((labels.shape[0], 1)) - 100], dim=-1)

        # if the current sequence has been completely processed
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        
        if self.chunk_number == 0:
            return
        min_local_size = 30
        self.local_size = random.choice(range(min_local_size,self.local_window-1))
        #autoencoder
        if autoencoder:
            local_chunk_number = random.choice(range(self.chunk_number))
            index_in_chunk = random.choice(range(self.local_window - self.local_size))
            self.local_query_start = local_chunk_number * self.local_window + index_in_chunk
        else:
            local_chunk_number = self.chunk_number - 1
            index_in_chunk = random.choice(range(1,self.local_window - self.local_size))
            self.local_query_start = local_chunk_number * self.local_window + index_in_chunk
            self.sequence_length = self.local_query_start 
        self.stride = random.choice(self.memory_stride)
        self.beacon_size = 1

        
    def retrieve_local_query(self, truncate = None):
        start = self.local_query_start
        end = self.local_query_start + self.local_size
        if truncate:
            truncate_size = min(512, self.stride, self.local_size)
            start = end - truncate_size
        input_ids = self.input_ids[:, start: end] 
        # if truncate:
        #     input_ids = torch.cat((input_ids,torch.tensor([2]).unsqueeze(0).cuda()), dim=1)
        if not input_ids[0][0] == 1:#start token
            input_ids = torch.cat((torch.tensor([1]).unsqueeze(0).cuda(), input_ids), dim=1) #add <s> to the beginning and </s> to the end
        return input_ids

    def retrieve_local_ans(self):
        start = self.local_query_start + self.local_size
        input_ids = self.input_ids[:, start: start+1]
        
        #insert local query before local ans
        local_q_input_ids = self.retrieve_local_query(truncate=False)
        labels = torch.cat([torch.zeros_like(local_q_input_ids[...,:-1]) - 100, input_ids], dim=1)
        local_q_input_ids = torch.cat([local_q_input_ids,input_ids[:,:-1]],dim=1)
        local_q_attention_mask = torch.ones_like(local_q_input_ids)
        
        input_ids = local_q_input_ids
        attention_mask = local_q_attention_mask
        
        raw_size_to_cache = 1
        past_key_values = []
        for layer_idx, (beacon_key, beacon_value) in enumerate(self.beacon_activations):
            layer_past_key_values = (beacon_key, beacon_value, 0, raw_size_to_cache, raw_size_to_cache)
            past_key_values.append(layer_past_key_values)
        
        # prepend 1 to attention mask for previous memory
        _, _, memory_size = self.get_memory_size()
        if memory_size > 0:
            attention_mask = torch.cat([attention_mask.new_ones(attention_mask.shape[0], memory_size), attention_mask], dim=1)

        return input_ids, attention_mask, past_key_values, labels 

    def step(self):
        start_idx = self.start_idx
        end_idx = start_idx + self.stride
        is_full_window = True
        if end_idx > self.sequence_length:
            end_idx = self.sequence_length
            is_full_window = False
        window_size = end_idx - start_idx
        beacon_size = self.beacon_size
        if is_full_window:
            next_start_idx = start_idx + self.stride
            raw_size_to_cache = end_idx - next_start_idx
            self.remaining_size = 0
        else:
            next_start_idx = start_idx
            raw_size_to_cache = window_size
            self.remaining_size = window_size
        
        # streamingly add new input_ids
        input_ids = self.input_ids[:, self.end_idx: end_idx]
        if self.attention_mask is not None:
            attention_mask = self.attention_mask[:, self.end_idx: end_idx]
        else:
            attention_mask = torch.ones_like(input_ids)
        if self.labels is not None:
            labels = self.labels[:, self.end_idx: end_idx]
        else:
            labels = None
        # lzy: append local query before beacons
        local_q_input_ids = self.retrieve_local_query(truncate=True)
        local_q_attention_mask = torch.ones_like(local_q_input_ids)
        
        input_ids = torch.cat([input_ids,local_q_input_ids],dim=1)
        attention_mask = torch.cat([attention_mask,local_q_attention_mask],dim=1)
        labels = torch.cat([labels, torch.zeros_like(local_q_input_ids) - 100], dim=1)

        past_key_values = []
        for i in range(self.num_layers):
            key = None
            value = None
            layer_past_key_values = (key, value, beacon_size, raw_size_to_cache, window_size)
            past_key_values.append(layer_past_key_values)

        self._beacon_sizes.append(beacon_size)
        self.start_idx = next_start_idx
        self.end_idx = end_idx
        self.step_idx += 1

        return input_ids, attention_mask, past_key_values, labels

    def find_available_device(self):
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            for i in range(num_gpus):
                gpu_props = torch.cuda.get_device_properties(i)
                total_memory = gpu_props.total_memory  # 总显存
                free_memory = total_memory - torch.cuda.memory_reserved(i)  # 剩余显存
                # print("free_memory",free_memory/1024/1024/1024)
                if free_memory > 7 * 1024 * 1024 * 1024:  # 1GB 转换为字节
                    return f'cuda:{i}'
        return 'cpu'

    def update_memory(self, past_key_values, chunk_idx = None,first_time_inferenece=True):
        """
        Accumulate beacon activations and raw activations.
        """
        for layer_idx, (key, value, beacon_size, raw_size_to_cache, window_size) in enumerate(past_key_values):
            previous_beacon_key, previous_beacon_value = self.beacon_activations[layer_idx]
            # previous_raw_key, previous_raw_value = self.raw_activations[layer_idx]
            #lzy: store past key/value to accelerate inference
            assert beacon_size == self.beacon_size
            if not self.training:
                if not first_time_inferenece:
                    device = self.raw_activations[chunk_idx][layer_idx][0].device
                    self.raw_activations[chunk_idx][layer_idx] = (cat_tensor([self.raw_activations[chunk_idx][layer_idx][0],
                                                                             slice_tensor(key,start = -beacon_size-1, end=-beacon_size, dim=self.k_seq_dim).to(device)],dim=self.k_seq_dim),
                                                                    cat_tensor([self.raw_activations[chunk_idx][layer_idx][1], 
                                                                                slice_tensor(value, start = -beacon_size-1, end = -beacon_size, dim=self.k_seq_dim).to(device)], dim=self.k_seq_dim))
                else:
                    stable_size = False 
                    available_device = self.find_available_device()
                    if stable_size:
                        self.raw_activations[chunk_idx][layer_idx]= (slice_tensor(key, start = 1, end=-beacon_size, dim=self.k_seq_dim).to(torch.device(available_device)), slice_tensor(value,  start = 1, end = -beacon_size, dim=self.k_seq_dim).to(torch.device(available_device)))
                    else:
                        self.raw_activations[chunk_idx][layer_idx]= (slice_tensor(key, end=-beacon_size, dim=self.k_seq_dim).to(torch.device(available_device)), slice_tensor(value, end = -beacon_size, dim=self.k_seq_dim).to(torch.device(available_device)))
                        
            reserve_size = 0
            beacon_key = cat_tensor([
                previous_beacon_key,
                slice_tensor(key, start=-(reserve_size+beacon_size), dim=self.k_seq_dim)
            ], dim=self.k_seq_dim)
            beacon_value = cat_tensor([
                previous_beacon_value,
                slice_tensor(value, start=-(reserve_size+beacon_size), dim=self.v_seq_dim)
            ], dim=self.v_seq_dim)
            
            self.beacon_activations[layer_idx] = (beacon_key, beacon_value)

    def update_loss(self, batch_loss, valid_token_num):
        """
        Accumulate loss for later perplexity computation and backward pass; past_key_values according to cache_method.
        """
        if self.batch_loss is None:
            # NOTE: multiply valid_token_num because batch_loss is divided by it in advance
            self.batch_loss = batch_loss * valid_token_num
            self.valid_token_num = valid_token_num
        else:
            # NOTE: avoid in-place operations, otherwise there will be gradient errors in training
            self.batch_loss = self.batch_loss + batch_loss * valid_token_num
            self.valid_token_num = self.valid_token_num + valid_token_num

    def output(self, model_outputs):
        """
        Override loss with accumulated loss.
        """
        # override loss
        if self.batch_loss is not None:
            # here the batch_loss is the summation of all token losses in each element
            loss = self.batch_loss.sum() / self.valid_token_num.sum()

            # NOTE: prevent nan
            batch_loss = self.batch_loss / self.valid_token_num
            if (self.valid_token_num == 0).any():
                batch_loss = batch_loss.masked_fill(self.valid_token_num == 0, 0.)

            # NOTE: we must use dict to override values, otherwise trainer cannot find loss
            model_outputs["loss"] = loss
            model_outputs["batch_loss"] = batch_loss
            model_outputs["valid_token_num"] = self.valid_token_num

        return model_outputs


def slice_tensor(x, start=None, end=None, dim=2):
    if x is None:
        return None
    if end == 0:
        return None
    if start == x.shape[dim]:
        return None
    if start == end:
        return None
    if dim == 2:
        if start is None and end is not None:
            return x[:, :, :end, ...]
        elif start is not None and end is None:
            return x[:, :, start:, ...]
        elif start is not None and end is not None:
            return x[:, :, start:end, ...]
    elif dim == 1:
        if start is None and end is not None:
            return x[:, :end, ...]
        elif start is not None and end is None:
            return x[:, start:, ...]
        elif start is not None and end is not None:
            return x[:, start:end, ...]
    else:
        raise NotImplementedError

def cat_tensor(list_of_tensors, dim=-1):
    list_of_tensors = [t for t in list_of_tensors if t is not None]
    if len(list_of_tensors) > 1:
        result = torch.cat(list_of_tensors, dim=dim)
    elif len(list_of_tensors) == 1:
        result = list_of_tensors[0]
    else:
        result = None
    return result

def softmax(x:np.ndarray, axis=-1, temperature=1):
    if isinstance(x, list):
        x = np.array(x)
    x = x / temperature
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def l1_norm(x):
    sum_x = sum(x)
    x = [y/sum_x for y in x]
    return x
