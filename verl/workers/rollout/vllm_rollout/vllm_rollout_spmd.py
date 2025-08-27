# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank
  to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""

import logging
import os
import pickle
import socket
import threading
from contextlib import contextmanager
from copy import deepcopy
from types import MethodType
from typing import Any, Dict, List, Union

import numpy as np
import ray
import torch
import torch.distributed
import zmq
from filelock import FileLock
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from vllm import LLM, SamplingParams
from vllm.distributed import parallel_state as vllm_ps
from vllm.lora.request import LoRARequest
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.worker.worker_base import WorkerWrapperBase

from verl import DataProto
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel
import re
import json
class checkresponseType(BaseModel):
    the_first_wrong_step_number: int
    
    
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# TODO
# 1. support pp in vllm
# 2. passing tokenizer is not necessary? no encoding/decoding is happending here
# 3. simplify init logics


# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id
    # is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


class vLLMRollout(BaseRollout):
    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
            model_hf_config: the huggingface config to initiallize the generating model in vllm
            **kwargs: train_tp, for Megatron Backend to initialize hybrid engine (zero redundancy) process group
        """
        super().__init__()
        self.config = config

        tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert tensor_parallel_size <= torch.distributed.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        max_num_batched_tokens = self.config.get("max_num_batched_tokens", 8192)

        if kwargs.get("train_tp") is not None:
            # deployed with megatron
            import os

            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            vllm_ps.initialize_model_parallel(tensor_model_parallel_size=tensor_parallel_size)

        rope_scaling_config = getattr(model_hf_config, "rope_scaling", None)
        if not rope_scaling_config:
            max_position_embeddings = None
            if hasattr(model_hf_config, "max_position_embeddings"):
                max_position_embeddings = model_hf_config.max_position_embeddings
            elif hasattr(model_hf_config, "llm_config") and hasattr(
                model_hf_config.llm_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.llm_config.max_position_embeddings
            elif hasattr(model_hf_config, "text_config") and hasattr(
                model_hf_config.text_config, "max_position_embeddings"
            ):
                max_position_embeddings = model_hf_config.text_config.max_position_embeddings
            if max_position_embeddings is None:
                raise ValueError("max_position_embeddings not found in model_hf_config")
            assert max_position_embeddings >= config.prompt_length + config.response_length, (
                "model context length should be greater than total sequence length"
            )
        else:
            # handle type where there's a length extend factor
            # see https://qwen.readthedocs.io/en/latest/deployment/vllm.html#extended-context-support
            # for using yarn as an example
            rope_scaling_factor = rope_scaling_config.get("factor", 1.0)

            assert (
                model_hf_config.max_position_embeddings * rope_scaling_factor
                >= config.prompt_length + config.response_length
            ), (
                "model context length should be greater than total sequence length, "
                + f"got rope_scaling_factor={rope_scaling_factor} and "
                + f"max_position_embeddings={model_hf_config.max_position_embeddings}"
            )

        max_model_len = int(config.max_model_len or config.prompt_length + config.response_length * 2 + 1024)

        if max_num_batched_tokens < max_model_len and self.config.enable_chunked_prefill:
            raise ValueError(
                "Enable chunked prefill, max_num_batched_tokens is smaller than max_model_len, \
                             please increase max_num_batched_tokens or disable chunked prefill"
            )

        trust_remote_code = kwargs.get("trust_remote_code", False)
        load_format = "dummy" if config.load_format.startswith("dummy") else config.load_format

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs
        # copy it to avoid secretly modifying the engine config
        engine_kwargs = (
            {}
            if "engine_kwargs" not in config or "vllm" not in config.engine_kwargs
            else OmegaConf.to_container(deepcopy(config.engine_kwargs.vllm))
        )
        # For each vLLM engine parameter,
        # - `None` means not setting it, so we pop it, and leave it to vLLM default value
        #    (which can vary across different vLLM versions);
        # - Otherwise it's the desired value we want to explicitly set.
        engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}
        if config.get("limit_images", None):  # support for multi-image data
            engine_kwargs["limit_mm_per_prompt"] = {"image": config.get("limit_images")}

        self.inference_engine = LLM(
            model=model_path,
            enable_sleep_mode=config.free_cache_engine,
            tensor_parallel_size=tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            dtype=config.dtype,
            enforce_eager=config.enforce_eager,
            gpu_memory_utilization=config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            skip_tokenizer_init=False,
            max_model_len=max_model_len,
            load_format=load_format,
            disable_log_stats=config.disable_log_stats,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_prefix_caching=False,
            trust_remote_code=trust_remote_code,
            seed=config.get("seed", 0),
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        if config.free_cache_engine:
            self.inference_engine.sleep(level=1)

        kwargs = dict(
            n=1,
            logprobs=0,  # can be set to 0 and let actor to recompute
            max_tokens=config.response_length,
        )

        kwargs["detokenize"] = False

        # supporting adding any sampling params from the config file
        for k in config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = config.get(k)
        kwargs["n"] = 1  # already repeat in ray_trainer
        print(f"kwargs: {kwargs}")
        self.sampling_params = SamplingParams(**kwargs)
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)
        yield
        # roll back to previous sampling params
        # if len(old_sampling_params_args):
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @GPUMemoryLogger(role="vllm rollout spmd", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generate sequences for a batch of prompts.

        Args:
            batch (DataProto): Input batch.

        Returns:
            DataProto: Output batch.
            - prompts: [bsz, prompt_length], prompt token ids from dataset.
            - responses: [bsz, response_length], output token ids include response tokens
              from LLM generation and observation tokens from tool_calls.
            - response_mask: [bsz, response_length], 1 for LLM generated tokens, 0 for observation/padding tokens.
            - input_ids: [bsz, prompt_length + response_length], whole sequence token ids, including prompt tokens
              and response tokens.
            - attention_mask: [bsz, prompt_length + response_length], 0 for padding tokens, 1 for other tokens.
            - position_ids: [bsz, prompt_length + response_length], incremental position ids.

            For multi-turn conversations:
            responses:     |<- LLM generation ->|<- tool_calls ->|<- LLM generation ->|<- padding ->|
            response_mask: | 1, 1, 1, ..., 1, 1 | 0, 0, .., 0, 0 | 1, 1, 1, ..., 1, 1 | 0, 0, ..., 0|
        """
        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        # left-padded attention_mask
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to construct attention_mask
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object
            )

        if batch_size != len(non_tensor_batch["raw_prompt_ids"]):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if "multi_modal_data" in non_tensor_batch:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"), non_tensor_batch.pop("multi_modal_data")
            ):
                vllm_inputs.append({"prompt_token_ids": raw_prompt_ids, "multi_modal_data": multi_modal_data})
        else:
            vllm_inputs = [
                {"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")
            ]

        # ensure the type of `prompt_token_ids` passed to vllm is list[int]
        # https://github.com/volcengine/verl/pull/772
        for input_data in vllm_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}"
                )

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = {
                "best_of": 1,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0.0,
                "temperature": 0,
                "n": 1,  # if greedy, only 1 response
            }
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,  # because we have already convert it to prompt token id
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=False,
            )

            # TODO(sgm): disable logprob when recompute_log_prob is enable
            # if n = 1: (bs, response_length) ; if n > 1: (bs * n, response_length)

            response = []
            rollout_log_probs = []
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    response_ids = output.outputs[sample_id].token_ids
                    response.append(response_ids)
                    if self.config.calculate_log_probs:
                        curr_log_prob = []
                        for i, logprob in enumerate(output.outputs[sample_id].logprobs):
                            curr_log_prob.append(logprob[response_ids[i]].logprob)
                        rollout_log_probs.append(curr_log_prob)

            response = pad_2d_list_to_length(response, self.pad_token_id, max_length=self.config.response_length).to(
                idx.device
            )
            if self.config.calculate_log_probs:
                rollout_log_probs = pad_2d_list_to_length(
                    rollout_log_probs, -1, max_length=self.config.response_length
                ).to(idx.device)
                rollout_log_probs = rollout_log_probs.to(torch.float32)

            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if self.config.calculate_log_probs:
            # we will recompute old log prob with actor
            batch["rollout_log_probs"] = rollout_log_probs


        ### print uniformly sampled 10 responses
        print("================================================")
        sampled_indices = torch.randint(0, batch_size, (10,))
        for i in sampled_indices:
            print(f"response {i}: {self.tokenizer.decode(response[i], skip_special_tokens=True)}")
            print("*****************************************************************************")
        # print("================================================")
        
        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
    
    
    @GPUMemoryLogger(role="vllm rollout spmd check_sequences", logger=logger)
    @torch.no_grad()
    def check_sequences(self, prompts: DataProto, tokenizer) -> DataProto:
        # breakpoint()        
        metrics = {}
        response_check_prompt = prompts.meta_info['response_check_prompt']
        questions = prompts.batch['prompts']
        responses = prompts.batch['responses']
        correctness = prompts.batch['token_level_scores']
        response_mask = prompts.batch['response_mask']
        
        assert response_mask.shape == responses.shape

        questions_ids = None
        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:    
            questions_ids = np.array(
                [_pre_process_inputs(self.pad_token_id, questions[i]) for i in range(questions.size(0))], dtype=object)
        else:
            questions_ids = non_tensor_batch['raw_prompt_ids']
        
        correct_mask = torch.sum(correctness * response_mask, dim=-1) # (bs,) 
        correct_mask = (correct_mask > 0.5)  # (True, False, True, False)
        assert len(questions_ids) == questions.size(0)
        
        num_of_responses = self.config.n

        batch_size = responses.size(0)
        num_of_questions = batch_size // num_of_responses
        assert num_of_questions * num_of_responses == batch_size
        token_level_mask = response_mask.detach().clone()
        
        vllm_inputs = []
        vllm_inputs_idxs = []
        response_step_maps = {}
        
        all_correct_question_num = 0
        all_wrong_question_num = 0
        correct_and_wrong_question_num = 0
        correct_response_num = 0
        
        for i in range(num_of_questions):
            start_idx = i * num_of_responses
            end_idx = (i + 1) * num_of_responses
            question = questions_ids[start_idx]
            question_correct_mask = correct_mask[start_idx:end_idx]
            
            # 找到正确的参考答案
            correct_indices = torch.where(question_correct_mask)[0] 
            wrong_indices = torch.where(~question_correct_mask)[0]
            
            correct_response_num += correct_indices.numel()
            
            if correct_indices.numel() == 0:
                all_wrong_question_num += 1
                continue
        
            if wrong_indices.numel() == 0:
                all_correct_question_num += 1
                continue
            
            correct_and_wrong_question_num += 1
            
            ref_response_idx = correct_indices[0] + start_idx
            ref_response = responses[ref_response_idx]
            
            # 处理每个错误的响应
            for wrong_idx in wrong_indices:
                actual_idx = wrong_idx + start_idx
                vllm_inputs_idxs.append(actual_idx)
                candidate_response = responses[actual_idx]
                
                # 准备分析提示，将候选响应和参考响应拆分为步骤
                formatted_candidate, candidate_steps, candidate_step_map = self._split_into_steps(candidate_response, tokenizer)
                
                # 存储步骤映射以便稍后查找
                response_step_maps[actual_idx] = candidate_step_map
                
                # 准备输入
                vllm_inputs.append({
                    'prompt_token_ids': self._prepare_analysis_prompt(tokenizer, question, formatted_candidate, ref_response, response_check_prompt)
                })
        metrics['check_sequences_statistics/acc_in_batch'] = correct_response_num / (num_of_responses * num_of_questions)
        metrics['check_sequences_statistics/all_wrong_question_num'] = all_wrong_question_num
        metrics['check_sequences_statistics/all_correct_question_num'] = all_correct_question_num
        metrics['check_sequences_statistics/correct_and_wrong_question_num'] = correct_and_wrong_question_num
        metrics['check_sequences_statistics/vllm_inputs_length'] = len(vllm_inputs)


        
        for input_data in vllm_inputs:
            if isinstance(input_data['prompt_token_ids'], np.ndarray):
                input_data['prompt_token_ids'] = input_data['prompt_token_ids'].tolist()
            elif not isinstance(input_data['prompt_token_ids'], list):
                raise TypeError(
                    f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        check_response = []
        json_schema = checkresponseType.model_json_schema()
        guided_decoding_params_json = GuidedDecodingParams(json=json_schema)
        kwargs = {
            'top_k': self.config.val_kwargs.top_k,
            'top_p': self.config.val_kwargs.top_p,
            'temperature': self.config.val_kwargs.temperature,
            'n': 1,  # if validate, already repeat in ray_trainer
            'min_tokens': 2,
            'guided_decoding': guided_decoding_params_json,
        }
        # kwargs = {
        #     'top_k': self.config.val_kwargs.top_k,
        #     'top_p': self.config.val_kwargs.top_p,
        #     'temperature': self.config.val_kwargs.temperature,
        #     'n': 1,  # if validate, already repeat in ray_trainer
        #     'min_tokens': 2,
        # }
        # max_token_id = max(self.tokenizer.get_vocab().values())
        # kwargs["allowed_token_ids"] = list(range(max_token_id + 1))

        if len(vllm_inputs) > 0:
            print(f"vllm_inputs[0]: {tokenizer.decode(vllm_inputs[0]['prompt_token_ids'], skip_special_tokens=True)}")
        
        
        with self.update_sampling_params(**kwargs):
            outputs = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )
            for output in outputs:
                for sample_id in range(len(output.outputs)):
                    check_response.append(output.outputs[sample_id].token_ids)
                    
        incorrect_token_level_mask, vllm_infer_success_rate, match_success_rate, no_exist_step_rate, error_ratio_in_response = self.get_incorrect_token_level_mask(
            check_response, 
            vllm_inputs_idxs, 
            tokenizer, 
            responses.shape[-1], 
            responses,
            response_step_maps,
            response_mask
        )
        metrics['check_sequences_statistics/vllm_infer_success_rate'] = vllm_infer_success_rate
        metrics['check_sequences_statistics/match_success_rate'] = match_success_rate
        metrics['check_sequences_statistics/no_exist_step_rate'] = no_exist_step_rate
        metrics['check_sequences_statistics/error_ratio_in_response'] = error_ratio_in_response
        
        for i in range(len(vllm_inputs_idxs)):
            token_level_mask[vllm_inputs_idxs[i]] = incorrect_token_level_mask[i]

        # 计算同时拥有正确和错误答案的问题中正负值的比例
        # 正值表示错误的token（需要纠正），负值表示正确的token
        positive_negative_ratio = self._calculate_positive_negative_ratio(
            token_level_mask, correct_mask, num_of_questions, num_of_responses
        )
        metrics['check_sequences_statistics/positive_negative_ratio_mixed_questions'] = positive_negative_ratio

        # 更新数据
        prompts.batch['token_level_mask_by_llm'] = token_level_mask
        prompts.meta_info['check_sequences_statistics'] = metrics
        
        # breakpoint()
        # 释放vllm缓存引擎
        print("================================================================================")
        return prompts

    def _split_into_steps(self, response_tokens, tokenizer):
        """将响应拆分为步骤并添加步骤标识"""
        # 将token ID转换为文本
        response_text = tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # 检测并删除step标识的模式（匹配行首的step标识）
        step_patterns = [
            r'^step\s*\d+[:\s]*',      # step 1:, step1:, step 1 , step1 
            r'^Step\s*\d+[:\s]*',      # Step 1:, Step1:, Step 1 , Step1
            r'^STEP\s*\d+[:\s]*',      # STEP 1:, STEP1:, STEP 1 , STEP1
            r'^第\s*\d+\s*步[:\s]*',   # 第1步:, 第1步 , 第1步
            r'^步骤\s*\d+[:\s]*',      # 步骤1:, 步骤1 , 步骤1
        ]
        
        # 直接处理整个response_text，删除每行开头的step标识
        cleaned_response_text = response_text
        for pattern in step_patterns:
            # 使用re.MULTILINE标志，让^匹配每行的开头
            cleaned_response_text = re.sub(pattern, '', cleaned_response_text, flags=re.IGNORECASE | re.MULTILINE)
        
        # 按行切分
        response_text_steps = cleaned_response_text.split("\n", maxsplit=199)

        steps = []
        step_map = {}  # 映射步骤编号到原始文本
        formatted_steps = []
        
        cnt = 1
        for sentence in response_text_steps:
            sentence = sentence.strip()
            if not sentence:  # 跳过空行
                continue
                
            step_id = f"Step{cnt}"
            step_map[step_id] = sentence
            steps.append((step_id, sentence))
            formatted_steps.append(f"{step_id}: {sentence}")
            cnt += 1
        
        # 将格式化的步骤组合成文本
        formatted_text = "\n".join(formatted_steps)
        formatted_tokens = tokenizer.encode(formatted_text, add_special_tokens=False)
        
        return formatted_tokens, steps, step_map




    def _prepare_analysis_prompt(self, tokenizer, question, formatted_candidate, formatted_ref, response_check_prompt):
        # breakpoint()
        question_start = len(tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nCutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n", add_special_tokens=False))
        question_end = len(tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False))
        question = question[question_start:-question_end]
        
        if isinstance(question, torch.Tensor):
            question = question.tolist()
            
        if isinstance(formatted_ref, torch.Tensor):
            formatted_ref = formatted_ref.tolist()
        
        # 过滤掉填充标记
        question = [t for t in question if t != self.pad_token_id]
        formatted_ref = [t for t in formatted_ref if t != self.pad_token_id]
        # formatted_ref = formatted_ref[:-1] 
        formatted_ref = formatted_ref
        
        prompt = [
            *response_check_prompt['bg_guide'],
            *response_check_prompt['question_guide'],
            *question,
            *response_check_prompt['correct_answer'],
            *formatted_ref,
            *response_check_prompt['candidate_answer'],
            *formatted_candidate,
            *response_check_prompt['end_str'],
        ]
        # breakpoint()
        return prompt
    
    def _extract_incorrect_steps(self, analysis_text):     
        # print("********************************************")
        # print(analysis_text)
        # print("********************************************")
        match = re.search(r'{.*?}', analysis_text, re.DOTALL)
        if match is None:
            # print(f"Error: No JSON format found in response: {analysis_text[:100]}...")
            return False, []
        eval_result_json_str = match.group(0)  # Get the full match including braces
        try:
            eval_result = json.loads(eval_result_json_str)  # Use json.loads instead of ast.literal_eval
            if "the_first_wrong_step_number" not in eval_result:
                return False, []
            pred_wrong_step_value = eval_result["the_first_wrong_step_number"]
            try:
                wrong_step_num = int(pred_wrong_step_value)
                # print(f"Parsed step number: {wrong_step_num}")
                if wrong_step_num == -1:
                    # print("All steps are correct (step number = -1)")
                    return True, []
                elif wrong_step_num > 0:
                    step_id = f"Step{wrong_step_num}"
                    # print(f"First incorrect step: {step_id}")
                    return True, [step_id]
                else:
                    print(f"Error: Invalid step number (must be positive or -1): {wrong_step_num}")
                    return False, []
            except ValueError as e:
                print(f"Error: Cannot parse boxed content as integer: '{pred_wrong_step_value}' - {e}")
                return False, []
            
        except Exception as e:
            print(f"Error: {e}")
            return False, []
            


    
    def get_incorrect_token_level_mask(self, check_response, vllm_inputs_idxs, tokenizer, gen_length, responses, response_step_maps, response_mask):
        """获取基于步骤的错误标记 - 考虑response_mask的版本"""
        token_level_mask = torch.ones((len(check_response), gen_length))
        
        check_response_text = [tokenizer.decode(r, skip_special_tokens=True) for r in check_response]
        
        assert len(check_response_text) == len(vllm_inputs_idxs)
        
        vllm_infer_success_count = 0
        match_count = 0
        match_success_count = 0
        no_exist_step_count = 0
        error_ratio_in_response = 0
        
        
        for i in range(len(check_response_text)):
            response_idx = vllm_inputs_idxs[i]
            token_level_mask[i] = response_mask[response_idx].clone()
            flag, incorrect_step = self._extract_incorrect_steps(check_response_text[i])
            if flag:
                vllm_infer_success_count += 1
                if incorrect_step:
                    # breakpoint()
                    match_count += 1
                    step_map = response_step_maps[response_idx]
                    response = responses[response_idx]
                    
                    # 使用response_mask来确定有效token的范围
                    current_response_mask = response_mask[response_idx]
                    valid_positions = torch.where(current_response_mask == 1)[0]
                    
                    if len(valid_positions) == 0:
                        print(f"Error: No valid positions found in response: {check_response_text[i]}")
                        continue
                        
                    valid_start = valid_positions[0].item()
                    valid_end = valid_positions[-1].item() + 1
                    
                    # 获取有效token
                    valid_tokens = response[valid_start:valid_end].tolist()
                    valid_tokens = [t for t in valid_tokens if t != self.pad_token_id] # tokens
                    
                    # 初始化mask为0
                    mask = torch.zeros(gen_length, dtype=torch.float)
                    
                    
                    # if incorrect_step == [-1]:
                    #     mask[:valid_end] = -1
                    #     match_success_count += 1
                    #     error_ratio_in_response += (valid_end - valid_start) / (valid_end - valid_start)
                    # else:
                    for step_id in incorrect_step:
                        if step_id in step_map:
                            error_text = step_map[step_id]
                            response_text = tokenizer.decode(valid_tokens, skip_special_tokens=True) # text
                            if error_text in response_text:
                                error_pos = response_text.find(error_text)
                                if error_pos != -1:
                                    prefix_text = response_text[:error_pos]
                                    prefix_tokens = tokenizer.encode(prefix_text, add_special_tokens=False)
                                    
                                    # 计算在原始序列中的位置 TODO:判断valid start 是不是为0
                                    start_pos = valid_start + len(prefix_tokens)
                                    start_pos = max(valid_start, min(start_pos, valid_end))
                                    
                                    # 只对有效范围内的token设置mask
                                    mask[:start_pos] = -1
                                    mask[start_pos:valid_end] = 1 * 4
                                    match_success_count += 1
                                    error_ratio_in_response += (start_pos - valid_start) / (valid_end - valid_start)
                                    break
                            else:
                                print(f"Error: {error_text} not found in response: {response_text}")
                        else:
                            no_exist_step_count += 1
                            print(f"Error: {step_id} not found in step_map")
                            
                    token_level_mask[i] = mask
        
        vllm_infer_success_rate = vllm_infer_success_count / len(check_response_text) if check_response_text else 0
        match_success_rate = match_success_count / match_count if match_count > 0 else 0
        no_exist_step_rate = no_exist_step_count / match_count if match_count > 0 else 0
        error_ratio_in_response = error_ratio_in_response / match_success_count if match_success_count > 0 else 0
        return token_level_mask, vllm_infer_success_rate, match_success_rate, no_exist_step_rate, error_ratio_in_response

    def _calculate_positive_negative_ratio(self, token_level_mask, correct_mask, num_of_questions, num_of_responses):
        """
        计算同时拥有正确和错误答案的问题中，所有回答的token_level_mask_by_llm中正值和负值的比例
        
        Args:
            token_level_mask: [batch_size, gen_length] 的mask，包含正值和负值
            correct_mask: [batch_size] 的布尔mask，True表示正确答案
            num_of_questions: 问题数量
            num_of_responses: 每个问题的回答数量
            
        Returns:
            float: 正值数量 / 负值数量的比例，如果没有负值则返回-1，如果没有正值则返回0
        """
        positive_count = 0
        negative_count = 0
        mixed_questions_count = 0
        
        for i in range(num_of_questions):
            start_idx = i * num_of_responses
            end_idx = (i + 1) * num_of_responses
            question_correct_mask = correct_mask[start_idx:end_idx]
            
            # 检查是否同时有正确和错误的答案
            has_correct = torch.any(question_correct_mask)
            has_wrong = torch.any(~question_correct_mask)
            
            if has_correct and has_wrong:
                mixed_questions_count += 1
                # 获取该问题的所有回答的token_level_mask
                question_masks = token_level_mask[start_idx:end_idx]  # [num_of_responses, gen_length]
                
                # 计算正值和负值的数量
                positive_tokens = torch.sum(question_masks > 0).item()
                negative_tokens = torch.sum(question_masks < 0).item()
                
                positive_count += positive_tokens
                negative_count += negative_tokens
        
        print(f"Mixed questions count: {mixed_questions_count}, positive tokens: {positive_count}, negative tokens: {negative_count}")
        
        # 计算比例
        if negative_count == 0:
            return 0.0 if positive_count == 0 else float('inf')
        else:
            return positive_count / negative_count

    


# https://github.com/vllm-project/vllm/issues/13175
def _monkey_patch_compute_logits(model, vocab_size: int):
    original_compute_logits = model.compute_logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = original_compute_logits(hidden_states, sampling_metadata)
        logits[..., vocab_size:] = float("-inf")
        return logits

    model.compute_logits = MethodType(compute_logits, model)


class vLLMAsyncRollout:
    """vLLMAsyncRollout is a thin wrapper of WorkerWrapperBase,
    which is engine in single worker process.
    """

    def __init__(self, model_path: str, config: DictConfig, tokenizer, model_hf_config, **kwargs):
        self.tokenizer = tokenizer

        # Engine is deferred to be initialized in init_worker
        self.config = config
        self.inference_engine: WorkerWrapperBase = None
        self.sharding_manager = None
        self.is_sleep = False
        self.address = self._init_zeromq()

    def _init_zeromq(self) -> str:
        tensor_parallel_size = self.config.tensor_model_parallel_size

        # single node: ipc, multi nodes: tcp
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        socket_type = "ipc" if tensor_parallel_size <= local_world_size else "tcp"

        # File lock to prevent multiple workers listen to same port
        with FileLock("/tmp/verl_vllm_zmq.lock"):
            if socket_type == "ipc":
                pid = os.getpid()
                address = f"ipc:///tmp/verl_vllm_zmq_{pid}.ipc"
            else:
                ip, port = self._get_free_port()
                address = f"tcp://{ip}:{port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.REP)
            self.socket.bind(address)

        self.loop_thread = threading.Thread(target=self._loop_forever)
        self.loop_thread.start()

        return address

    def _get_free_port(self):
        ip = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return ip, port

    def _loop_forever(self):
        while True:
            message = self.socket.recv()
            method, args, kwargs = pickle.loads(message)
            result = self.execute_method(method, *args, **kwargs)
            self.socket.send(pickle.dumps(result))

    def get_zeromq_address(self):
        return self.address

    def init_worker(self, all_kwargs: List[Dict[str, Any]]):
        """Initialize worker engine."""
        all_kwargs[0]["rank"] = int(os.environ["RANK"])
        all_kwargs[0]["local_rank"] = 0

        self.vllm_config = all_kwargs[0]["vllm_config"]
        self.inference_engine = WorkerWrapperBase(vllm_config=self.vllm_config)
        self.inference_engine.init_worker(all_kwargs)

    def load_model(self, *args, **kwargs):
        self.inference_engine.load_model(*args, **kwargs)

        # inference engine is initialized now, update sharding manager
        self.sharding_manager.inference_engine = self.inference_engine
        self.sharding_manager.model_runner = self.inference_engine.worker.model_runner

        _monkey_patch_compute_logits(self.inference_engine.worker.model_runner.model, len(self.tokenizer))

    def sleep(self, *args, **kwargs):
        """Offload model weights and discard kv cache."""
        if self.is_sleep:
            return
        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True

    def wake_up(self, *args, **kwargs):
        """Load model weights and build kv cache."""
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801
        self.is_sleep = False

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "init_worker":
            return self.init_worker(*args, **kwargs)
        elif method == "load_model":
            return self.load_model(*args, **kwargs)
        elif method == "sleep":
            return self.sleep(*args, **kwargs)
        elif method == "wake_up":
            return self.wake_up(*args, **kwargs)
        else:
            return self.inference_engine.execute_method(method, *args, **kwargs)
