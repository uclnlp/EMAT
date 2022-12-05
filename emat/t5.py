# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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
""" PyTorch T5 model. """
import asyncio
import copy
import warnings
from concurrent import futures
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from emat.fusion_net import FusionWeight

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.file_utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
    ModelOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerNorm,
    T5Stack,
    T5ForConditionalGeneration,
    T5_START_DOCSTRING,
    T5_INPUTS_DOCSTRING,
    __HEAD_MASK_WARNING_MSG as HEAD_MASK_WARNING_MSG,
    _CONFIG_FOR_DOC,
)
from transformers.utils import logging
from utils.utils import reduce_query_or_key_embeds
from emat.retriever.utils import mips
from emat.retrieval_adapter import RetAdapter

logger = logging.get_logger(__name__)


@dataclass
class KeyValueOutput(ModelOutput):
    key: torch.FloatTensor = None
    value: Optional[torch.FloatTensor] = None


@dataclass
class CATEncoderOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    query_embeds: Optional[torch.tensor] = None
    readout_indices: Optional[Any] = None
    updated_attention_mask: Optional[torch.tensor] = None


@dataclass
class CATSeq2SeqLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cat_encoder_outputs: Optional[CATEncoderOutput] = None


class ConvKeyEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv_layer = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            stride=1,
            padding="same",
            bias=False
        )
        self.relu = nn.ReLU()
        self.linear = nn.Linear(config.d_model, config.d_key, bias=False)

    def forward(self, hidden_states, attention_mask):
        attention_mask = attention_mask[:, None, :]
        masked_hidden = hidden_states.transpose(1, 2) * attention_mask  # shape: [batch_size, d_model, seq_length]

        conv_out = self.conv_layer(masked_hidden)  # shape: [batch_size, d_model, seq_length]
        relu_out = self.relu(conv_out)  # shape: [batch_size, d_model, seq_length]
        relu_out = relu_out * attention_mask  # mask again

        sum_pooled = torch.sum(relu_out, dim=2)  # shape: [batch_size, d_model]
        lengths = torch.sum(attention_mask, dim=(1, 2))  # shape: [batch_size]
        mean_pooled = sum_pooled / lengths[:, None]  # shape: [batch_size, d_model]

        final_out = self.linear(mean_pooled)  # shape: [batch_size, d_key]
        return final_out


# T5StackWithKeyValueMemory is encoder
class T5StackWithKeyValueMemory(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super(T5Stack, self).__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        # Create prefix embeddings
        assert not self.is_decoder, "This module should only be used as encoder"
        self.prefix_length = config.prefix_length  # length of the prefix
        self.prefix_embedding = nn.Parameter(torch.empty((self.prefix_length, config.d_model), dtype=torch.float))
        self.model_dim = config.d_model

        # Key configs
        self.key_layer = config.key_layer  # the layer that conducts key querying

        self.cat_layer = getattr(config, "cat_layer", None)

        # Initialize the key encoder
        self.key_encoder_type = config.key_encoder_type
        if self.key_encoder_type == "linear":
            self.d_key = config.d_key  # dimension of the key/query embedding
            self.key_encoder = nn.Linear(self.prefix_length * config.d_model, self.d_key, bias=False)
        elif self.key_encoder_type == "conv":
            self.key_encoder = ConvKeyEncoder(config)
        elif self.key_encoder_type == "prefix":
            self.key_encoder = None
        else:
            raise ValueError(f"Incorrect key_encoder_type: {self.key_encoder_type}")

        # self.qk_scorer = nn.Linear(1, 1, bias=True)  # calibrate the query-key match scores into gating

        # Value configs
        self.value_layer = config.value_layer  # the layer that it conducts value infilling
        self.num_values = config.num_values  # number of value embeddings to infill
        assert self.key_layer <= self.value_layer, "Key layer should be smaller than or equal to value layer"

        self.value_fusion_method = config.value_fusion_method

        if self.value_fusion_method is not None and "cat" in self.value_fusion_method:
            # add_position_bias_layer = self.value_layer
            # if "delay" in self.value_fusion_method:
            #     add_position_bias_layer = self.key_layer
            if self.cat_layer is not None:
                add_position_bias_layer = self.cat_layer
            else:
                add_position_bias_layer = min(self.key_layer, self.value_layer)
            self.block = nn.ModuleList(
                [T5Block(config, has_relative_attention_bias=bool(i == 0 or i == add_position_bias_layer))
                 for i in range(config.num_layers)]
            )
        else:
            self.block = nn.ModuleList(
                [T5Block(config, has_relative_attention_bias=bool(i == 0))
                 for i in range(config.num_layers)]
            )

        if self.value_fusion_method is not None and "g(" in self.value_fusion_method:
            self.fusion_weight_proj = FusionWeight(fusion_type=self.value_fusion_method)
        else:
            self.fusion_weight_proj = None

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.key_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        nn.init.normal_(self.prefix_embedding.data, mean=0.0, std=config.initializer_factor * 1.0)
        # self.qk_scorer.weight.data.copy_(torch.tensor([[1.0]]))
        # self.qk_scorer.bias.data.copy_(torch.tensor([0.0]))

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            key_embeds=None,
            value_embeds=None,
            key_faiss_index=None,
            key_reduce_method=None,
            value_qas_input_ids=None,
            value_qas_attention_mask=None,
            readout_top_k=1,
            value_fusion_method=None,
            key_embeds_of_value=None,
            key_memory=None,
            value_memory=None,
            embedding_index=None,
    ):
        assert key_embeds is None
        assert value_fusion_method == self.value_fusion_method
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Sanity check
        assert not use_cache, "This class does not support use_cache because it is encoder only"

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        # Concatenate the prefix embeddings and extend the attention masks
        prefix_embeds = self.prefix_embedding[None, :, :].expand(batch_size, -1, -1).to(inputs_embeds.device)
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        # Extend the attention masks
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length + self.prefix_length).to(inputs_embeds.device)
        else:
            prefix_mask = torch.ones((batch_size, self.prefix_length), dtype=attention_mask.dtype).to(
                inputs_embeds.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length + self.prefix_length), inputs_embeds.device
        )

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        query_embeds = None
        readout_indices = None

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if not use_cache:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            # Query-key matching
            if i == self.key_layer:  # key-layer is Query-Layer
                raw_query_embeds = self._encode_key(hidden_states, attention_mask)  # shape:[batch_size, d_key]
                raw_query_embeds = raw_query_embeds.view(hidden_states.shape[0], -1, hidden_states.shape[-1])

                normed_query_embeds = self.kv_output_layer(raw_query_embeds)  # Query is normed !!!
                query_embeds = reduce_query_or_key_embeds(normed_query_embeds, key_reduce_method)

                if embedding_index is not None:
                    assert value_embeds is None and key_embeds_of_value is None
                    if type(embedding_index) is not list:
                        embedding_index = [embedding_index]
                    half_query_embeds = query_embeds.half()
                    memory_size, hidden_num, hidden_size = value_memory.shape
                    if memory_size > 20000000:
                        # if scale is large: calculate topk in each chunk -> combine all-topk -> select final topk
                        chunk_top_scores = []
                        chunk_top_indices = []
                        idx_shift = 0
                        for chunk_key_memory in embedding_index:
                            chunk_key_memory_cuda = chunk_key_memory.cuda()
                            chunk_topk = torch.mm(half_query_embeds, chunk_key_memory_cuda.t()).topk(50, dim=1)
                            chunk_top_scores.append(chunk_topk.values)  # chunk_topk.scores: [query_batch, local_size]
                            chunk_top_indices.append(chunk_topk.indices + idx_shift)
                            idx_shift += len(chunk_key_memory)
                            del chunk_key_memory_cuda
                            torch.cuda.empty_cache()
                        chunk_top_scores = torch.cat(chunk_top_scores, dim=1)  # q_batch, local_size*seg_n
                        chunk_top_indices = torch.cat(chunk_top_indices, dim=1)  # q_batch, local_size*seg_n
                        topk = chunk_top_scores.topk(readout_top_k, dim=1)  # q_batch, local_size
                        top_indices_indices = topk.indices
                        readout_indices = []
                        for cur_indices_indices, cur_indices in zip(top_indices_indices, chunk_top_indices):
                            readout_indices.append([cur_indices[idx] for idx in cur_indices_indices])
                    else:
                        all_chunk_scores = []
                        for chunk_key_memory in embedding_index:
                            chunk_key_memory_cuda = chunk_key_memory.cuda()
                            chunk_scores = torch.mm(half_query_embeds, chunk_key_memory_cuda.t())  # query_batch
                            all_chunk_scores.append(chunk_scores)
                            del chunk_key_memory_cuda
                        scores = torch.cat(all_chunk_scores, dim=1)
                        readout_indices = scores.topk(readout_top_k, dim=1).indices.tolist()

                    top_indices = torch.tensor(readout_indices)
                    bs = input_ids.shape[0]
                    value_embeds = torch.index_select(value_memory, 0, top_indices.view(-1)).float().cuda()
                    value_embeds = value_embeds.view(input_ids.shape[0], readout_top_k, hidden_num, hidden_size)
                    key_embeds_of_value = torch.index_select(key_memory, 0, top_indices.view(-1)).float().cuda()
                    key_embeds_of_value = key_embeds_of_value.view(bs, readout_top_k, hidden_num, hidden_size)

                if value_fusion_method == "cat_k_delay+v":
                    # Serial mode, cat key directly.
                    batch_size, num_values, key_nums, hidden_size = key_embeds_of_value.shape
                    if key_nums != self.prefix_length:
                        assert key_nums == 1
                        key_embeds_of_value = key_embeds_of_value.repeat(1, 1, self.prefix_length, 1)
                    hidden_states = torch.cat(
                        [key_embeds_of_value.view(batch_size, num_values * self.prefix_length, hidden_size),
                         hidden_states], dim=1
                    )
                    extend_length = num_values * self.prefix_length
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule

            if self.cat_layer is not None and i == self.cat_layer:
                if value_fusion_method == "async_cat_k_delay+v":
                    # Async mode, emat key in cat_layer. the implementation of delay + v is same to serial mode.
                    batch_size, num_values, key_nums, hidden_size = key_embeds_of_value.shape
                    hidden_states = torch.cat(
                        [key_embeds_of_value.view(batch_size, num_values * self.prefix_length, hidden_size),
                         hidden_states], dim=1
                    )
                    extend_length = num_values * self.prefix_length
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule

            if i == self.value_layer:
                batch_size, num_values, _, hidden_size = key_embeds_of_value.shape

                # assert query_embeds is not None, "Use query_embeds to read memory before assignment."
                if "delay" in value_fusion_method:
                    updated_key = hidden_states[:, :num_values * self.prefix_length]
                    updated_key = updated_key.view(batch_size, num_values, self.prefix_length, hidden_size)
                else:
                    updated_key = None
                integrated_value = self.get_integrated_values(value_embeds, key_embeds_of_value, value_fusion_method,
                                                              query_embeds=query_embeds, updated_key=updated_key,
                                                              key_reduce_method=key_reduce_method)
                if value_fusion_method == "infill":
                    assert self.num_values == 1
                    hidden_states = torch.cat([integrated_value, hidden_states[:, self.prefix_length:]], dim=1)
                elif "cat" in value_fusion_method and "delay" in value_fusion_method:
                    hidden_states[:, :num_values * self.prefix_length] = integrated_value
                    hidden_states = hidden_states.contiguous()
                elif "cat" in value_fusion_method and "delay" not in value_fusion_method:
                    hidden_states = torch.cat([integrated_value, hidden_states], dim=1)
                    extend_length = integrated_value.shape[1]
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule
                else:
                    raise NotImplementedError(f"{value_fusion_method} is not defined.")

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    query_embeds,
                    readout_indices
                ]
                if v is not None
            )
        return CATEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            query_embeds=query_embeds,
            readout_indices=readout_indices,
            updated_attention_mask=None,
        )

    def get_integrated_values(self, group_value_embeds, group_key_embeds, value_fusion_method,
                              query_embeds=None, updated_key=None, key_reduce_method=None):
        # group_value_embeds: [batch_size, num_values, prefix_length, hidden_size]
        # group_key_embeds: [batch_size, num_values, key_num_tokens, hidden_size]
        if value_fusion_method == "cat_k_delay+v":
            batch_size, num_values, prefix_length, hidden_size = group_value_embeds.shape
            integrated_value = updated_key + group_value_embeds.contiguous()
            integrated_value = integrated_value.view(batch_size, num_values * prefix_length, hidden_size)
        elif value_fusion_method == "async_cat_k_delay+v":
            batch_size, num_values, prefix_length, hidden_size = group_value_embeds.shape
            integrated_value = updated_key + group_value_embeds.contiguous()
            integrated_value = integrated_value.view(batch_size, num_values * prefix_length, hidden_size)
        elif value_fusion_method == "cat_v":
            batch_size, num_values, prefix_length, hidden_size = group_value_embeds.shape
            group_value_embeds = group_value_embeds.contiguous()
            integrated_value = group_value_embeds.view(batch_size, num_values * prefix_length, hidden_size)
        elif value_fusion_method == "async_cat_k+v":
            batch_size, num_values, prefix_length, hidden_size = group_value_embeds.shape
            group_key_add_value = group_key_embeds + group_value_embeds
            integrated_value = group_key_add_value.view(batch_size, num_values * self.prefix_length, hidden_size)
        elif value_fusion_method == "cat_k+v":
            batch_size, num_values, prefix_length, hidden_size = group_value_embeds.shape
            group_key_add_value = group_key_embeds + group_value_embeds
            integrated_value = group_key_add_value.view(batch_size, num_values * self.prefix_length, hidden_size)
        elif value_fusion_method == "cat_k_delay+v_g(kv)":
            # batch_size, num_values, 1
            key_weight = self.fusion_weight_proj(key=updated_key, value=group_value_embeds.contiguous())
            key_weight = key_weight.unsqueeze(dim=-1)
            integrated_value = key_weight * updated_key + (1 - key_weight) * group_value_embeds
            batch_size, num_values, key_nums, hidden_size = updated_key.shape
            integrated_value = integrated_value.view(batch_size, num_values * self.prefix_length, hidden_size)
        elif value_fusion_method == "infill":
            batch_size, num_values, prefix_length, hidden_size = group_value_embeds.shape
            assert num_values == self.num_values == 1
            integrated_value = group_value_embeds.view(batch_size, prefix_length, hidden_size)
        elif value_fusion_method == "cat_kv":
            group_key_cat_value = torch.cat((group_key_embeds, group_value_embeds), dim=2)
            # [batch_size, num_values, prefix_length + key_num_tokens, hidden_size]
            batch_size, num_values, integrated_prefix_length, hidden_size = group_key_cat_value.shape
            integrated_value = group_key_cat_value.view(batch_size, num_values * integrated_prefix_length, hidden_size)
        elif value_fusion_method == "cat_avgk+v":
            batch_size, num_values, key_nums, hidden_size = group_key_embeds.shape
            reduced_key_embeds = (group_key_embeds.sum(dim=2) / key_nums).unsqueeze(dim=2)
            group_key_add_value = reduced_key_embeds + group_value_embeds
            integrated_value = group_key_add_value.view(batch_size, num_values * self.prefix_length, hidden_size)
        elif value_fusion_method == "cat_k+v_g(kq)":
            batch_size, num_values, key_nums, hidden_size = group_key_embeds.shape
            squeezed_key_embeds = group_key_embeds.view(batch_size * num_values, key_nums, hidden_size)
            reduced_key_embeds = reduce_query_or_key_embeds(squeezed_key_embeds, key_reduce_method)
            reduced_key_embeds = reduced_key_embeds.view(batch_size, num_values, hidden_size)
            key_weight = self.fusion_weight_proj(key=reduced_key_embeds, query=query_embeds)  # b_s, num_values, 1
            key_weight = key_weight.unsqueeze(dim=-1)
            integrated_value = key_weight * group_key_embeds + (1 - key_weight) * group_value_embeds
            integrated_value = integrated_value.view(batch_size, num_values * self.prefix_length, hidden_size)
        else:
            raise NotImplementedError(f"{value_fusion_method} is not defined.")
        return integrated_value

    def embed_kv(self, input_ids, attention_mask=None, head_mask=None, compute_key=True, compute_value=True,
                 embed_for_ae_task=False) -> Dict:
        """Compute the key/value embeddings for the input."""
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)

        # Sanity check
        assert compute_key or compute_value, "At least one of compute_key and compute_value needs to be True"
        assert not (compute_key and compute_value), "Only compute key or value once forward."
        assert input_ids is not None

        original_input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_ids.shape[-1])  # the last dimension is seq_length
        inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_ids.shape

        # Concatenate the prefix embeddings and extend the attention masks
        prefix_embeds = self.prefix_embedding[None, :, :].expand(batch_size, -1, -1).to(inputs_embeds.device)
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        # Extend the attention masks
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, seq_length + self.prefix_length).to(inputs_embeds.device)
        else:
            prefix_mask = torch.ones((batch_size, self.prefix_length), dtype=attention_mask.dtype).to(
                inputs_embeds.device)
            attention_mask = torch.cat([prefix_mask, attention_mask.view(batch_size, seq_length)], dim=1)

        # initialize past_key_values with `None` if past does not exist
        past_key_values = [None] * len(self.block)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length + self.prefix_length), inputs_embeds.device
        )

        encoder_hidden_states = None
        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(None, self.config.num_layers)
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        key_embeds, value_embeds = None, None
        normed_key_embeds, normed_value_embeds = None, None
        key_embeds_to_cat = None

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=False,
                output_attentions=False,
            )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            # Encode the key
            if compute_key and i == self.key_layer:
                key_embeds = self._encode_key(hidden_states, attention_mask)  # shape:[batch_size, d_key]
                key_embeds = key_embeds.view(hidden_states.shape[0], -1, hidden_states.shape[-1])
                if not compute_value and self.cat_layer is None:
                    break
            if compute_value and i == self.cat_layer:
                key_embeds_to_cat = self._encode_key(hidden_states, attention_mask)
                key_embeds_to_cat = key_embeds_to_cat.view(hidden_states.shape[0], -1, hidden_states.shape[-1])

            # Encode the value
            if compute_value and i == self.value_layer:
                value_embeds = hidden_states[:, :self.prefix_length]
                break  # (jimmycode): early stop to reduce cost

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        if value_embeds is not None:
            normed_value_embeds = self.kv_output_layer(value_embeds)
        if key_embeds is not None:
            normed_key_embeds = self.kv_output_layer(key_embeds)

        return {"key_embeds": key_embeds, "normed_key_embeds": normed_key_embeds,
                "value_embeds": value_embeds, "normed_value_embeds": normed_value_embeds,
                "key_embeds_to_cat": key_embeds_to_cat}

    def kv_output_layer(self, embeds):
        assert len(embeds.shape) == 3
        embeds = self.key_layer_norm(embeds)
        embeds = self.dropout(embeds)
        return embeds

    def _encode_key(self, hidden_states, attention_mask, prefix_embs=None):
        assert hidden_states.ndim == 3 and attention_mask.ndim == 2

        if self.key_encoder_type == "linear":
            if prefix_embs is None:
                prefix_embs = hidden_states[:, :self.prefix_length].view(hidden_states.shape[0], -1)
            # shape: [batch_size, prefix_length * d_model]
            key_embeds = self.key_encoder(prefix_embs)  # shape: [batch_size, d_key]
        elif self.key_encoder_type == "conv":
            if prefix_embs is None:
                key_embeds = self.key_encoder(hidden_states, attention_mask)
            else:
                prefix_mask = torch.ones(prefix_embs.shape[:2].to(prefix_embs.device))
                key_embeds = self.key_encoder(prefix_embs, prefix_mask)
        elif self.key_encoder_type == "prefix":
            prefix_embs = hidden_states[:, :self.prefix_length]  # [batch_size, prefix-len, hidden_size]
            key_embeds = prefix_embs.view(hidden_states.shape[0], -1)  # [batch_size, d_key)
        else:
            raise ValueError(f"Incorrect key_encoder_type: {self.key_encoder_type}")

        return key_embeds

    def forward_with_faiss(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            value_memory=None,
            not_reduced_key_memory=None,
            key_faiss_index=None,
            key_reduce_method=None,
            readout_top_k=1,
            value_fusion_method=None,
    ):
        assert value_memory is not None
        assert key_faiss_index is not None
        assert not_reduced_key_memory is not None
        assert value_fusion_method == self.value_fusion_method

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Sanity check
        assert not use_cache, "This class does not support use_cache because it is encoder only"

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        # Concatenate the prefix embeddings and extend the attention masks
        prefix_embeds = self.prefix_embedding[None, :, :].expand(batch_size, -1, -1).to(inputs_embeds.device)
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        # Extend the attention masks
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length + self.prefix_length).to(inputs_embeds.device)
        else:
            prefix_mask = torch.ones((batch_size, self.prefix_length), dtype=attention_mask.dtype).to(
                inputs_embeds.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length + self.prefix_length), inputs_embeds.device
        )

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        query_embeds = None
        readout_indices = None

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if not use_cache:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            # Query-key matching
            if i == self.key_layer:  # key-layer is Query-Layer
                raw_query_embeds = self._encode_key(hidden_states, attention_mask)  # shape:[batch_size, d_key]
                raw_query_embeds = raw_query_embeds.view(hidden_states.shape[0], -1, hidden_states.shape[-1])

                normed_query_embeds = self.kv_output_layer(raw_query_embeds)  # Query is normed !!!
                query_embeds = reduce_query_or_key_embeds(normed_query_embeds, key_reduce_method)

                # serial mode
                value_embeds, key_embeds_of_value, readout_indices = self.query_memory(
                    value_memory, not_reduced_key_memory, key_faiss_index, query_embeds, readout_top_k
                )
                value_embeds = value_embeds.to(query_embeds.device)
                key_embeds_of_value = key_embeds_of_value.to(query_embeds.device)
                value_embeds = value_embeds.to(query_embeds.dtype)
                key_embeds_of_value = key_embeds_of_value.to(query_embeds.dtype)

                if value_fusion_method == "cat_k_delay+v":
                    # Serial mode, emat key directly.
                    batch_size, num_values, key_nums, hidden_size = key_embeds_of_value.shape
                    if key_nums != self.prefix_length:
                        assert key_nums == 1
                        key_embeds_of_value = key_embeds_of_value.repeat(1, 1, self.prefix_length, 1)
                    hidden_states = torch.cat(
                        [key_embeds_of_value.view(batch_size, num_values * self.prefix_length, hidden_size),
                         hidden_states], dim=1
                    )
                    extend_length = num_values * self.prefix_length
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule

            if self.cat_layer is not None and i == self.cat_layer:
                if value_fusion_method == "async_cat_k_delay+v":
                    # Async mode, emat key in cat_layer. the implementation of delay + v is same to serial mode.
                    batch_size, num_values, key_nums, hidden_size = key_embeds_of_value.shape
                    hidden_states = torch.cat(
                        [key_embeds_of_value.view(batch_size, num_values * self.prefix_length, hidden_size),
                         hidden_states], dim=1
                    )
                    extend_length = num_values * self.prefix_length
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule

            if i == self.value_layer:
                batch_size, num_values, _, hidden_size = key_embeds_of_value.shape

                # assert query_embeds is not None, "Use query_embeds to read memory before assignment."
                if "delay" in value_fusion_method:
                    updated_key = hidden_states[:, :num_values * self.prefix_length]
                    updated_key = updated_key.view(batch_size, num_values, self.prefix_length, hidden_size)
                else:
                    updated_key = None
                integrated_value = self.get_integrated_values(value_embeds, key_embeds_of_value, value_fusion_method,
                                                              query_embeds=query_embeds, updated_key=updated_key,
                                                              key_reduce_method=key_reduce_method)
                if value_fusion_method == "infill":
                    assert self.num_values == 1
                    hidden_states = torch.cat([integrated_value, hidden_states[:, self.prefix_length:]], dim=1)
                elif "cat" in value_fusion_method and "delay" in value_fusion_method:
                    hidden_states[:, :num_values * self.prefix_length] = integrated_value
                    hidden_states = hidden_states.contiguous()
                elif "cat" in value_fusion_method and "delay" not in value_fusion_method:
                    hidden_states = torch.cat([integrated_value, hidden_states], dim=1)
                    extend_length = integrated_value.shape[1]
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule
                else:
                    raise NotImplementedError(f"{value_fusion_method} is not defined.")

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    query_embeds,
                    readout_indices
                ]
                if v is not None
            )
        return CATEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            query_embeds=query_embeds,
            readout_indices=readout_indices,
            updated_attention_mask=None,
        )

    async def forward_with_async_faiss(
            self,
            input_ids,
            attention_mask,
            return_dict,
            readout_top_k,
            key_reduce_method,
            value_fusion_method,
            key_faiss_index,
            value_memory,
            not_reduced_key_memory,
            encoder_hidden_states=None,
            inputs_embeds=None,
            head_mask=None,
            cross_attn_head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ):
        assert value_memory is not None
        assert key_faiss_index is not None
        assert not_reduced_key_memory is not None

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Sanity check
        assert not use_cache, "This class does not support use_cache because it is encoder only"

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        # Concatenate the prefix embeddings and extend the attention masks
        prefix_embeds = self.prefix_embedding[None, :, :].expand(batch_size, -1, -1).to(inputs_embeds.device)
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        # Extend the attention masks
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length + self.prefix_length).to(inputs_embeds.device)
        else:
            prefix_mask = torch.ones((batch_size, self.prefix_length), dtype=attention_mask.dtype).to(
                inputs_embeds.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, (batch_size, seq_length + self.prefix_length), inputs_embeds.device
        )

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        query_embeds = None
        readout_indices = None

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warn(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if not use_cache:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            # Query-key matching
            if i == self.key_layer:  # key-layer is Query-Layer
                raw_query_embeds = self._encode_key(hidden_states, attention_mask)  # shape:[batch_size, d_key]
                raw_query_embeds = raw_query_embeds.view(hidden_states.shape[0], -1, hidden_states.shape[-1])

                normed_query_embeds = self.kv_output_layer(raw_query_embeds)  # Query is normed !!!
                query_embeds = reduce_query_or_key_embeds(normed_query_embeds, key_reduce_method)

                # async mode

                # loop = asyncio.get_event_loop()
                # executor = futures.ThreadPoolExecutor()  # futures.ProcessPoolExecutor()

                async_query_future = asyncio.get_event_loop().run_in_executor(
                    futures.ThreadPoolExecutor(), self.query_memory,
                    value_memory, not_reduced_key_memory, key_faiss_index, query_embeds, readout_top_k
                )

            if i == self.cat_layer:
                value_embeds, key_embeds_of_value, readout_indices = await async_query_future
                value_embeds = value_embeds.to(query_embeds.device)
                key_embeds_of_value = key_embeds_of_value.to(query_embeds.device)
                value_embeds = value_embeds.to(query_embeds.dtype)
                key_embeds_of_value = key_embeds_of_value.to(query_embeds.dtype)

                if value_fusion_method == "async_cat_k_delay+v":
                    # Async mode, emat key in cat_layer. the implementation of delay + v is same to serial mode.
                    batch_size, num_values, key_nums, hidden_size = key_embeds_of_value.shape
                    hidden_states = torch.cat(
                        [key_embeds_of_value.view(batch_size, num_values * self.prefix_length, hidden_size),
                         hidden_states], dim=1
                    )
                    extend_length = num_values * self.prefix_length
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule

            if i == self.value_layer:
                batch_size, num_values, _, hidden_size = key_embeds_of_value.shape

                # assert query_embeds is not None, "Use query_embeds to read memory before assignment."
                if "delay" in value_fusion_method:
                    updated_key = hidden_states[:, :num_values * self.prefix_length]
                    updated_key = updated_key.view(batch_size, num_values, self.prefix_length, hidden_size)
                else:
                    updated_key = None
                integrated_value = self.get_integrated_values(value_embeds, key_embeds_of_value, value_fusion_method,
                                                              query_embeds=query_embeds, updated_key=updated_key,
                                                              key_reduce_method=key_reduce_method)
                if value_fusion_method == "infill":
                    assert self.num_values == 1
                    hidden_states = torch.cat([integrated_value, hidden_states[:, self.prefix_length:]], dim=1)
                elif "cat" in value_fusion_method and "delay" in value_fusion_method:
                    hidden_states[:, :num_values * self.prefix_length] = integrated_value
                    hidden_states = hidden_states.contiguous()
                elif "cat" in value_fusion_method and "delay" not in value_fusion_method:
                    hidden_states = torch.cat([integrated_value, hidden_states], dim=1)
                    extend_length = integrated_value.shape[1]
                    extend_mask = torch.ones((batch_size, extend_length), dtype=attention_mask.dtype)
                    attention_mask = torch.cat([extend_mask.to(inputs_embeds.device), attention_mask], dim=1)
                    extended_attention_mask = self.get_extended_attention_mask(
                        attention_mask, attention_mask.shape[:2], inputs_embeds.device
                    )
                    position_bias = None  # clean the position_bias, compute in T5-SelfAttentionModule
                else:
                    raise NotImplementedError(f"{value_fusion_method} is not defined.")

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    query_embeds,
                    readout_indices
                ]
                if v is not None
            )
        return CATEncoderOutput(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            query_embeds=query_embeds,
            readout_indices=readout_indices,
            updated_attention_mask=None,
        )

    def query_memory(self, value_memory, not_reduced_key_memory, key_faiss_index,
                     query_embeds, readout_top_k):
        assert value_memory.shape[1] == self.prefix_length
        assert value_memory.shape[2] == self.model_dim
        if type(query_embeds) == torch.tensor:
            query_embeds = query_embeds.n
        top_indices, _ = mips(key_faiss_index, query_embeds.cpu(), readout_top_k, n_queries_to_parallelize=20480)
        memory_size, hidden_num, hidden_size = value_memory.shape
        bs = query_embeds.shape[0]
        top_indices = torch.tensor(top_indices)
        readout_value = torch.index_select(value_memory, 0, top_indices.view(-1))
        readout_value = readout_value.view(bs, readout_top_k, hidden_num, hidden_size)
        readout_key_embeds_of_value = torch.index_select(not_reduced_key_memory, 0, top_indices.view(-1))
        readout_key_embeds_of_value = readout_key_embeds_of_value.view(bs, readout_top_k, hidden_num, hidden_size)
        return readout_value, readout_key_embeds_of_value, top_indices


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5WithKeyValueMemory(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]

    # _keys_to_ignore_on_load_unexpected = [
    #     r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    # ]

    def __init__(self, config):
        super(T5ForConditionalGeneration, self).__init__(config)
        self.model_dim = config.d_model
        # self.generate
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackWithKeyValueMemory(encoder_config, self.shared)
        if config.not_share_encoder:
            self.kv_encoder = T5StackWithKeyValueMemory(copy.deepcopy(encoder_config), self.shared)
        else:
            self.kv_encoder = None
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.adapter = getattr(config, "adapter", None)
        if self.adapter is not None:
            self.adapter = RetAdapter(config.d_model, config.adapter_out_dim, adapter_type=self.adapter)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.config = config

    def get_tunable_key_value_parameters(self):
        add_pos_layer = min(self.config.key_layer, self.config.value_layer)
        tunable_parameters_list = list(self.encoder.block[add_pos_layer].layer[0].
                                       SelfAttention.relative_attention_bias.parameters())
        if not self.config.not_share_encoder:
            if self.encoder.key_encoder is not None:
                tunable_parameters_list += list(self.encoder.key_encoder.parameters())
            tunable_parameters_list += [self.encoder.prefix_embedding]
            tunable_parameters_list += list(self.encoder.key_layer_norm.parameters())
        elif self.config.not_share_encoder:
            tunable_parameters_list += list(self.kv_encoder.parameters())
        return tunable_parameters_list

    def freeze_t5_params(self):
        tunable_key_value_parameters = self.get_tunable_key_value_parameters()
        requires_grad_nums = 0
        for param in self.parameters():
            if any(param is tp for tp in tunable_key_value_parameters):
                param.requires_grad = True
                requires_grad_nums += 1
            else:
                param.requires_grad = False
        assert requires_grad_nums == len(tunable_key_value_parameters)
        logger.info(f"tunable params num: {len(tunable_key_value_parameters)}")

    def freeze_kv_encoder_params(self):
        assert self.encoder.key_encoder is not None
        kv_encoder_params = list(self.encoder.key_encoder.parameters())
        for param in self.parameters():
            if any(param is tp for tp in kv_encoder_params):
                param.requires_grad = False

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CATSeq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            key_value_input_ids=None,
            key_value_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            decoder_only_attend_on_prefix=False,
            encoder_outputs_are_key_or_value=False,
            key_embeds=None,
            value_embeds=None,
            key_memory=None,
            value_memory=None,
            key_faiss_index=None,
            key_reduce_method=None,
            value_fusion_method=None,
            key_embeds_of_value=None,
            use_ae_lm_head=False,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small')

            >>> # training
            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> # inference
            >>> input_ids = tokenizer("summarize: studies have shown that owning a dog is good for you", return_tensors="pt").input_ids  # Batch size 1
            >>> outputs = model.generate(input_ids)
            >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
            >>> # studies have shown that owning a dog is good for you.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                key_embeds=key_embeds,
                value_embeds=value_embeds,
                key_memory=key_memory,
                value_memory=value_memory,
                key_faiss_index=key_faiss_index,
                key_reduce_method=key_reduce_method,
                value_fusion_method=value_fusion_method,
                key_embeds_of_value=key_embeds_of_value
            )
        elif return_dict:
            assert isinstance(encoder_outputs, CATEncoderOutput)

        hidden_states = encoder_outputs.last_hidden_state

        # Extend attention_mask on the left for attention for the prefix
        batch_size, seq_length = hidden_states.shape[:2]
        # seq_length: prefix + original hidden length
        # attn_length: original hidden length
        if encoder_outputs_are_key_or_value:
            encoder_attention_mask = None
        else:
            attn_length = attention_mask.shape[1]
            assert seq_length > attn_length, f"{seq_length} is not larger than {attn_length}"
            if decoder_only_attend_on_prefix:
                hidden_states = hidden_states[:, :seq_length - attn_length]
                encoder_attention_mask = torch.ones(hidden_states.shape[:2]).to(attention_mask.device)
            else:
                prefix_mask = torch.ones(attention_mask.shape[0], seq_length - attn_length).to(attention_mask.device)
                encoder_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        if use_ae_lm_head:
            lm_logits = self.ae_lm_head(sequence_output)
        else:
            lm_logits = self.lm_head(sequence_output)

        # loss_dict = {}
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        assert isinstance(encoder_outputs, CATEncoderOutput)
        return CATSeq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
            cat_encoder_outputs=encoder_outputs
        )

    def _prepare_encoder_decoder_kwargs_for_generation(
            self, input_ids: torch.LongTensor, model_kwargs
    ) -> Dict[str, Any]:
        if "encoder_outputs" not in model_kwargs:
            # retrieve encoder hidden states
            encoder = self.get_encoder()
            encoder_kwargs = {
                argument: value
                for argument, value in model_kwargs.items()
                if not (argument.startswith("decoder_") or argument.startswith("cross_attn"))
            }

            if model_kwargs.get("key_value_input_ids", None) is not None:
                key_embeds, value_embeds = encoder.embed_kv(
                    input_ids=model_kwargs["key_value_input_ids"],
                    attention_mask=model_kwargs.get("key_value_attention_mask", None),
                    head_mask=model_kwargs.get("head_mask", None),
                )
                encoder_kwargs["key_embeds"] = key_embeds
                encoder_kwargs["value_embeds"] = value_embeds
                encoder_kwargs.pop("key_value_input_ids")
                if "key_value_attention_mask" in encoder_kwargs:
                    encoder_kwargs.pop("key_value_attention_mask")

            model_kwargs["encoder_outputs"]: ModelOutput = encoder(input_ids, return_dict=True, **encoder_kwargs)
        return model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            encoder_outputs_are_key_or_value=False,
            decoder_only_attend_on_prefix=False,
            value_fusion_method=None,
            # use_ae_decoder=False,
            **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "encoder_outputs_are_key_or_value": encoder_outputs_are_key_or_value,
            "attention_mask": attention_mask,
            "decoder_only_attend_on_prefix": decoder_only_attend_on_prefix,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "value_fusion_method": value_fusion_method,
            # "use_ae_decoder": use_ae_decoder
        }

    def CAT_embed_kv(self, *args, **kwargs) -> Dict:
        if self.kv_encoder is None:
            # share kv-encoder with query-encoder
            return self.encoder.embed_kv(*args, **kwargs)
        else:
            # Do not share encoder
            return self.kv_encoder.embed_kv(*args, **kwargs)

    def CAT_embed_q(self, *args, **kwargs) -> Dict:
        return self.encoder.embed_kv(*args, **kwargs)  # self.encoder is always query-encoder

    def compute_key_value_ae_loss(
            self,
            separate_task=True,
            key_value_input_ids=None,
            key_value_attention_mask=None,
            key_input_ids=None,
            key_attention_mask=None,
            value_input_ids=None,
            value_attention_mask=None,
            key_labels_input_ids=None,
            value_labels_input_ids=None,
            train_key=False,
            train_value=False,
            use_ae_lm_head=False,
            **kwargs
    ):
        loss_dict = dict()
        embed_dict = self.wrapped_embed_kv(
            separate_task=separate_task,
            key_value_input_ids=key_value_input_ids,
            key_value_attention_mask=key_value_attention_mask,
            key_input_ids=key_input_ids,
            key_attention_mask=key_attention_mask,
            value_input_ids=value_input_ids,
            value_attention_mask=value_attention_mask,
            compute_key=train_key,
            compute_value=train_value,
            embed_for_ae_task=True
        )
        key_embeds = embed_dict["normed_key_embeds"] if train_key else None
        if "async_cat_k+v" == self.config.value_fusion_method:
            value_embeds = self.encoder.kv_output_layer(embed_dict["value_embeds"] + embed_dict["key_embeds"]) \
                if train_value else None  # normed value for generation
        else:
            value_embeds = embed_dict["normed_value_embeds"] if train_value else None  # normed value for generation
        # key_embeds = key_embeds.view(key_embeds.shape[0], -1, self.model_dim) if key_embeds is not None else None
        # value_embeds = value_embeds.view(value_embeds.shape[0], -1, self.model_dim)
        # key_embeds/value_embeds [batch_size, prefix_length, model_dim]
        # the length of key_embeds/value_embeds is prefix_length, do not need attention_mask
        if train_key:
            key_ae_outputs = self.forward(
                # batch, num, hidden_size; 1024 -> 2, 512
                encoder_outputs=CATEncoderOutput(last_hidden_state=key_embeds, hidden_states=None, attentions=None),
                encoder_outputs_are_key_or_value=True,
                labels=key_labels_input_ids,
                use_ae_lm_head=use_ae_lm_head,
            )
            loss_dict["key_ae_loss"] = key_ae_outputs["loss"]
        if train_value:
            value_ae_outputs = self.forward(
                encoder_outputs=CATEncoderOutput(last_hidden_state=value_embeds, hidden_states=None, attentions=None),
                encoder_outputs_are_key_or_value=True,
                labels=value_labels_input_ids,
                use_ae_lm_head=use_ae_lm_head,
            )
            loss_dict["value_ae_loss"] = value_ae_outputs["loss"]
        return loss_dict

    def compute_text_pair_key_value_ae_loss(
            self,
            key_input_ids=None,
            key_attention_mask=None,
            value_input_ids=None,
            value_attention_mask=None,
            key_labels_input_ids=None,
            value_labels_input_ids=None,
            separate_decode=False,
            hypothesis_decoder_input_ids=None,
            hypothesis_decoder_labels=None,
            premise_decoder_input_ids=None,
            premise_decoder_labels=None,
            train_key=False,
            train_value=False,
            **kwargs
    ):
        # Auto-encoding pretraining for text-pair task (e.g. NLI)
        # separate_decode argument choices whether the model generates text-pair respectively.
        loss_dict = dict()
        embed_dict = self.wrapped_embed_kv(
            separate_task=True,
            key_value_input_ids=None,
            key_value_attention_mask=None,
            key_input_ids=key_input_ids,
            key_attention_mask=key_attention_mask,
            value_input_ids=value_input_ids,
            value_attention_mask=value_attention_mask,
            compute_key=train_key,
            compute_value=train_value,
            embed_for_ae_task=True
        )
        key_embeds = embed_dict["normed_key_embeds"] if train_key else None
        value_embeds = embed_dict["normed_value_embeds"] if train_value else None
        if train_key:
            if separate_decode:
                # hypothesis auto-encoding
                hypothesis_ae_outputs = self.forward(
                    encoder_outputs=CATEncoderOutput(last_hidden_state=key_embeds, hidden_states=None, attentions=None),
                    encoder_outputs_are_key_or_value=True,
                    decoder_input_ids=hypothesis_decoder_input_ids,
                    labels=hypothesis_decoder_labels,
                    use_ae_lm_head=False,
                )
                loss_dict["hypothesis_ae_loss"] = hypothesis_ae_outputs["loss"]
                # premise auto-encoding
                premise_ae_outputs = self.forward(
                    encoder_outputs=CATEncoderOutput(last_hidden_state=key_embeds, hidden_states=None, attentions=None),
                    encoder_outputs_are_key_or_value=True,
                    decoder_input_ids=premise_decoder_input_ids,
                    labels=premise_decoder_labels,
                    use_ae_lm_head=False,
                )
                loss_dict["premise_ae_loss"] = premise_ae_outputs["loss"]
            else:
                key_ae_outputs = self.forward(
                    # batch, num, hidden_size; 1024 -> 2, 512
                    encoder_outputs=CATEncoderOutput(last_hidden_state=key_embeds, hidden_states=None, attentions=None),
                    encoder_outputs_are_key_or_value=True,
                    labels=key_labels_input_ids,
                    use_ae_lm_head=False,
                )
                loss_dict["key_ae_loss"] = key_ae_outputs["loss"]
        if train_value:
            value_ae_outputs = self.forward(
                encoder_outputs=CATEncoderOutput(last_hidden_state=value_embeds, hidden_states=None, attentions=None),
                encoder_outputs_are_key_or_value=True,
                labels=value_labels_input_ids,
                use_ae_lm_head=False,
            )
            loss_dict["value_ae_loss"] = value_ae_outputs["loss"]
        return loss_dict

    def compute_qa_loss(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            decoder_only_attend_on_prefix=False,
            encoder_outputs_are_key_or_value=False,
            key_memory=None,
            value_memory=None,
            key_faiss_index=None,
            key_reduce_method=None,
            positive_key_embeds=None,
            negative_key_embeds=None,
            value_embeds=None,
            matching_targets=None,
            value_fusion_method=None,
            key_embeds_of_value=None,
            negative_mask=None,
            only_train_adapter=False
    ):
        loss_dict = dict()
        if only_train_adapter:
            embed_dict = self.CAT_embed_q(
                input_ids=input_ids,
                attention_mask=attention_mask,
                compute_key=True, compute_value=False
            )
            query_embeds = embed_dict["normed_key_embeds"]
            query_embeds = reduce_query_or_key_embeds(query_embeds, key_reduce_method)
            gen_loss = torch.tensor(0.0)
        else:
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_only_attend_on_prefix=decoder_only_attend_on_prefix,
                encoder_outputs_are_key_or_value=encoder_outputs_are_key_or_value,
                key_embeds=None,
                value_embeds=value_embeds,
                key_memory=key_memory,
                value_memory=value_memory,
                key_faiss_index=key_faiss_index,
                key_reduce_method=key_reduce_method,
                value_fusion_method=value_fusion_method,
                key_embeds_of_value=key_embeds_of_value
            )
            query_embeds = outputs.cat_encoder_outputs.query_embeds
            gen_loss = outputs.loss
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss_dict["gen_loss"] = gen_loss
        if positive_key_embeds is not None:
            if self.adapter is not None:
                query_embeds = self.adapter(query_embeds)
                positive_key_embeds = self.adapter(positive_key_embeds)
                negative_key_embeds = self.adapter(negative_key_embeds)
            scores1 = torch.mm(query_embeds, positive_key_embeds.t())
            scores2 = torch.mm(query_embeds, negative_key_embeds.t())
            if negative_mask is not None:
                negative_mask = ~negative_mask.bool().to(negative_key_embeds.device)
                scores2 = scores2.masked_fill(negative_mask, float('-inf'))
            scores = torch.cat((scores1, scores2), dim=1)
            match_loss = loss_fct(scores, matching_targets)
            loss_dict["match_loss"] = match_loss

        return loss_dict

    def compute_gen_and_match_loss(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            decoder_only_attend_on_prefix=False,
            encoder_outputs_are_key_or_value=False,
            key_reduce_method=None,
            value_embeds=None,
            value_fusion_method=None,
            key_embeds_of_value=None,
            positive_and_negative_embeds=None,
            matching_mask=None,
            matching_targets=None,
            use_triple_loss=None,
    ):
        """
        value_embeds: retrieved key's value embeds, shape: [batch_size, num_values, prefix_length, hidden_size]
        key_embeds_of_value: retrieved key embeds, shape: [batch_size, num_values, key_dim // model_dim, hidden_size]
        """
        loss_dict = dict()
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_only_attend_on_prefix=decoder_only_attend_on_prefix,
            encoder_outputs_are_key_or_value=encoder_outputs_are_key_or_value,
            key_embeds=None,
            value_embeds=value_embeds,
            key_memory=None,
            value_memory=None,
            key_faiss_index=None,
            key_reduce_method=key_reduce_method,
            value_fusion_method=value_fusion_method,
            key_embeds_of_value=key_embeds_of_value
        )

        gen_loss = outputs.loss
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss_dict["gen_loss"] = gen_loss
        if not use_triple_loss and positive_and_negative_embeds is not None:
            query_embeds = outputs.cat_encoder_outputs.query_embeds
            # query_embeds: batch_size, hidden
            # positive_and_negative_embeds:  N, hidden
            scores = torch.mm(query_embeds, positive_and_negative_embeds.transpose(1, 0))
            matching_mask = ~matching_mask.bool().to(positive_and_negative_embeds.device)
            scores = scores.masked_fill(matching_mask, float('-inf'))
            match_loss = loss_fct(scores, matching_targets)
            loss_dict["match_loss"] = match_loss
        # elif use_triple_loss and positive_key_embeds is not None:
        #     triple_loss_fct = nn.TripletMarginLoss(margin=0.5, p=2)
        #     query_embeds = outputs.cat_encoder_outputs.query_embeds
        #     batch_size, hidden_size = query_embeds.shape
        #     # negative_nums = negative_key_embeds.shape[0] // batch_size
        #     negative_key_embeds = negative_key_embeds.view(batch_size, -1, hidden_size)
        #     group_negative_key_embeds = negative_key_embeds.transpose(0, 1)
        #     triple_losses = []
        #     for cur_negative_key_embeds in group_negative_key_embeds:
        #         triple_losses.append(
        #             triple_loss_fct(query_embeds, positive_key_embeds, cur_negative_key_embeds)
        #         )
        #     triple_loss = sum(triple_losses) / len(triple_losses)
        #     loss_dict["triple_loss"] = torch.nan_to_num(triple_loss)

        return loss_dict

    def wrapped_embed_kv(
            self,
            separate_task=False,
            key_value_input_ids=None,
            key_value_attention_mask=None,
            key_input_ids=None,
            key_attention_mask=None,
            value_input_ids=None,
            value_attention_mask=None,
            compute_key=False,
            compute_value=False,
            embed_for_ae_task=False,
    ) -> Dict:
        device = self.device
        # key_embeds, value_embeds = None, None
        if separate_task:
            res = dict()
            if compute_key:
                key_res = self.CAT_embed_kv(
                    input_ids=key_input_ids.to(device), attention_mask=key_attention_mask.to(device),
                    compute_key=compute_key, compute_value=False, embed_for_ae_task=embed_for_ae_task
                )
                res.update({k: v for k, v in key_res.items() if "key" in k})
            if compute_value:
                value_res = self.CAT_embed_kv(
                    input_ids=value_input_ids.to(device), attention_mask=value_attention_mask.to(device),
                    compute_key=False, compute_value=compute_value, embed_for_ae_task=embed_for_ae_task,

                )
                res.update({k: v for k, v in value_res.items() if "value" in k})
        else:
            res = self.CAT_embed_kv(
                input_ids=key_value_input_ids.to(device),
                attention_mask=key_value_attention_mask.to(device),
                compute_key=compute_key, compute_value=compute_value,
                embed_for_ae_task=embed_for_ae_task
            )
        return res
