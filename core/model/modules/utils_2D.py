"""
This is the adaptation of T5+2D implementation from DUE benchmark paper

Reference: 
    paper: 
        + T5+2D: "Going Full-TILT Boogie on Document Understanding with Text-Image-Layout Transformer"
        + DUE: "DUE: End-to-End Document Understanding Benchmark"
    github: https://github.com/due-benchmark
"""

import torch
from torch import nn
import random
from abc import ABC, abstractmethod
from torch import Optional, Tensor
from typing import Any, Dict, Optional, Sequence, Union, Tuple
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Block, T5Stack, T5ForConditionalGeneration, T5Attention, T5PreTrainedModel, T5LayerNorm
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


get_relative_position_bucket = T5Attention._relative_position_bucket
AUGMENTATION_RANGE = (0.80, 1.25)

class RelativePositionBiasBase(nn.Module, ABC):
    """
    Base class of relative biases
    :param num_heads: number of heads in lm model, it will create embeddings of size `num_heads`,
        which will be added to scores per each token pair
    :param relative_attention_num_buckets: pair token metric
        (distance in the sequence, distance in pixels etc.) will be bucketed,
        parameter is defining number of such buckets
    :param bidirectional: defining if for pair of tokens distance should be bidirecional,
        if bidirectional=False, then distance(tok1, tok2) == distance(tok2, tok1)
    :param scaling_factor: defining factor which will be used to scale relative distance
    :param max_distance: all distances above this value will end up in the one/same bucket
    :param augmentation: whether to multiple relative distances by random scalar
    :param expand: used for re-using pretrained model with subsequent addition of prefix_bucket
    """

    def __init__(self, num_heads=None, relative_attention_num_buckets=32,
                 bidirectional=True, scaling_factor=1, max_distance=128,
                 augmentation=False, prefix_bucket=False, expand=False, device="cuda"):

        super(RelativePositionBiasBase, self).__init__()
        self.device = device
        self.prefix_bucket = prefix_bucket
        self.augmentation = augmentation
        self.max_distance = max_distance
        self.scaling_factor = scaling_factor
        self.bidirectional = bidirectional
        self.num_heads = num_heads
        self.expand = expand
        self.relative_attention_num_buckets = relative_attention_num_buckets
        extra_head = 2 if prefix_bucket and not self.expand else 0
        self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets + extra_head, self.num_heads).to(self.device)


    @abstractmethod
    def prepare_input(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    def get_bucket(self, input_ids: Optional[Tensor] = None,  # type: ignore
                   attention_mask: Optional[Tensor] = None,
                   coordinates: Optional[Tensor] = None) -> Tensor:
        relative_position = self.prepare_input(input_ids, attention_mask, coordinates)
        relative_position = relative_position.to(device=next(self.parameters()).device)
        rp_bucket: Tensor = get_relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.max_distance,
        )
        return rp_bucket

    def get_relative_position(self, positions):
        context_position = positions[:, :, None]
        memory_position = positions[:, None, :]
        relative_position = memory_position - context_position
        if self.augmentation and self.training:
            relative_position *= random.uniform(*AUGMENTATION_RANGE)
        relative_position *= self.scaling_factor

        return relative_position.to(torch.long)

    def forward(self, input_ids: Optional[Tensor] = None,  # type: ignore
                attention_mask: Optional[Tensor] = None,
                coordinates: Optional[Tensor] = None) -> Tensor:

        # re-using pretrained model with subsequent addition of prefix_bucket
        if self.expand and self.prefix_bucket:
            new_bias = nn.Embedding(self.relative_attention_num_buckets + 2, self.num_heads).to(self.device)
            new_bias.weight.data[:self.relative_attention_num_buckets] = self.relative_attention_bias.weight.data
            new_bias.weight.data[self.relative_attention_num_buckets:] = 0.1
            new_bias = new_bias.to(self.relative_attention_bias.weight.device)
            self.relative_attention_bias = new_bias
            self.expand = False

        rp_bucket = self.get_bucket(input_ids, attention_mask, coordinates)

        if self.prefix_bucket:
            if rp_bucket.size(0) == 1 and input_ids.size(0) > 1:
                rp_bucket = rp_bucket.repeat(input_ids.size(0), 1, 1)
            # based on assumption that prefix bboxes are negative
            is_prefix = coordinates[:, :, 1] < 0
            num_prefix = is_prefix.sum(-1)
            for idx, num_prefix_row in enumerate(num_prefix.cpu().numpy()):
                rp_bucket[idx, :num_prefix_row, num_prefix_row:] = self.relative_attention_num_buckets
                rp_bucket[idx, num_prefix_row:, :num_prefix_row] = self.relative_attention_num_buckets + 1

        values = self.relative_attention_bias(rp_bucket)
        assert values.dim() == 4, "Wrong dimension of values tensor"
        values = values.permute([0, 3, 1, 2])

        return values


class RelativePositionBias1D(RelativePositionBiasBase):
    def __init__(self, scaling_factor=1, max_distance=128, **kwargs):
        """
        Reimplementation of T5 relative position bias. Distance between given tokens is
        their distance in the sequence. Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, input_ids: Optional[Tensor] = None,
                      attention_mask: Optional[Tensor] = None,
                      coordinates: Optional[Tensor] = None) -> Tensor:
        assert self.scaling_factor == 1, "No need to scale 1d features"
        assert input_ids is not None
        relative_position = self.get_relative_position(torch.arange(input_ids.size(1), dtype=torch.long, device=self.device)[None, :])

        return relative_position


class RelativePositionBiasHorizontal(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings horizontal distance between two tokens.
        Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, input_ids: Optional[Tensor] = None,
                      attention_mask: Optional[Tensor] = None,
                      coordinates: Optional[Tensor] = None) -> Tensor:
        assert self.scaling_factor > 1.0, \
            "Need to scale the values of bboxes, as there are in small (0,1) range"
        # get x positions of left point of bbox
        assert coordinates is not None
        horizontal_position = coordinates[:, :, [0, 2]].mean(dim=-1)


        return self.get_relative_position(horizontal_position)


class RelativePositionBiasVertical(RelativePositionBiasBase):
    def __init__(self, scaling_factor=100, max_distance=100, **kwargs):
        """
        Represents in the bucket embeddings vertical distance between two tokens.
        Parameters are the same as in base class
        """
        super().__init__(scaling_factor=scaling_factor, max_distance=max_distance, **kwargs)

    def prepare_input(self, input_ids: Optional[Tensor] = None,
                      attention_mask: Optional[Tensor] = None,
                      coordinates: Optional[Tensor] = None) -> Tensor:
        assert self.scaling_factor > 1.0, \
            "Need to scale the values of bboxes, as there are in small (0,1) range"
        # get y positions of middle of bbox
        assert coordinates is not None
        vertical_position = coordinates[:, :, [1, 3]].mean(dim=-1)


        return self.get_relative_position(vertical_position)


class RelativePositionBiasAggregated(nn.Module):
    def __init__(self, modules: Sequence[RelativePositionBiasBase]):
        """
        Class will sums up computed biases
        :param modules: list of relative bias modules
        """
        super().__init__()
        self.biases = nn.ModuleList(modules)

    def forward(
        self,
        input_ids: Optional[Tensor] = None,  # type: ignore
        attention_mask: Optional[Tensor] = None,
        coordinates: Optional[Tensor] = None,
    ) -> Union[float, Tensor]:
        x = 0.0
        for bias in self.biases:  # type: ignore
            x = bias(input_ids, attention_mask, coordinates) + x

        return x


class T52DStack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False


    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)


    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

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
        position_bias = None
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            if self.embed_tokens is None:
                raise ValueError("You have to initialize the model with valid token embeddings")
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        #position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

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
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
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
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

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
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class T52dForConditionalGeneration(T5ForConditionalGeneration):
    """
    Copied from original T5ForConditionalGeneration class with signature extended with 2D data.
    :param config: a `T5Config` instance
    """

    def __init__(self, config):
        super(T52dForConditionalGeneration, self).__init__(config)

        self.encoder = T52DStack(self.encoder.config, self.shared)
        self.decoder = T5Stack(self.decoder.config, self.shared)

        # get max length of decoder part, for T5 decoder lenght depends
        # on the task and it can be modified by passing `_max_decoder_length` to the model/config
        self._max_decoder_length = config.max_decoder_length if hasattr(config, "max_decoder_length") else 200

        self.config.decoder_start_token_id = self.config.pad_token_id


        self.init_weights()

    def _init_weights(self, module):
        """Initialize the weights"""
        super()._init_weights(module)
        if isinstance(module, RelativePositionBiasBase):
            factor = self.config.initializer_factor
            d_model = self.config.d_model
            module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def forward(
        self,
        input_ids: Tensor = None,
        attention_mask: Tensor = None,
        decoder_input_ids: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        encoder_outputs: Optional[Tensor] = None,
        past_key_values: Optional[Tensor] = None,
        position_bias = None,
        class_labels: Optional[Tensor] = None,
        masked_lm_labels: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
        decoder_inputs_embeds: Optional[Tensor] = None,
        use_cache=True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[Tensor, ...]:

        # Compute encoder output and pass modified bias
        if encoder_outputs is None:
            # compute positional bias (can be aggregation of 1D and 2D biases)

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                position_bias=position_bias,

            )

        if encoder_outputs is None:
            return None

        # ugly hack for model to work as an encoder
        if decoder_input_ids is None and masked_lm_labels is None:
            return encoder_outputs
        #import pdb; pdb.set_trace()

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=masked_lm_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        return outputs  # type: ignore

    def get_encoder(self):
        return self