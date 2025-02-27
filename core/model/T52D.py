from torch import nn
from .modules import (
    T52dForConditionalGeneration,
    RelativePositionBiasAggregated,
    RelativePositionBias1D,
    RelativePositionBiasHorizontal,
    RelativePositionBiasVertical
    )

class T52D(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.backbone = T52dForConditionalGeneration.from_pretrained(self.config._name_or_path)
        self.rel2Dbias = RelativePositionBiasAggregated([RelativePositionBias1D(num_heads = self.backbone.config.num_heads),
                                                        RelativePositionBiasHorizontal(num_heads = self.backbone.config.num_heads),
                                                        RelativePositionBiasVertical(num_heads = self.backbone.config.num_heads)])

    def forward(self,
                input_ids,
                label_ids,
                src_attention_mask,
                label_attention_mask,
                coordinates):

        position_bias = self.rel2Dbias(input_ids, src_attention_mask, coordinates)

        encoder_outputs = self.backbone.encoder(
                attention_mask=src_attention_mask,
                inputs_embeds=self.backbone.shared(input_ids),
                position_bias = position_bias
            ).last_hidden_state

        decoder_outputs = self.backbone.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.backbone.shared(label_ids),
            attention_mask = label_attention_mask
        ).last_hidden_state


        return self.backbone.lm_head(decoder_outputs)

    def generate(self,
                input_ids,
                max_length,
                src_attention_mask,
                coordinates):

        position_bias = self.rel2Dbias(input_ids, src_attention_mask, coordinates)

        return self.backbone.generate(input_ids = input_ids,
                                      position_bias = position_bias,
                                      max_length = max_length)