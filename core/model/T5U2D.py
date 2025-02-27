from torch import nn
from .modules import UNet, RoIPool

from .modules import (
    T52dForConditionalGeneration,
    RelativePositionBiasAggregated,
    RelativePositionBias1D,
    RelativePositionBiasHorizontal,
    RelativePositionBiasVertical
    )

class SpatialModule(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        
        self.unet = UNet()
        self.roi_pool = RoIPool((3,3))
        self.proj = nn.Linear(128*3*3, d_model)
        nn.init.xavier_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pixel_values, coordinates):
        out = self.roi_pool(self.unet(pixel_values), 
                            coordinates)
        
        batch_size, seq_length = out.size(0), out.size(1)
        
        return self.proj(out.reshape(batch_size, seq_length, -1)) 


class T5U2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.spatial_module = SpatialModule(self.config.d_model)
        
        self.backbone = T52dForConditionalGeneration.from_pretrained(self.config._name_or_path)

        self.rel2Dbias = RelativePositionBiasAggregated([RelativePositionBias1D(num_heads = self.backbone.config.num_heads),
                                                        RelativePositionBiasHorizontal(num_heads = self.backbone.config.num_heads),
                                                        RelativePositionBiasVertical(num_heads = self.backbone.config.num_heads)])

    def forward(self,
                pixel_values,
                coordinates,
                input_ids,
                label_ids,
                src_attention_mask,
                label_attention_mask,
                ):

        position_bias = self.rel2Dbias(input_ids, src_attention_mask, coordinates)

        inputs_embeds = self.calculate_embedding(input_ids, pixel_values, coordinates)
        
        encoder_outputs = self.backbone.encoder(
                attention_mask=src_attention_mask,
                inputs_embeds=inputs_embeds,
                position_bias = position_bias
            ).last_hidden_state

        decoder_outputs = self.backbone.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.backbone.shared(label_ids),
            attention_mask = label_attention_mask
        ).last_hidden_state


        return self.backbone.lm_head(decoder_outputs)
    
    def calculate_embedding(self, input_ids, pixel_values, coordinates):
        visual_embedding = self.spatial_module(pixel_values = pixel_values, coordinates = coordinates)
        semantic_embedding = self.backbone.shared(input_ids)

        return semantic_embedding + visual_embedding


    def generate(self,
                pixel_values,
                coordinates,
                input_ids,
                src_attention_mask,
                max_length,
                ):

        inputs_embeds = self.calculate_embedding(input_ids, pixel_values, coordinates)

        position_bias = self.rel2Dbias(input_ids, src_attention_mask, coordinates)

        return self.backbone.generate(  inputs_embeds = inputs_embeds, 
                                        position_bias = position_bias,
                                        max_length = max_length)