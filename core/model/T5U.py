from torch import nn
from transformers import T5ForConditionalGeneration
from .modules import UNet, RoIPool

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


class T5U(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.spatial_module = SpatialModule(self.config.d_model)
        
        self.backbone = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)
        

    def forward(self,
                pixel_values,
                coordinates,
                input_ids,
                label_ids,
                src_attention_mask,
                label_attention_mask,
                ):


        inputs_embeds = self.calculate_embedding(input_ids, pixel_values, coordinates)
        
        encoder_outputs = self.backbone.encoder(
                attention_mask=src_attention_mask,
                inputs_embeds=inputs_embeds,
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
                max_length,
                ):

        inputs_embeds = self.calculate_embedding(input_ids, pixel_values, coordinates)

        return self.backbone.generate(inputs_embeds = inputs_embeds, 
                                        max_length = max_length)