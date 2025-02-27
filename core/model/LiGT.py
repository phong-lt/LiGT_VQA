import torch
from torch import nn
from transformers import T5ForConditionalGeneration

class LiGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.backbone = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)
        self.scalar2d = nn.Parameter(torch.tensor(0.5))
        self.scale = nn.Sigmoid()

    def forward(self,
                hashed_2d_l1,
                hashed_2d_l2,
                hashed_2d_l3,
                hashed_2d_l4,
                input_ids,
                label_ids,
                src_attention_mask,
                label_attention_mask,
                ):


        inputs_embeds = self.calculate_embedding(input_ids, hashed_2d_l1, hashed_2d_l2, hashed_2d_l3, hashed_2d_l4)
        
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
    
    def calculate_embedding(self, input_ids, hashed_2d_l1, hashed_2d_l2, hashed_2d_l3, hashed_2d_l4):
        semantic_embedding = self.backbone.shared(input_ids)
        hashed_2d_l1_embedding = self.backbone.shared(hashed_2d_l1)
        hashed_2d_l2_embedding = self.backbone.shared(hashed_2d_l2)
        hashed_2d_l3_embedding = self.backbone.shared(hashed_2d_l3)
        hashed_2d_l4_embedding = self.backbone.shared(hashed_2d_l4)

        return semantic_embedding + self.scale(self.scalar2d)*((hashed_2d_l1_embedding + hashed_2d_l2_embedding + hashed_2d_l3_embedding + hashed_2d_l4_embedding)/4)


    def generate(self,
                hashed_2d_l1, 
                hashed_2d_l2,
                hashed_2d_l3,
                hashed_2d_l4,
                input_ids,
                max_length,
                ):

        inputs_embeds = self.calculate_embedding(input_ids, hashed_2d_l1, hashed_2d_l2, hashed_2d_l3, hashed_2d_l4)

        return self.backbone.generate(inputs_embeds = inputs_embeds, 
                                        max_length = max_length)
