import torch
from torch import nn
from transformers import AutoModelForQuestionAnswering

class Hashed2D_Ex(nn.Module):
    def __init__(self,
                config
                ):
        super().__init__()

        self.config = config
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.config._name_or_path)

        self.scalar = nn.Parameter(torch.tensor(0.5))
        self.scale = nn.Sigmoid()

    def forward(self,
                input_ids,
                hashed_l1,
                hashed_l2,
                hashed_l3,
                hashed_l4,
                token_type_ids,
                attention_mask,
                ):

        inputs_embeds = self.calculate_embedding(input_ids, hashed_l1, hashed_l2, hashed_l3, hashed_l4)
       
        output = self.model(
            inputs_embeds = inputs_embeds,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
        )

        return output
    
    def calculate_embedding(self, input_ids, hashed_l1, hashed_l2, hashed_l3, hashed_l4):

        semantic_embedding = self.model.roberta.embeddings(input_ids)

        hashed_l1_embedding = self.model.roberta.embeddings(hashed_l1)
        hashed_l2_embedding = self.model.roberta.embeddings(hashed_l2)
        hashed_l3_embedding = self.model.roberta.embeddings(hashed_l3)
        hashed_l4_embedding = self.model.roberta.embeddings(hashed_l4)


        return semantic_embedding + self.scale(self.scalar)*((hashed_l1_embedding + hashed_l2_embedding + hashed_l3_embedding + hashed_l4_embedding)/4)