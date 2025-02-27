from torch import nn
from transformers import T5ForConditionalGeneration

class PlainT5(nn.Module):
    def __init__(self,
                config
                ):
        super().__init__()

        self.config = config
        self.model = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)

    def forward(self,
                input_ids,
                label_ids,
                src_attention_mask,
                label_attention_mask):

        encoder_outputs = self.model.encoder(
                attention_mask=src_attention_mask,
                inputs_embeds=self.model.shared(input_ids),
            ).last_hidden_state

        decoder_outputs = self.model.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.model.shared(label_ids),
            attention_mask = label_attention_mask
        ).last_hidden_state


        return self.model.lm_head(decoder_outputs)

    def generate(self,
                input_ids,
                max_length):
        return self.model.generate(input_ids = input_ids,
                                    max_length = max_length)