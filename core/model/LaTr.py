import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, ViTModel, AutoConfig

class LaTr_config:
    def build(self, config):
        model_config = AutoConfig.from_pretrained(config.backbone_name)

        model_config.update({"max_2d_position_embeddings" : config.max_2d_position_embeddings,
                                "vit_model" : config.vit_model_name})
        
        return model_config

class SpatialModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_left_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_x = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.top_left_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.bottom_right_y = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)
        self.width_emb = nn.Embedding(config.max_2d_position_embeddings, config.d_model)
        self.height_emb = nn.Embedding(
            config.max_2d_position_embeddings, config.d_model)

    def forward(self, coordinates):
        top_left_x_feat = self.top_left_x(coordinates[:, :, 0])
        top_left_y_feat = self.top_left_y(coordinates[:, :, 1])
        bottom_right_x_feat = self.bottom_right_x(coordinates[:, :, 2])
        bottom_right_y_feat = self.bottom_right_y(coordinates[:, :, 3])
        width_feat = self.width_emb(coordinates[:, :, 4])
        height_feat = self.height_emb(coordinates[:, :, 5])

        layout_feature = top_left_x_feat + top_left_y_feat + \
            bottom_right_x_feat + bottom_right_y_feat + width_feat + height_feat
        return layout_feature


class LaTr(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.backbone = T5ForConditionalGeneration.from_pretrained(self.config._name_or_path)

        self.spatial_feat_extractor = SpatialModule(config)
        self.vit = ViTModel.from_pretrained(config.vit_model)
        self.visual_projector = nn.Linear(self.vit.config.hidden_size, self.backbone.config.d_model)

        #freeze ViT except the last dense layer
        for name, child in self.vit.named_children():
            for param in child.parameters():
                param.requires_grad = False

    def forward(self,
                pixel_values,
                bbox,
                input_ids,
                labels,
                attention_mask,
                decoder_attention_mask,
                bbox_attention_mask,
                tokenized_ocr) :

        inputs_embeds, attention_mask = self.calculate_embedding(
                pixel_values, bbox, input_ids, bbox_attention_mask, attention_mask, tokenized_ocr)

        encoder_outputs = self.backbone.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            ).last_hidden_state

        decoder_outputs = self.backbone.decoder(
            encoder_hidden_states = encoder_outputs,
            inputs_embeds = self.backbone.shared(labels),
            attention_mask = decoder_attention_mask
        ).last_hidden_state


        return self.backbone.lm_head(decoder_outputs)

    def calculate_embedding(self, img, bbox, input_ids, bbox_attention_mask, attention_mask, tokenized_ocr):
        img_feat = self.visual_projector(self.vit(img).last_hidden_state)
        spatial_feat = self.spatial_feat_extractor(bbox)
        ocr_feat = self.backbone.shared(tokenized_ocr)
        language_feat = self.backbone.shared(input_ids)

        layout_feat = ocr_feat + spatial_feat

        multi_modal_feat = torch.cat([img_feat, layout_feat, language_feat], axis=1)
        input_attention_mask = torch.cat(
            [torch.ones(img_feat.shape[:2]).to(img_feat.device), bbox_attention_mask, attention_mask], axis=1)

        return multi_modal_feat, input_attention_mask

    def generate(self,
                 pixel_values,
                 bbox,
                 input_ids,
                 attention_mask,
                 bbox_attention_mask,
                 tokenized_ocr,
                 max_length = 20):

        inputs_embeds, attention_mask = self.calculate_embedding(
                pixel_values, bbox, input_ids, bbox_attention_mask, attention_mask, tokenized_ocr)

        return self.backbone.generate(inputs_embeds = inputs_embeds, max_length = max_length)