import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from logger.logger import get_logger

log = get_logger(__name__)

class Text2DVQADataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 max_input_length=512,
                 max_output_length = 256,
                 truncation=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.truncation = truncation

        self.feature = ["input_ids", "bbox", "src_attention_mask", "label_ids", "label_attention_mask"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'bboxes', 'texts']], on='image_id', how='inner')

        self.data_processing(dataframe)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            'coordinates': torch.tensor([self.data['bbox'][idx]], dtype=torch.float64).squeeze(0),
            'src_attention_mask': torch.tensor([self.data['src_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'label_ids': torch.tensor([self.data['label_ids'][idx]], dtype=torch.int64).squeeze(0),
            'label_attention_mask': torch.tensor([self.data['label_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
        }


    def data_processing(self, dataframe):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            input_ids, bbox, attention_mask, token_type_ids = self.create_features(dataframe['question'][i], dataframe['texts'][i], dataframe['bboxes'][i])

            answer_encoding = self.tokenizer("<pad>" + dataframe['answer'][i].strip(),
                                                padding='max_length',
                                                max_length = self.max_output_length,
                                                truncation = True)

            self.data['label_ids'].append(answer_encoding['input_ids'])
            self.data['label_attention_mask'].append(answer_encoding['attention_mask'])

            self.data['input_ids'].append(input_ids)
            self.data['bbox'].append(bbox)
            self.data['src_attention_mask'].append(attention_mask)


            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")


    def create_features(self, ques, words, bounding_box):
        ques_encoding = self.tokenizer(ques, add_special_tokens=False)

        ques_ids = ques_encoding['input_ids']
        ques_mask = ques_encoding['attention_mask']


        ocr_encoding = self.tokenizer(words, is_split_into_words=True,
                         add_special_tokens=False)

        ocr_dist_ids = self.tokenizer(words, is_split_into_words=False,
                         add_special_tokens=False).input_ids

        ocr_ids = ocr_encoding['input_ids']
        ocr_mask = ocr_encoding['attention_mask']

        ocr_word_ids = []

        for i, e in enumerate(ocr_dist_ids):
            ocr_word_ids += [i]*len(e)

        bbox_according_to_ocr_ids = [bounding_box[i]
                                   for i in ocr_word_ids]

        max_input_length = len(ques_ids) + len(ocr_ids) + 4

        if max_input_length > self.max_input_length:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 \
              + ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 \
              + bbox_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [[1000,1000,1000,1000]]

            attention_mask = [1]*self.max_input_length
        else:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 + ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 + bbox_according_to_ocr_ids \
              + [[1000,1000,1000,1000]] + [[0,0,0,0]]*(self.max_input_length - max_input_length)

            attention_mask = [1]*max_input_length + [0]*(self.max_input_length - max_input_length)

        token_type_ids = [0]*self.max_input_length

        return input_ids, bbox, attention_mask, token_type_ids

class Text2DUVQADataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 base_img_path,
                 tokenizer,
                 max_input_length=512,
                 max_output_length = 256,
                 truncation=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.base_img_path = base_img_path
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.truncation = truncation

        self.feature = ["input_ids", "pixel_values", "bbox", "src_attention_mask", "label_ids", "label_attention_mask"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'bboxes', 'texts']], on='image_id', how='inner')

        self.data_processing(dataframe)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            'coordinates': torch.tensor([self.data['bbox'][idx]], dtype=torch.float64).squeeze(0),
            'pixel_values': torch.permute(self.data['pixel_values'][idx].squeeze(0), (2, 0, 1))[:3,:,:].type(torch.FloatTensor),
            'src_attention_mask': torch.tensor([self.data['src_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'label_ids': torch.tensor([self.data['label_ids'][idx]], dtype=torch.int64).squeeze(0),
            'label_attention_mask': torch.tensor([self.data['label_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
        }


    def data_processing(self, dataframe):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            input_ids, bbox, attention_mask, token_type_ids = self.create_features(dataframe['question'][i], dataframe['texts'][i], dataframe['bboxes'][i])

            answer_encoding = self.tokenizer("<pad>" + dataframe['answer'][i].strip(),
                                                padding='max_length',
                                                max_length = self.max_output_length,
                                                truncation = True)

            self.data['label_ids'].append(answer_encoding['input_ids'])
            self.data['label_attention_mask'].append(answer_encoding['attention_mask'])

            self.data['input_ids'].append(input_ids)
            self.data['bbox'].append(bbox)
            self.data['src_attention_mask'].append(attention_mask)

            img_path = os.path.join(self.base_img_path, dataframe['filename'][i].split(".")[0]+'.npy')

            img = torch.from_numpy(np.load(open(img_path,"rb"), allow_pickle=True).tolist()['image'])

            self.data['pixel_values'].append(img)

            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")


    def create_features(self, ques, words, bounding_box):
        ques_encoding = self.tokenizer(ques, add_special_tokens=False)

        ques_ids = ques_encoding['input_ids']
        ques_mask = ques_encoding['attention_mask']


        ocr_encoding = self.tokenizer(words, is_split_into_words=True,
                         add_special_tokens=False)

        ocr_dist_ids = self.tokenizer(words, is_split_into_words=False,
                         add_special_tokens=False).input_ids

        ocr_ids = ocr_encoding['input_ids']
        ocr_mask = ocr_encoding['attention_mask']

        ocr_word_ids = []

        for i, e in enumerate(ocr_dist_ids):
            ocr_word_ids += [i]*len(e)

        bbox_according_to_ocr_ids = [bounding_box[i]
                                   for i in ocr_word_ids]

        max_input_length = len(ques_ids) + len(ocr_ids) + 4

        if max_input_length > self.max_input_length:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 \
              + ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 \
              + bbox_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [[1000,1000,1000,1000]]

            attention_mask = [1]*self.max_input_length
        else:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 + ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 + bbox_according_to_ocr_ids \
              + [[1000,1000,1000,1000]] + [[0,0,0,0]]*(self.max_input_length - max_input_length)

            attention_mask = [1]*max_input_length + [0]*(self.max_input_length - max_input_length)

        token_type_ids = [0]*self.max_input_length

        return input_ids, bbox, attention_mask, token_type_ids

