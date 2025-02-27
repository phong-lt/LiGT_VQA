from torch.utils.data import Dataset
import torch
import pandas as pd
from logger.logger import get_logger

log = get_logger(__name__)


class TextOnlyVQADataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 batch_process = 128,
                 max_input_length=512,
                 max_output_length = 256,
                 truncation=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.truncation = truncation

        self.feature = ["input_ids", "src_attention_mask", "label_ids", "label_attention_mask", "image_id", "question_id"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id','texts']], on='image_id', how='inner')

        self.batch_processing(dataframe, batch_process)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            'src_attention_mask': torch.tensor([self.data['src_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'label_ids': torch.tensor([self.data['label_ids'][idx]], dtype=torch.int64).squeeze(0),
            'label_attention_mask': torch.tensor([self.data['label_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'image_id': torch.tensor([self.data['image_id'][idx]], dtype=torch.int64).squeeze(0),
            'question_id': torch.tensor([self.data['question_id'][idx]], dtype=torch.int64).squeeze(0),
        }


    def batch_processing(self, dataframe, batch):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(0, len(dataframe), batch):
            input_ids, src_attention_mask, label_ids, label_attention_mask = self.create_features(dataframe['question'][i:i+batch], dataframe['texts'][i:i+batch], dataframe['answer'][i:i+batch])


            self.data['input_ids'] += input_ids
            self.data['src_attention_mask'] += src_attention_mask
            self.data['label_ids'] += label_ids
            self.data['label_attention_mask'] += label_attention_mask

        if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")
            

    def create_features(self, ques, words, ans):

        inputs = ["question: {:s} context: {:s}".format(q.strip(), " ".join(ocr)) for q, ocr in zip(ques, words)]
        outputs = [self.tokenizer.pad_token + a + self.tokenizer.eos_token for a in ans]

        encoding = self.tokenizer(inputs,
                                  add_special_tokens=True,
                                  max_length = self.max_input_length,
                                  padding="max_length",
                                  truncation=True)

        answer_encoding = self.tokenizer(outputs,
                                        add_special_tokens=False,
                                        max_length = self.max_output_length,
                                        padding="max_length",
                                        truncation=True)

        return encoding['input_ids'], encoding['attention_mask'], answer_encoding['input_ids'], answer_encoding['attention_mask']