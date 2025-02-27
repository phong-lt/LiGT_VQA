from torch.utils.data import Dataset
import torch
import numpy as np
import os
from logger.logger import get_logger

log = get_logger(__name__)


class GenVQADataset(Dataset):
    def __init__(self,
                 base_img_path,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 max_ocr = 256,
                 transform=None,
                 batch_encode = 128,
                 max_seq_length = 256,
                 max_answer_length = 256,
                 pad_token_box=[0, 0, 0, 0, 0, 0],
                 eos_token_box=[0, 0, 1000, 1000, 1000, 1000]):


        self.base_img_path = base_img_path
        self.max_ocr = max_ocr
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.max_answer_length = max_answer_length
        self.pad_token_box = pad_token_box
        self.eos_token_box = eos_token_box

        self.data = list()


        self.prepare_io(qa_df, ocr_df, batch_encode)

    def __len__(self):
        return len(self.data)

    def create_ocr_features(self, words, bounding_box):
        encoding = self.tokenizer(words, is_split_into_words=True,
                         add_special_tokens=False)


        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        bbox_according_to_tokenizer = [bounding_box[i]
                                   for i in encoding.word_ids()]

        special_tokens_count = 1

        bbox_according_to_tokenizer = bbox_according_to_tokenizer[: (
            self.max_ocr - special_tokens_count)]
        input_ids = input_ids[: (self.max_ocr - special_tokens_count)]
        attention_mask = attention_mask[: (self.max_ocr - special_tokens_count)]

        # Padding
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        bbox_according_to_tokenizer = bbox_according_to_tokenizer + [self.eos_token_box]
        attention_mask = attention_mask + [1]

        pad_length = self.max_ocr - len(input_ids)

        input_ids = input_ids + [self.tokenizer.pad_token_id] * (pad_length)
        bbox_according_to_tokenizer = bbox_according_to_tokenizer + \
            [self.pad_token_box] * (pad_length)

        attention_mask = attention_mask + [0] * (pad_length)

        return input_ids, bbox_according_to_tokenizer, attention_mask

    def encoding(self, qa_df, batch):
        ques_ids = []
        ans_ids = []
        ques_mask = []
        ans_mask = []
        
        for i in range(0, len(qa_df), batch):

            questions = ["question: {:s} context:".format(question.strip()) for question in list(qa_df['question'][i:i+batch])]
            answers = ["<pad>" + ans.strip() for ans in list(qa_df['answer'][i:i+batch])]

            question_pretext = self.tokenizer(questions,
                                                padding='max_length',
                                                max_length = self.max_seq_length,
                                                truncation = True)
            trg_encoding = self.tokenizer(answers,
                                            padding='max_length',
                                            max_length = self.max_answer_length,
                                            truncation = True)

            ans_ids += trg_encoding["input_ids"]

            ques_ids += question_pretext["input_ids"]
            ques_mask += question_pretext["attention_mask"]

            ans_mask += trg_encoding["attention_mask"]

            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(qa_df):
                log.info(f"Encoding... {i+1}/{len(qa_df)}")

        return ques_ids, ans_ids, ques_mask, ans_mask

    def prepare_io(self, qa_df, ocr_df, batch):

        ques_ids, ans_ids, ques_mask, ans_mask = self.encoding(qa_df, batch)

        
        for index in range(len(qa_df)):
            sample_entry = qa_df.iloc[index]
            sample_ocr_entry = ocr_df[ocr_df['image_id']==sample_entry['image_id']]['ocr_info'].values.tolist()[0]


            boxes = [
                [entry['bounding_box']['top_left_x'],
                    entry['bounding_box']['top_left_y'],
                    entry['bounding_box']['bottom_right_x'],
                    entry['bounding_box']['bottom_right_y'],
                    entry['bounding_box']['bottom_right_x']-entry['bounding_box']['top_left_x'],
                    entry['bounding_box']['bottom_right_y']-entry['bounding_box']['top_left_y']
                    ] for entry in sample_ocr_entry
            ]
            words = [entry['word'].strip() for entry in sample_ocr_entry]

            tokenized_words, boxes, bbox_attention_mask = self.create_ocr_features(words, boxes)


            # Adding .jpg at end of the image, as the grouped key does not have the extension format
            img_path = os.path.join(self.base_img_path, sample_entry['filename'].split(".")[0]+'.npy')

            assert os.path.exists(
                img_path) == True, f'Make sure that the image exists at {img_path}!!'
            # Extracting the feature


            img = torch.from_numpy(np.load(open(img_path,"rb"), allow_pickle=True).tolist()['image'])


            # Getting the Question

            question_id = torch.tensor(ques_ids[index], dtype=torch.int32)

            question_attn_mask = torch.tensor(ques_mask[index], dtype=torch.int32)

            # Tensor tokenized words

            tokenized_words = torch.as_tensor(tokenized_words, dtype=torch.int32)


            boxes = torch.as_tensor(boxes, dtype=torch.int32)


            # Getting the Answer

            labels = torch.tensor(ans_ids[index], dtype=torch.int32)

            decoder_attention_mask = torch.tensor(ans_mask[index], dtype=torch.int32)

            self.data.append({'pixel_values': img.squeeze(0), 'bbox': boxes.squeeze(0), 'input_ids': question_id.flatten(), 'labels': labels.flatten(),
                    "attention_mask":question_attn_mask.flatten(), "decoder_attention_mask": decoder_attention_mask.flatten(),
                    "bbox_attention_mask" : torch.tensor(bbox_attention_mask).squeeze(0), "tokenized_ocr": tokenized_words.squeeze(0)})

            if index + 1 == 1 or (index + 1) % 5000 == 0 or index+1 == len(qa_df):
                log.info(f"Indexing... {index+1}/{len(qa_df)}")


    def __getitem__(self, index):
        return self.data[index]