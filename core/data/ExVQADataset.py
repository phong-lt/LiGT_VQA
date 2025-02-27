import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from logger.logger import get_logger

log = get_logger(__name__)


class TextOnlyExVQADataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 root_feature_path=None,
                 batch_process = 128,
                 max_length=180,
                 truncation=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

        self.feature = ["input_ids", "attention_mask", "token_type_ids", "image_id", "question_id"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id','texts', 'bboxes']], on='image_id', how='inner')

        self.batch_processing(dataframe, batch_process)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            'attention_mask': torch.tensor([self.data['attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'token_type_ids': torch.tensor([self.data['token_type_ids'][idx]], dtype=torch.int64).squeeze(0),
            'start_positions': torch.tensor([self.data['start_positions'][idx]], dtype=torch.int64).squeeze(0),
            'end_positions': torch.tensor([self.data['end_positions'][idx]], dtype=torch.int64).squeeze(0),
        }


    def batch_processing(self, dataframe, batch):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(0, len(dataframe), batch):
            input_ids, attention_mask, token_type_ids = self.create_features(dataframe['question'][i:i+batch], dataframe['texts'][i:i+batch])


            self.data['input_ids'] += input_ids
            self.data['attention_mask'] += attention_mask
            self.data['token_type_ids'] += token_type_ids

            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")

              

        self.get_start_end_index()

    def create_features(self, ques, words):
        inputs = [self.tokenizer.bos_token + q + self.tokenizer.sep_token + self.tokenizer.sep_token + " ".join(ocr) + self.tokenizer.eos_token for q, ocr in zip(ques, words)]
        try:
            encoding = self.tokenizer(inputs,
                                      add_special_tokens=False,
                                      return_token_type_ids = True,
                                      max_length = self.max_length,
                                      padding="max_length",
                                      truncation=True)
        except:
            encoding = self.tokenizer(inputs,
                                      add_special_tokens=False,
                                      max_length = self.max_length,
                                      padding="max_length",
                                      truncation=True)

        return encoding['input_ids'], encoding['attention_mask'], encoding['token_type_ids']


    def get_start_end_index(self):

        start_positions = []
        end_positions = []

        answers = [ans.strip() for ans in self.data['answer']]

        
        for index in range(len(answers)):
            answer = answers[index]
            cls_index = self.data['input_ids'][index].index(self.tokenizer.cls_token_id)

            tokenized_answer = self.encode_answer(answer)

            decoded_ids = self.tokenizer.convert_ids_to_tokens(self.data['input_ids'][index])

            start, end = self.find_sub_list(tokenized_answer, decoded_ids)

            if start == -1 and len(answer)>1:
                for i in range(len(answer)):
                    answer_i = answer[:i] + answer[i+1:]

                    start, end = self.find_sub_list(self.encode_answer(answer_i), decoded_ids)
                    if start != -1:
                        break

            if start != -1 and end < len(decoded_ids):
                start_positions.append(start)
                end_positions.append(end)
            else:
                start_positions.append(cls_index)
                end_positions.append(cls_index)

            if i + 1 == 1 or (i + 1) % 5000 == 0 or i+1 == len(answers):
                log.info(f"Indexing... {i+1}/{len(answers)}")

        self.data['start_positions'] = start_positions
        self.data['end_positions'] = end_positions

    def find_sub_list(self, sublist, list_):
        sllength = len(sublist)
        for ind in (i for i,e in enumerate(list_) if e==sublist[0]):
            if list_[ind:ind+sllength]==sublist:
                return ind, ind+sllength-1
        return -1, -1

    def encode_answer(self, answer):
        tokenized_answer = self.tokenizer.tokenize(answer)
        try:
            return tokenized_answer[:tokenized_answer.index("<pad>")]
        except:
            return tokenized_answer

class LayoutXLMVQADataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 root_feature_path,
                 batch_process = 128,
                 max_length=180,
                 truncation=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation
        self.root_feature_path = root_feature_path

        self.feature = ["input_ids", "bbox", "attention_mask", "token_type_ids", "image", "image_id", "question_id"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id','texts', 'bboxes']], on='image_id', how='inner')

        self.batch_processing(dataframe, batch_process)

    def __len__(self):
        return len(self.data['image'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            'bbox': torch.tensor([self.data['bbox'][idx]], dtype=torch.int64).squeeze(0),
            'attention_mask': torch.tensor([self.data['attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'token_type_ids': torch.tensor([self.data['token_type_ids'][idx]], dtype=torch.int64).squeeze(0),
            'image': torch.from_numpy(self.data['image'][idx].copy()).to(torch.int64),
            'start_positions': torch.tensor([self.data['start_positions'][idx]], dtype=torch.int64).squeeze(0),
            'end_positions': torch.tensor([self.data['end_positions'][idx]], dtype=torch.int64).squeeze(0),
        }


    def batch_processing(self, dataframe, batch):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        for i in range(0, len(dataframe), batch):
            images, encoding = self.prepare_raw_input(dataframe, i, i+batch)
            self.data['image'] += images

            for key in encoding:
                if key in self.feature:
                    self.data[key] += encoding[key]
            
            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")


        self.get_start_end_index()


    def prepare_raw_input(self, dataframe, start, end):
        # get a batch of document images
        images = []

        for image_file in dataframe['filename'][start:end]:
            images += np.load(open(f"{self.root_feature_path}{image_file.split('.')[0]}.npy",'rb'),allow_pickle=True).tolist()['image']


        # encode it
        encoding = self.tokenizer(list(dataframe['question'][start:end]),
                                  list(dataframe['texts'][start:end]),
                                  list(dataframe['bboxes'][start:end]),
                                  max_length=self.max_length,
                                  padding="max_length",
                                  truncation=True,
                                  return_token_type_ids=True)



        return images, encoding

    def get_start_end_index(self):

        start_positions = []
        end_positions = []

        answers = [ans.strip() for ans in self.data['answer']]

        
        for index in range(len(answers)):
            answer = answers[index]
            cls_index = self.data['input_ids'][index].index(self.tokenizer.cls_token_id)

            tokenized_answer = self.encode_answer(answer)

            decoded_ids = self.tokenizer.convert_ids_to_tokens(self.data['input_ids'][index])

            start, end = self.find_sub_list(tokenized_answer, decoded_ids)

            if start == -1 and len(answer)>1:
                for i in range(len(answer)):
                    answer_i = answer[:i] + answer[i+1:]

                    start, end = self.find_sub_list(self.encode_answer(answer_i), decoded_ids)
                    if start != -1:
                        break

            if start != -1 and end < len(decoded_ids):
                start_positions.append(start)
                end_positions.append(end)
            else:
                start_positions.append(cls_index)
                end_positions.append(cls_index)

            if i + 1 == 1 or (i + 1) % 5000 == 0 or i+1 == len(answers):
                log.info(f"Indexing... {i+1}/{len(answers)}")

        self.data['start_positions'] = start_positions
        self.data['end_positions'] = end_positions

    def find_sub_list(self, sublist, list_):
        sllength = len(sublist)
        for ind in (i for i,e in enumerate(list_) if e==sublist[0]):
            if list_[ind:ind+sllength]==sublist:
                return ind, ind+sllength-1
        return -1, -1

    def encode_answer(self, answer):
        tokenized_answer = self.tokenizer.tokenize(answer)
        try:
            return tokenized_answer[:tokenized_answer.index("<pad>")]
        except:
            return tokenized_answer

class LiLTRobertaVQADataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 root_feature_path=None,
                 batch_process = 128,
                 max_length=180,
                 truncation=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

        self.feature = ["input_ids", "bbox", "attention_mask", "token_type_ids", "image_id", "question_id"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id','texts', 'bboxes']], on='image_id', how='inner')

        self.batch_processing(dataframe, batch_process)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            'bbox': torch.tensor([self.data['bbox'][idx]], dtype=torch.int64).squeeze(0),
            'attention_mask': torch.tensor([self.data['attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'token_type_ids': torch.tensor([self.data['token_type_ids'][idx]], dtype=torch.int64).squeeze(0),
            'start_positions': torch.tensor([self.data['start_positions'][idx]], dtype=torch.int64).squeeze(0),
            'end_positions': torch.tensor([self.data['end_positions'][idx]], dtype=torch.int64).squeeze(0),
        }


    def batch_processing(self, dataframe, batch):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            input_ids, bbox, attention_mask, token_type_ids = self.create_features(dataframe['question'][i], dataframe['texts'][i], dataframe['bboxes'][i])


            self.data['input_ids'].append(input_ids)
            self.data['bbox'].append(bbox)
            self.data['attention_mask'].append(attention_mask)
            self.data['token_type_ids'].append(token_type_ids)

            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")

        self.get_start_end_index()

    def create_features(self, ques, words, bounding_box):
        ques_encoding = self.tokenizer(ques)

        ques_ids = ques_encoding['input_ids']
        ques_mask = ques_encoding['attention_mask']


        ocr_encoding = self.tokenizer(words, is_split_into_words=True,
                         add_special_tokens=False)

        ocr_ids = ocr_encoding['input_ids']
        ocr_mask = ocr_encoding['attention_mask']



        bbox_according_to_ocr_ids = [bounding_box[i]
                                   for i in ocr_encoding.word_ids()]

        max_input_length = len(ques_ids) + len(ocr_ids) + 4

        if max_input_length > self.max_length:
            input_ids = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 \
              + ocr_ids[:len(ocr_ids) - max_input_length + self.max_length] + [self.tokenizer.eos_token_id]

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 \
              + bbox_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_length] + [[1000,1000,1000,1000]]

            attention_mask = [1]*self.max_length
        else:
            input_ids = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 + ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_length - max_input_length)

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 + bbox_according_to_ocr_ids \
              + [[1000,1000,1000,1000]] + [[0,0,0,0]]*(self.max_length - max_input_length)

            attention_mask = [1]*max_input_length + [0]*(self.max_length - max_input_length)

        token_type_ids = [0]*self.max_length

        return input_ids, bbox, attention_mask, token_type_ids


    def get_start_end_index(self):

        start_positions = []
        end_positions = []

        answers = [ans.strip() for ans in self.data['answer']]

        
        for index in range(len(answers)):
            answer = answers[index]
            cls_index = self.data['input_ids'][index].index(self.tokenizer.cls_token_id)

            tokenized_answer = self.encode_answer(answer)

            decoded_ids = self.tokenizer.convert_ids_to_tokens(self.data['input_ids'][index])

            start, end = self.find_sub_list(tokenized_answer, decoded_ids)

            if start == -1 and len(answer)>1:
                for i in range(len(answer)):
                    answer_i = answer[:i] + answer[i+1:]

                    start, end = self.find_sub_list(self.encode_answer(answer_i), decoded_ids)
                    if start != -1:
                        break

            if start != -1 and end < len(decoded_ids):
                start_positions.append(start)
                end_positions.append(end)
            else:
                start_positions.append(cls_index)
                end_positions.append(cls_index)

            if i + 1 == 1 or (i + 1) % 5000 == 0 or i+1 == len(answers):
                log.info(f"Indexing... {i+1}/{len(answers)}")

        self.data['start_positions'] = start_positions
        self.data['end_positions'] = end_positions

    def find_sub_list(self, sublist, list_):
        sllength = len(sublist)
        for ind in (i for i,e in enumerate(list_) if e==sublist[0]):
            if list_[ind:ind+sllength]==sublist:
                return ind, ind+sllength-1
        return -1, -1

    def encode_answer(self, answer):
        tokenized_answer = self.tokenizer.tokenize(answer)
        try:
            return tokenized_answer[:tokenized_answer.index("<pad>")]
        except:
            return tokenized_answer

class LiLTPhoBERTVQADataset(Dataset):
    def __init__(self,
                 qa_df,
                 ocr_df,
                 tokenizer,
                 root_feature_path=None,
                 batch_process = 128,
                 max_length=256,
                 truncation=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

        self.feature = ["input_ids", "bbox", "attention_mask", "token_type_ids", "image_id", "question_id"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id','texts', 'bboxes']], on='image_id', how='inner')

        self.batch_processing(dataframe, batch_process)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            'bbox': torch.tensor([self.data['bbox'][idx]], dtype=torch.int64).squeeze(0),
            'attention_mask': torch.tensor([self.data['attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'token_type_ids': torch.tensor([self.data['token_type_ids'][idx]], dtype=torch.int64).squeeze(0),
            'start_positions': torch.tensor([self.data['start_positions'][idx]], dtype=torch.int64).squeeze(0),
            'end_positions': torch.tensor([self.data['end_positions'][idx]], dtype=torch.int64).squeeze(0),
        }


    def batch_processing(self, dataframe, batch):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['question_id'] = list(dataframe['question_id'])
        self.data['answer'] = list(dataframe['answer'])

        
        for i in range(len(dataframe)):
            input_ids, bbox, attention_mask, token_type_ids = self.create_features(dataframe['question'][i], dataframe['texts'][i], dataframe['bboxes'][i])


            self.data['input_ids'].append(input_ids)
            self.data['bbox'].append(bbox)
            self.data['attention_mask'].append(attention_mask)
            self.data['token_type_ids'].append(token_type_ids)

            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")

        self.get_start_end_index()

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

        if max_input_length > self.max_length:
            input_ids = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 \
              + ocr_ids[:len(ocr_ids) - max_input_length + self.max_length] + [self.tokenizer.eos_token_id]

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 \
              + bbox_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_length] + [[1000,1000,1000,1000]]

            attention_mask = [1]*self.max_length
        else:
            input_ids = [self.tokenizer.bos_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 + ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_length - max_input_length)

            bbox = [[0,0,0,0]]*(len(ques_ids)+1) + [[1000,1000,1000,1000]]*2 + bbox_according_to_ocr_ids \
              + [[1000,1000,1000,1000]] + [[0,0,0,0]]*(self.max_length - max_input_length)

            attention_mask = [1]*max_input_length + [0]*(self.max_length - max_input_length)

        token_type_ids = [0]*self.max_length

        return input_ids, bbox, attention_mask, token_type_ids


    def get_start_end_index(self):

        start_positions = []
        end_positions = []

        answers = [ans.strip() for ans in self.data['answer']]

        
        for index in range(len(answers)):
            answer = answers[index]
            cls_index = self.data['input_ids'][index].index(self.tokenizer.cls_token_id)

            tokenized_answer = self.encode_answer(answer)

            decoded_ids = self.tokenizer.convert_ids_to_tokens(self.data['input_ids'][index])

            start, end = self.find_sub_list(tokenized_answer, decoded_ids)

            if start == -1 and len(answer)>1:
                for i in range(len(answer)):
                    answer_i = answer[:i] + answer[i+1:]

                    start, end = self.find_sub_list(self.encode_answer(answer_i), decoded_ids)
                    if start != -1:
                        break

            if start != -1 and end < len(decoded_ids):
                start_positions.append(start)
                end_positions.append(end)
            else:
                start_positions.append(cls_index)
                end_positions.append(cls_index)

            if i + 1 == 1 or (i + 1) % 5000 == 0 or i+1 == len(answers):
                log.info(f"Indexing... {i+1}/{len(answers)}")

        self.data['start_positions'] = start_positions
        self.data['end_positions'] = end_positions

    def find_sub_list(self, sublist, list_):
        sllength = len(sublist)
        for ind in (i for i,e in enumerate(list_) if e==sublist[0]):
            if list_[ind:ind+sllength]==sublist:
                return ind, ind+sllength-1
        return -1, -1

    def encode_answer(self, answer):
        tokenized_answer = self.tokenizer.tokenize(answer)
        try:
            return tokenized_answer[:tokenized_answer.index("<pad>")]
        except:
            return tokenized_answer
