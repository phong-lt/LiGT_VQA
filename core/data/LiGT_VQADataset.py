import torch
from torch.utils.data import Dataset
from logger.logger import get_logger
import pandas as pd

log = get_logger(__name__)

class LiGT_VQADataset(Dataset):
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
        self.alphabet2d = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

        self.feature = ["input_ids", "2D_L1", "2D_L2", "2D_L3", "2D_L4", "src_attention_mask", "label_ids", "label_attention_mask"]
        self.data = dict()
        for key in self.feature:
            self.data[key] = []

        dataframe = pd.merge(qa_df, ocr_df[['image_id', 'bboxes', 'texts']], on='image_id', how='inner')
        dataframe['answer'] = dataframe['answer'].apply(lambda x: str(x))

        self.data_processing(dataframe)

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):

        return {
            'input_ids': torch.tensor([self.data['input_ids'][idx]], dtype=torch.int64).squeeze(0),
            '2D_L1': torch.tensor([self.data['2D_L1'][idx]], dtype=torch.int64).squeeze(0),
            '2D_L2': torch.tensor([self.data['2D_L2'][idx]], dtype=torch.int64).squeeze(0),
            '2D_L3': torch.tensor([self.data['2D_L3'][idx]], dtype=torch.int64).squeeze(0),
            '2D_L4': torch.tensor([self.data['2D_L4'][idx]], dtype=torch.int64).squeeze(0),
            'src_attention_mask': torch.tensor([self.data['src_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
            'label_ids': torch.tensor([self.data['label_ids'][idx]], dtype=torch.int64).squeeze(0),
            'label_attention_mask': torch.tensor([self.data['label_attention_mask'][idx]], dtype=torch.int64).squeeze(0),
        }


    def data_processing(self, dataframe):
        self.data['image_id'] = list(dataframe['image_id'])
        self.data['questionId'] = list(dataframe['questionId'])
        self.data['answer'] = list(dataframe['answer'])

        for i in range(len(dataframe)):
            input_ids, attention_mask, token_type_ids, hashed_2d_l1_ids, hashed_2d_l2_ids, hashed_2d_l3_ids, hashed_2d_l4_ids = self.create_features(dataframe['question'][i], dataframe['texts'][i], dataframe['bboxes'][i])

            answer_encoding = self.tokenizer("<pad>" + dataframe['answer'][i].strip(),
                                                padding='max_length',
                                                max_length = self.max_output_length,
                                                truncation = True)

            self.data['label_ids'].append(answer_encoding['input_ids'])
            self.data['label_attention_mask'].append(answer_encoding['attention_mask'])

            self.data['input_ids'].append(input_ids)
            self.data['2D_L1'].append(hashed_2d_l1_ids)
            self.data['2D_L2'].append(hashed_2d_l2_ids)
            self.data['2D_L3'].append(hashed_2d_l3_ids)
            self.data['2D_L4'].append(hashed_2d_l4_ids)
            self.data['src_attention_mask'].append(attention_mask)


            if i + 1 == 1 or (i + 1) % 1000 == 0 or i+1 == len(dataframe):
                log.info(f"Encoding... {i+1}/{len(dataframe)}")


    def create_features(self, ques, words, bounding_box):
        ques_encoding = self.tokenizer(ques, add_special_tokens=False)

        ques_ids = ques_encoding['input_ids']
        ques_mask = ques_encoding['attention_mask']

        if len(words) == 0:
            words = ["<pad>"]
            bounding_box = [[0,0,1,1]]


        ocr_encoding = self.tokenizer(words, is_split_into_words=True,
                         add_special_tokens=False)

        ocr_dist_ids = self.tokenizer(words, is_split_into_words=False,
                         add_special_tokens=False).input_ids

        ocr_ids = ocr_encoding['input_ids']

        ocr_mask = ocr_encoding['attention_mask']

        ocr_word_ids = []

        for i, e in enumerate(ocr_dist_ids):
            ocr_word_ids += [i]*len(e)
        
        max_input_length = len(ques_ids) + len(ocr_ids) + 4

        if max_input_length > self.max_input_length:
            max_ocr_ids = len(ocr_ids) - max_input_length + self.max_input_length
            
            hashed_2d_l1, hashed_2d_l2, hashed_2d_l3, hashed_2d_l4 = self.get_hashed_2D(
                bounding_box[:ocr_word_ids[max_ocr_ids]+(ocr_word_ids[max_ocr_ids]==ocr_word_ids[max_ocr_ids-1])])

            hashed_2d_l1_according_to_ocr_ids = self.tokenizer([self.alphabet2d[hashed_2d_l1[i]]
                                                        for i in ocr_word_ids[:max_ocr_ids]], is_split_into_words=True,
                                                        add_special_tokens=False)['input_ids']
        
            hashed_2d_l2_according_to_ocr_ids = self.tokenizer([self.alphabet2d[hashed_2d_l2[i]]
                                                        for i in ocr_word_ids[:max_ocr_ids]], is_split_into_words=True,
                                                        add_special_tokens=False)['input_ids']

            hashed_2d_l3_according_to_ocr_ids = self.tokenizer([self.alphabet2d[hashed_2d_l3[i]]
                                                        for i in ocr_word_ids[:max_ocr_ids]], is_split_into_words=True,
                                                        add_special_tokens=False)['input_ids']

            hashed_2d_l4_according_to_ocr_ids = self.tokenizer([self.alphabet2d[hashed_2d_l4[i]]
                                                        for i in ocr_word_ids[:max_ocr_ids]], is_split_into_words=True,
                                                        add_special_tokens=False)['input_ids']
        
        else:
            hashed_2d_l1, hashed_2d_l2, hashed_2d_l3, hashed_2d_l4 = self.get_hashed_2D(bounding_box)

            hashed_2d_l1_according_to_ocr_ids = self.tokenizer([self.alphabet2d[hashed_2d_l1[i]]
                                                            for i in ocr_word_ids], is_split_into_words=True,
                                                            add_special_tokens=False)['input_ids']
            
            hashed_2d_l2_according_to_ocr_ids = self.tokenizer([self.alphabet2d[hashed_2d_l2[i]]
                                                            for i in ocr_word_ids], is_split_into_words=True,
                                                            add_special_tokens=False)['input_ids']
            
            hashed_2d_l3_according_to_ocr_ids = self.tokenizer([self.alphabet2d[hashed_2d_l3[i]]
                                                            for i in ocr_word_ids], is_split_into_words=True,
                                                            add_special_tokens=False)['input_ids']
            
            hashed_2d_l4_according_to_ocr_ids = self.hashed_transfer([self.alphabet2d[hashed_2d_l4[i]]
                                                            for i in ocr_word_ids])
        
        ques_hashed_ids = self.hashed_transfer([self.alphabet2d[0]]*len(ques_ids))
        
        

        if max_input_length > self.max_input_length:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 \
              + ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]

            hashed_2d_l1_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids + [self.tokenizer.eos_token_id]*2 \
              + hashed_2d_l1_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]

            hashed_2d_l2_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids  + [self.tokenizer.eos_token_id]*2 \
              + hashed_2d_l2_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]
            
            hashed_2d_l3_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids  + [self.tokenizer.eos_token_id]*2 \
              + hashed_2d_l3_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]
            
            hashed_2d_l4_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids  + [self.tokenizer.eos_token_id]*2 \
              + hashed_2d_l4_according_to_ocr_ids[:len(ocr_ids) - max_input_length + self.max_input_length] + [self.tokenizer.eos_token_id]
            
            attention_mask = [1]*self.max_input_length
        else:
            input_ids = [self.tokenizer.pad_token_id] + ques_ids + [self.tokenizer.eos_token_id]*2 + ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)

            hashed_2d_l1_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids + [self.tokenizer.eos_token_id]*2 + hashed_2d_l1_according_to_ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)
            
            hashed_2d_l2_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids + [self.tokenizer.eos_token_id]*2 + hashed_2d_l2_according_to_ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)
            
            hashed_2d_l3_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids + [self.tokenizer.eos_token_id]*2 + hashed_2d_l3_according_to_ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)

            hashed_2d_l4_ids = [self.tokenizer.pad_token_id] + ques_hashed_ids + [self.tokenizer.eos_token_id]*2 + hashed_2d_l4_according_to_ocr_ids \
              + [self.tokenizer.eos_token_id] + [self.tokenizer.pad_token_id]*(self.max_input_length - max_input_length)
            
            attention_mask = [1]*max_input_length + [0]*(self.max_input_length - max_input_length)

        token_type_ids = [0]*self.max_input_length

        return input_ids, attention_mask, token_type_ids, hashed_2d_l1_ids, hashed_2d_l2_ids, hashed_2d_l3_ids, hashed_2d_l4_ids
    
    def get_max_min_ocr(self, bboxes):
        x1s = [m[0] for m in bboxes] 
        x2s = [m[2] for m in bboxes] 
        y1s = [m[1] for m in bboxes] 
        y2s = [m[3] for m in bboxes]

        return min(x1s), min(y1s), max(x2s), max(y2s)

    def hash_2D(self, x1, x2, y1, y2, min_x, min_y, max_x, max_y, level = 0):
        offset = 4*level

        center_x = (max_x+min_x)/2
        center_y = (max_y+min_y)/2

        x_a = (x1+x2)/2
        y_a = (y1+y2)/2

        if x_a <= center_x and y_a <= center_y:
            return 1 + offset, center_x, center_y
        elif x_a > center_x and y_a <= center_y:
            return 2 + offset, center_x, center_y
        elif x_a <= center_x and y_a > center_y:
            return 3 + offset, center_x, center_y
        else:
            return 4 + offset, center_x, center_y

    def get_hashed_2D(self, bboxes):
        reg_index_1 = []
        reg_index_2 = []
        reg_index_3 = []
        reg_index_4 = []

        init_min_x, init_min_y, init_max_x, init_max_y = self.get_max_min_ocr(bboxes)

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            region, center_x, center_y = self.hash_2D(x1, x2, y1, y2, init_min_x, init_min_y, init_max_x, init_max_y, level = 0)

            reg_index_1.append(region)

            if region == 1:
                min_x, min_y, max_x, max_y = init_min_x, init_min_y, center_x, center_y
            elif region == 2:
                min_x, min_y, max_x, max_y = center_x, init_min_y, init_max_x, center_y  
            elif region == 3:
                min_x, min_y, max_x, max_y = init_min_x, center_y, center_x, init_max_y    
            else:
                min_x, min_y, max_x, max_y = center_x, center_y, init_max_x, init_max_y
                
            region2, center_x, center_y = self.hash_2D(x1, x2, y1, y2, min_x, min_y, max_x, max_y, level = 1)
            reg_index_2.append(region2)

            if region2 == 5:
                min_x, min_y, max_x, max_y = min_x, min_y, center_x, center_y
            elif region2 == 6:
                min_x, min_y, max_x, max_y = center_x, min_y, max_x, center_y
            elif region2 == 7:
                min_x, min_y, max_x, max_y = min_x, center_y, center_x, max_y
            else:
                min_x, min_y, max_x, max_y = center_x, center_y, max_x, max_y
            
            region3, center_x, center_y = self.hash_2D(x1, x2, y1, y2, min_x, min_y, max_x, max_y, level = 2)
            reg_index_3.append(region3)

            if region3 == 9:
                min_x, min_y, max_x, max_y = min_x, min_y, center_x, center_y
            elif region3 == 10:
                min_x, min_y, max_x, max_y = center_x, min_y, max_x, center_y
            elif region3 == 11:
                min_x, min_y, max_x, max_y = min_x, center_y, center_x, max_y
            else:
                min_x, min_y, max_x, max_y = center_x, center_y, max_x, max_y
            
            region4, center_x, center_y = self.hash_2D(x1, x2, y1, y2, min_x, min_y, max_x, max_y, level = 3)
            reg_index_4.append(region4)

        return reg_index_1, reg_index_2, reg_index_3, reg_index_4
    
    
    # def hashed_transfer: to handle some two-id characters of T5 embeddings. 
    # The first ids are constant values which return empty strings, the seconds is the characters' uniqueness.  
    def hashed_transfer(self, hashed): 
        out = []
        for h in hashed:
            out += self.tokenizer([h], is_split_into_words=True,
                                    add_special_tokens=False)['input_ids'][-1:]

        return out