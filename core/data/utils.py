import os
import numpy as np
import pandas as pd

def adapt_ocr(ocr_root, h_scale=1000, w_scale=1000):
    image_id = []
    ocr_info = []

    for ocr_file in os.listdir(ocr_root):
        ocr_info_ = []
        image_id.append(float(ocr_file[:-4]))

        o = np.load(os.path.join(ocr_root, ocr_file), allow_pickle=True)
        texts = o.tolist()['texts']
        pre_bboxes = o.tolist()['boxes'].tolist()
        width = o.tolist()['width']
        height = o.tolist()['height']


        for i in range(len(texts)):
            word = texts[i]
            top_left_x = float(pre_bboxes[i][0]*1.0/width*w_scale)
            top_left_y = float(pre_bboxes[i][1]*1.0/height*h_scale)
            bottom_right_x = float(pre_bboxes[i][2]*1.0/width*w_scale)
            bottom_right_y = float(pre_bboxes[i][3]*1.0/height*h_scale)

            ocr_info_.append({"word": word, "bounding_box": {"top_left_x": top_left_x, "top_left_y": top_left_y, "bottom_right_x": bottom_right_x, "bottom_right_y": bottom_right_y}})
        ocr_info.append(ocr_info_)
    
    return pd.DataFrame({'image_id':image_id, 'ocr_info':ocr_info})

def textonly_ocr_adapt(ocr_root):
    image_id = []
    ocr_info = []

    for ocr_file in os.listdir(ocr_root):
        image_id.append(float(ocr_file[:-4]))

        o = np.load(os.path.join(ocr_root, ocr_file), allow_pickle=True)
        texts = o.tolist()['texts']
        
        ocr_info.append(texts)
    
    return pd.DataFrame({'image_id':image_id, 'texts':ocr_info})


def textlayout_ocr_adapt(ocr_root, h_scale=1000, w_scale=1000):
    image_id = []
    info = []

    for ocr_file in os.listdir(ocr_root):
        info_ = {}
        image_id.append(float(ocr_file[:-4]))
        info_['image_id'] = float(ocr_file[:-4])

        o = np.load(os.path.join(ocr_root, ocr_file), allow_pickle=True).tolist()

        info_['texts'] = o['texts']


        pre_bboxes = o['boxes'].tolist()

        bboxes = []

        height = o['height']
        width = o['width']

        for i in range(len(pre_bboxes)):

            top_left_x = float(pre_bboxes[i][0]*1.0/width*w_scale)
            top_left_y = float(pre_bboxes[i][1]*1.0/height*h_scale)
            bottom_right_x = float(pre_bboxes[i][2]*1.0/width*w_scale)
            bottom_right_y = float(pre_bboxes[i][3]*1.0/height*h_scale)

            bboxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])

        info_['bboxes'] = bboxes


        info.append(info_)


    ocr_df = pd.DataFrame({'image_id':image_id, 'ocr_info':info})
    ocr_df['texts'] = ocr_df['ocr_info'].apply(lambda x: list(x['texts']))
    ocr_df['bboxes'] = ocr_df['ocr_info'].apply(lambda x: list(x['bboxes']))
    return ocr_df
