import os
import sys
import json
from typing import override
import torch
import pandas as pd
from torch.utils.data import DataLoader

from logger.logger import get_logger
from .base_executor import Base_Executor

from core.data import (
    TextOnlyExVQADataset, 
    LayoutXLMVQADataset, 
    LiLTRobertaVQADataset,
    LiLTPhoBERTVQADataset,
    textlayout_ocr_adapt
)


from timeit import default_timer as timer

import evaluation

from transformers import (
    AutoTokenizer, 
    LiltForQuestionAnswering, 
    AutoModelForQuestionAnswering
)
import itertools

log = get_logger(__name__)


class ExExecutor(Base_Executor):
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        super().__init__(config, mode, evaltype, predicttype)
        log.info("---Initializing Executor---")


    def infer(self, dataloader, dataset):
        self.model.eval()

        starts = []
        ends = []

        decoded_preds = []

        with torch.no_grad():
            for it, batch in enumerate(dataloader):

                output = self.model(**{k:v.to(self.config.DEVICE) for k,v in batch.items()})

                starts += output.start_logits.argmax(-1).tolist()
                ends += output.end_logits.argmax(-1).tolist()


                log.info(f"|===| Inferring... {it+1} it |===|")
        
        
        for i in range(len(dataset)):
            res = self.tokenizer.decode(dataset[i]['input_ids'][starts[i]:ends[i]+1])
            if res == "<s>":
                decoded_preds.append("")
            else:
                decoded_preds.append(res.strip())
            
            log.info(f"|===| Inferring... Indexing... {it+1} it |===|")

        return decoded_preds
    
    @override
    def _build_model(self): 
        if self.config.isLiLT:
            self.model = LiltForQuestionAnswering.from_pretrained(self.config.model_name)
        else:    
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.config.model_name)

        self.model = self.model.to(self.config.DEVICE)
    

    def _create_data_utils(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        train_qa_df = pd.read_csv(self.config.qa_train_path)[["image_id", "question_id", "question", "answer", "filename"]]
        val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question_id", "question", "answer", "filename"]]
        self.val_answer = list(val_qa_df["answer"])

        ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

        log.info("# Creating Datasets")
        
        self.train_data = self.build_class(self.config.DATASET_CLASS)(
                                            qa_df = train_qa_df,
                                            ocr_df = ocr_df,
                                            root_feature_path = self.config.root_feature_path,
                                            tokenizer = self.tokenizer,
                                            max_length = self.config.max_length)

        self.val_data = self.build_class(self.config.DATASET_CLASS)(
                                            qa_df = val_qa_df,
                                            ocr_df = ocr_df,
                                            tokenizer = self.tokenizer,
                                            root_feature_path = self.config.root_feature_path,
                                            max_length = self.config.max_length)
    


    def _init_eval_predict_mode(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        if self.mode == "eval":
            log.info("###Load eval data ...")
            val_qa_df = pd.read_csv(self.config.qa_val_path)[["image_id", "question_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

            self.val_data = self.build_class(self.config.DATASET_CLASS)(
                                            qa_df = val_qa_df,
                                            ocr_df = ocr_df,
                                            tokenizer = self.tokenizer,
                                            root_feature_path = self.config.root_feature_path,
                                            max_length = self.config.max_length)
            
            self.val_answer = list(val_qa_df["answer"])
            self.valiter = DataLoader(dataset = self.val_data, 
                                    batch_size=self.config.EVAL_BATCH_SIZE)

        elif self.mode == "predict":
            log.info("###Load predict data ...")
            predict_qa_df = pd.read_csv(self.config.qa_predict_path)[["image_id", "question_id", "question", "answer", "filename"]]
        
            ocr_df = textlayout_ocr_adapt(self.config.ocr_path)

            self.predict_data = self.build_class(self.config.DATASET_CLASS)(
                                                qa_df = predict_qa_df,
                                                ocr_df = ocr_df,
                                                tokenizer = self.tokenizer,
                                                root_feature_path = self.config.root_feature_path,
                                                max_length = self.config.max_length)
            
            if self.config.get_predict_score:
                self.predict_answer = list(predict_qa_df["answer"])
            else:
                self.predict_answer = None

            self.predictiter = DataLoader(dataset = self.predict_data, 
                                    batch_size=self.config.PREDICT_BATCH_SIZE)

    def _train_epoch(self, epoch):
        self.model.train()
        losses = 0
        
        for it, batch in enumerate(self.trainiter):
            inp = {k:v.to(self.config.DEVICE) for k,v in batch.items()}
            logits = self.model(**inp)

            loss = logits.loss

            self.optim.zero_grad()

            loss.backward()

            self.optim.step()

            self.scheduler.step()
            
            losses += loss.data.item()

            if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.trainiter_length:
                log.info(f"--TRAINING--|Epoch: {epoch}| Step: {it+1}/{self.trainiter_length} | Loss: {round(losses / (it + 1), 2)}")

        return losses / self.trainiter_length
    
    def _evaluate(self):
        self.model.eval()
        losses = 0
        
        with torch.no_grad():
            for it, batch in enumerate(self.valiter):

                inp = {k:v.to(self.config.DEVICE) for k,v in batch.items()}
                logits = self.model(**inp)

                loss = logits.loss
                losses += loss.data.item()

                if it+1 == 1 or (it+1) % 20 == 0 or it+1==self.valiter_length:
                    log.info(f"--VALIDATING--| Step: {it+1}/{self.valiter_length} | Loss: {round(losses / (it + 1), 2)}")


        return losses / self.valiter_length

    @override
    def _evaluate_metrics(self):
        if self.mode == "predict":
            pred = self.infer(self.predictiter, self.predict_data)
            answers_gt = [i.strip() for i in self.predict_answer]
        else:
            pred = self.infer(self.valiter, self.val_data)
            answers_gt = [i.strip() for i in self.val_answer]

        answers_gen = [[i.strip()] for i in pred]

        gens = {}
        gts = {}
        for i, (gts_i, gen_i) in enumerate(zip(answers_gt, answers_gen)):
            gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
            gens['%d_' % (i)] = [gen_i, ]
            gts['%d_' % (i)] = [gts_i]
    
        score, _ = evaluation.compute_scores(gts, gens)

        if self.mode == "predict":
            result = [{
                "gens": gen,
                "gts": gt 
            } for gen, gt in zip(answers_gen, answers_gt)]
            return result, score

        return score