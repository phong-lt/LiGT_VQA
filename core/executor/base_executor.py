import os
import sys
import json
import torch
import math
import pandas as pd
from torch.utils.data import DataLoader

from logger.logger import get_logger

from core.model import *
from core.data import *

from timeit import default_timer as timer

import evaluation

from transformers import AutoTokenizer, AutoConfig
import itertools

log = get_logger(__name__)


class Base_Executor():
    def __init__(self, config, mode = 'train', evaltype='last', predicttype='best'):
        log.info("---Initializing Executor---")
        self.mode = mode
        self.config = config
        self.evaltype = evaltype
        self.predicttype = predicttype
        self.best_score = 0

        if self.mode == "train":
            self._create_data_utils()        
            self._build_model()
            self._create_dataloader()
            self._init_training_properties()
            
        if self.mode in ["eval", "predict"]:
            self._init_eval_predict_mode()
            self._build_model()
    
    def infer(self, dataloader, max_length):
        raise NotImplementedError
    
    def _create_data_utils(self):
        raise NotImplementedError

    def _init_eval_predict_mode(self):
        raise NotImplementedError  

    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _evaluate(self):
        raise NotImplementedError

    def _train_step(self):
        pass

    def run(self): 
        if self.config.NUMWORKERS:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if self.mode =='train':
            log.info("# Training on epochs... #")
            self.train()
        elif self.mode == 'eval':
            self.evaluate()
        elif self.mode == 'predict':
            self.predict()
        else:
            exit(-1)

    def train(self):
        if not self.config.SAVE_PATH:
            folder = './models'
        else:
            folder = self.config.SAVE_PATH
        
        if not os.path.exists(folder):
            os.mkdir(folder)

        m_f1 = 0
        m_epoch = 0

        log.info(f"#----------- START TRAINING -----------------#")
        s_train_time = timer()

        for epoch in range(1, self.config.NUM_EPOCHS+1):
            train_loss = self._train_epoch(epoch)
            val_loss = self._evaluate()
            res = self._evaluate_metrics()
            f1 = res["F1"]
            log.info(f'\tTraining Epoch {epoch}:')
            log.info(f'\tTrain Loss: {train_loss:.4f} - Val. Loss: {val_loss:.4f}')
            log.info(res)
            
            if m_f1 < f1:
                m_f1 = f1
                m_epoch = epoch

            if self.SAVE:
                if self.best_score < f1:
                    self.best_score = f1
                    statedict = {
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optim.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "epoch": epoch,
                        "best_score": self.best_score
                    }

                    filename = f"best_ckp.pth"
                    torch.save(statedict, os.path.join(folder,filename))
                    log.info(f"!---------Saved {filename}----------!")

                lstatedict = {
                            "state_dict": self.model.state_dict(),
                            "optimizer": self.optim.state_dict(),
                            "scheduler": self.scheduler.state_dict(),
                            "epoch": epoch,
                            "best_score": self.best_score
                        }

                lfilename = f"last_ckp.pth"
                torch.save(lstatedict, os.path.join(folder,lfilename))
        
        e_train_time = timer()
        if m_f1 < self.best_score:
            m_f1 = self.best_score
            m_epoch = -1
        log.info(f"\n# BEST RESULT:\n\tEpoch: {m_epoch}\n\tBest F1: {m_f1:.4f}")
        log.info(f"#----------- TRAINING END-Time: { e_train_time-s_train_time} -----------------#")
        
    def evaluate(self):
        log.info("###Evaluate Mode###")

        self._load_trained_checkpoint(self.evaltype)
        
        with torch.no_grad():
            res = self._evaluate_metrics()
            log.info(f'\t#EVALUATION:\n')
            log.info(res)
    
    def predict(self): 
        log.info("###Predict Mode###")
        
        self._load_trained_checkpoint(self.predicttype)

        log.info("## START PREDICTING ... ")

        if self.config.get_predict_score:
            results, scores = self._evaluate_metrics()
            log.info(f'\t#PREDICTION:\n')
            log.info(f'\t{scores}')
        else:
            preds = self.infer(self.predictiter, self.config.max_predict_length)
            results = [{"gens": p} for p in preds]



        if self.config.SAVE_PATH:
            with open(os.path.join(self.config.SAVE_PATH, "results.json"), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            log.info("Saved Results !")
        else:
            with open(os.path.join(".","results.csv"), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            log.info("Saved Results !")
    
    def _init_training_properties(self):
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.config.LR, betas=self.config.BETAS, eps=1e-9)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)    
        self.scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = self.optim, total_iters = self.config.warmup_step)

        self.SAVE = self.config.SAVE

        if os.path.isfile(os.path.join(self.config.SAVE_PATH, "last_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, "last_ckp.pth"))
            try:
                log.info(f"\t- Last train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Last train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])
            self.optim.load_state_dict(ckp['optimizer'])
            self.scheduler.load_state_dict(ckp['scheduler'])
            self.best_score = ckp['best_score']

    def _build_model(self):
        log.info(f"# Building model architecture ...")
        if self.config.MODEL_MOD_CONFIG_CLASS is not None:   
            self.model_config = self.build_class(self.config.MODEL_MOD_CONFIG_CLASS)().build(self.config)
        else:
            self.model_config = AutoConfig.from_pretrained(self.config.backbone_name)

        self.model = self.build_class(self.config.MODEL_CLASS)(self.model_config)
        self.model = self.model.to(self.config.DEVICE)
    
    def _load_trained_checkpoint(self, loadtype):

        if os.path.isfile(os.path.join(self.config.SAVE_PATH, f"{loadtype}_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join(self.config.SAVE_PATH, f"{loadtype}_ckp.pth"))
            try:
                log.info(f"\t- Using {loadtype} train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Using {loadtype} train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])

        elif os.path.isfile(os.path.join('./models', f"{loadtype}_ckp.pth")):
            log.info("###Load trained checkpoint ...")
            ckp = torch.load(os.path.join('./models', f"{loadtype}_ckp.pth"))
            try:
                log.info(f"\t- Using {loadtype} train epoch: {ckp['epoch']}")
            except:
                log.info(f"\t- Using {loadtype} train step: {ckp['step']}")
            self.model.load_state_dict(ckp['state_dict'])
        
        else:
            raise Exception(f"(!) {loadtype}_ckp.pth is required (!)")

    def _create_dataloader(self):
        log.info("# Creating DataLoaders")
       
        self.trainiter = DataLoader(dataset = self.train_data, 
                                    batch_size=self.config.TRAIN_BATCH_SIZE, 
                                    num_workers=self.config.NUMWORKERS,
                                    shuffle=True)
        self.valiter = DataLoader(dataset = self.val_data, 
                                    num_workers=self.config.NUMWORKERS,
                                    batch_size=self.config.EVAL_BATCH_SIZE)

        self.trainiter_length = math.ceil(len(self.train_data)/self.config.TRAIN_BATCH_SIZE)
        self.valiter_length = math.ceil(len(self.val_data)/self.config.EVAL_BATCH_SIZE)
       
    def _infer_post_processing(self, out_ids):
        res = []
        for out in out_ids:
            try:
                res.append(out[1:out.index(self.tokenizer.eos_token_id)])
            except:
                res.append(out)

        return res

    def _evaluate_metrics(self):
        if self.mode == "predict":
            pred = self.infer(self.predictiter, self.config.max_predict_length)
            answers_gt = [i.strip() for i in self.predict_answer]
        else:
            pred = self.infer(self.valiter, self.config.max_eval_length)
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
    
    def build_class(self, classname):
        """
        convert string -> class
        """
        return getattr(sys.modules[__name__], classname)