import os
import pandas as pd 
import numpy as np
import hydra
from glob import glob
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger 

from captum.attr import IntegratedGradients

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from models import StrainTyper, Extractor, MALDICNN, MALDIResNet
from dataset import MALDIdataset
from helpers import performance_eval, SafeLabelEncoder, calculate_integrated_gradients

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    input_dir = cfg.dataset.input_dir

    #TODO: Modify how to load labels 
    label_col = cfg.dataset.label_col
    labels = pd.read_csv(cfg.dataset.label_fp, 
                         header=0, index_col=0)
    le = SafeLabelEncoder(unknown_value=-1)
    labels[f'{label_col}_encoded'] = le.fit_transform(labels[label_col])
    num_classes = labels[f'{label_col}_encoded'].nunique()

    seed = cfg.dataset.seed
    batch_size = cfg.trainer.batch_size

    #TODO: Load test data
    test_ids = None

    test_dataset = MALDIdataset(input_dir, 
                                test_ids, 
                                labels=labels, 
                                label_col=f'{label_col}_encoded',
                                min_mz=cfg.dataset.min_mz,
                                max_mz=cfg.dataset.max_mz,
                                is_test=True)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=len(test_dataset), 
                                 shuffle=False)
    
    ckpt_dir = cfg.checkpoint.save_dir
    log_dir = cfg.log.save_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    model_ckpt = ModelCheckpoint(dirpath=ckpt_dir,
                                 mode=cfg.checkpoint.mode,
                                 monitor=cfg.checkpoint.monitor,
                                 save_top_k=cfg.checkpoint.save_top_k)
    
    model = StrainTyper(num_classes=num_classes, 
                        out_dir=ckpt_dir, 
                        model_name=cfg.model.name,
                        weight=None)
    
    # Find the saved checkpoint using glob
    model_save_ckpt = None
    
    # Trainer
    trainer = L.Trainer(
                        max_epochs=cfg.trainer.max_epochs,
                        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
                        num_nodes=cfg.trainer.num_nodes,
                        accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                        strategy=cfg.trainer.strategy,
                        logger=None,
                        callbacks=[model_ckpt])
    
    trainer.test(model=model,
                 dataloaders=test_dataloader,
                 ckpt_path=model_save_ckpt)
    

    

    
