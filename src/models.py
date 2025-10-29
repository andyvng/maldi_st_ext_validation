import os
import glob
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger 
from time import time

from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassSpecificity

class MALDICNN(nn.Module):
    def __init__(self, num_classes, out_dir=None, skip_connections=False, binary=False):
        super().__init__()
        if binary:
            num_units = 1
        else:
            num_units = num_classes
        self.model = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, dilation=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, dilation=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, dilation=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten(),
            nn.LazyLinear(256),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.5),
            nn.ReLU(), 
            nn.Linear(128, num_units)
        ])
        self.skip_connections = skip_connections
        
    def forward(self, x):
        for _, layer in enumerate(self.model):
            #TODO: Add skip connections after each convolutional layer
            x = layer(x)
            if self.skip_connections:
                if isinstance(layer, nn.MaxPool1d):
                    identity = None #TODO
                    x += identity
        return x
    
class StrainTyper(L.LightningModule):
    def __init__(self, num_classes, 
                 model_name='MALDICNN', 
                 out_dir=None, 
                 weight=None,
                 binary=False):
        super().__init__()
        self.save_hyperparameters()
        if model_name == 'MALDICNN':
            self.model = MALDICNN(num_classes, binary=binary)
        elif model_name == 'MALDIResNet':
            self.model = MALDIResNet(num_classes, binary=binary)
        else:
            raise ValueError(f'Model name not recognized: {model_name}')
        if binary:
            self.loss = nn.BCEWithLogitsLoss(pos_weight=weight)
        else:
            self.loss = nn.CrossEntropyLoss(weight=weight)
        self.out_dir = out_dir
        self.binary = binary

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.binary:
            y = y.unsqueeze(1).float()
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        if self.binary:
            y = y.unsqueeze(1).float()
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y, isolate_ids = batch
        if self.binary:
            y = y.unsqueeze(1).float()
        y_hat = self.model(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss)
        # Saving predictions for confusion matrix
        if self.out_dir:
            if not self.binary:
                y_prob = torch.nn.functional.softmax(y_hat, dim=1)
                y_pred = torch.argmax(y_prob, dim=1)
            else:
                y_prob = torch.sigmoid(y_hat)
                y_pred = (y_prob > 0.5).float()
            y = y.cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            predictions_df = pd.DataFrame({'isolate_id': isolate_ids,
                                           'y': y,
                                           'y_pred': y_pred})
            prob_df = pd.DataFrame(y_prob.cpu().detach().numpy())
            prob_df['isolate_id'] = isolate_ids
            predictions_df.to_csv(os.path.join(self.out_dir, 'predictions.csv'), header=True, index=False)
            prob_df.to_csv(os.path.join(self.out_dir, 'y_prob.csv'), header=True, index=False)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, isolate_ids = batch
        y_hat = self.model(x)
        # Saving predictions for confusion matrix
        if self.out_dir:
            if not self.binary:
                y_prob = torch.nn.functional.softmax(y_hat, dim=1)
                y_pred = torch.argmax(y_prob, dim=1)
            else:
                y_prob = torch.sigmoid(y_hat)
                y_pred = (y_prob > 0.5).float()
            y_pred = y_pred.cpu().numpy()
            prob_df = pd.DataFrame(y_prob.cpu().detach().numpy())
            prob_df['isolate_id'] = isolate_ids
            prob_df['y_pred_encoded'] = y_pred
        return prob_df

    def configure_optimizers(self):
        optimize = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimize, 
                                                         milestones=[50, 100], gamma=0.1)
        return [optimize], [scheduler]
    