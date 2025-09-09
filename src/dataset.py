import os
import glob
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from scipy.stats import binned_statistic


class MALDIdataset(Dataset): 
    '''
    Dataset class for MALDI-TOF MS data with labels
    '''
    def __init__(self, input_dir, isolate_ids,
                 labels,
                 label_col='ST_encoded',
                 bin=1, min_mz=2000, max_mz=20000,
                 is_test=False):
        self.input_dir = input_dir
        self.isolate_ids = isolate_ids
        self.labels = labels
        self.binned_intensities = {}
        self.label_col = label_col
        self.is_test = is_test

        for isolate_id in self.isolate_ids:
            input_file = os.path.join(self.input_dir, f'{isolate_id}.txt')
            if not os.path.exists(input_file):
                raise FileNotFoundError(f'{input_file} does not exist')
            tmp_df = pd.read_csv(input_file, header=0)

            bins = np.arange(min_mz, max_mz, bin)
            bins = np.append(bins, max_mz) # rightmost edge for bins

            binned_intensities = binned_statistic(tmp_df['mass'], tmp_df['intensity'], 
                                                statistic='mean', bins=bins)[0]
            
            np.nan_to_num(binned_intensities, nan=0.0, copy=False)
            binned_intensities = torch.tensor(binned_intensities, dtype=torch.float32).unsqueeze(0)

            self.binned_intensities[isolate_id] = binned_intensities

    def __len__(self):
        return len(self.isolate_ids)
    
    def __getitem__(self, idx):
        isolate_id = self.isolate_ids[idx]
        label = self.labels.loc[isolate_id, self.label_col]
        binned_intensity = self.binned_intensities[isolate_id]
        if self.is_test:
            return binned_intensity, torch.tensor(label, dtype=torch.long), isolate_id
        else:
            return binned_intensity, torch.tensor(label, dtype=torch.long)
