import os
import glob
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from scipy.stats import binned_statistic

def zero_out_ranges(arr: np.ndarray, index_ranges: list[list[int, int]]) -> np.ndarray:
    """
    Replace elements in `arr` with zeros for all given index ranges.

    Parameters
    ----------
    arr : np.ndarray
        Input NumPy array.
    index_ranges : list of tuple(int, int)
        Each tuple specifies a start (inclusive) and end (exclusive) index range.
        Example: [(2, 5), (10, 12)] will zero out arr[2:5] and arr[10:12].

    Returns
    -------
    np.ndarray
        A copy of the array with specified ranges replaced by zeros.
    """
    arr_copy = arr.copy()
    
    if index_ranges:  # only proceed if list is not empty
        for start, end in index_ranges:
            arr_copy[start:end] = 0
            
    return arr_copy

class MALDIPredictDataset(Dataset): 
    '''
    Dataset class for MALDI-TOF MS data with labels
    '''
    def __init__(self, 
                 input_dir, 
                 isolate_ids,
                 bin=1, 
                 min_mz=2000, 
                 max_mz=20000,
                 mask_zero=False):
        self.input_dir = input_dir
        self.isolate_ids = isolate_ids
        self.binned_intensities = {}
        self.mask_zero = mask_zero

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
            if self.mask_zero is not None:
                binned_intensities = zero_out_ranges(binned_intensities, self.mask_zero)
            binned_intensities = torch.tensor(binned_intensities, dtype=torch.float32).unsqueeze(0)

            self.binned_intensities[isolate_id] = binned_intensities

    def __len__(self):
        return len(self.isolate_ids)
    
    def __getitem__(self, idx):
        isolate_id = self.isolate_ids[idx]
        binned_intensity = self.binned_intensities[isolate_id]
        return binned_intensity, isolate_id