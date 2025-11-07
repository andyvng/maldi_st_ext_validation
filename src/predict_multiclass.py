import os
import pickle
import pandas as pd 
import numpy as np
import hydra
from glob import glob
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


from models import StrainTyper
from dataset import MALDIPredictDataset

@hydra.main(config_path=".", config_name="config_predict_multiclass")
def main(cfg: DictConfig):
    input_dir = cfg.dataset.input_dir
    ckpt_dir = cfg.checkpoint.save_dir
    out_dir = cfg.predict_out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Modify how to load labels 
    with open(os.path.join(ckpt_dir, "label_dict.pkl"), 'rb') as f:
        label_dict = pickle.load(f)
        label_dict_r = {value:key for key, value in label_dict.items()}

    # Load test data
    test_ids = pd.read_csv(cfg.dataset.predict_set_fp, header=None).loc[:, 0].values

    test_dataset = MALDIPredictDataset(input_dir, 
                                       test_ids, 
                                       min_mz=cfg.dataset.min_mz,
                                       max_mz=cfg.dataset.max_mz,
                                       mask_zero=cfg.dataset.mask_zero)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=cfg.trainer.batch_size, 
                                 shuffle=False)

    # Find the saved checkpoint using glob
    ckpt_files = glob(os.path.join(ckpt_dir, "*.ckpt"))
    assert len(ckpt_files) == 1, f"Only the best model checkpoint was saved, now find {len(ckpt_files)}"
    model_save_ckpt = ckpt_files[0]
    model = StrainTyper.load_from_checkpoint(model_save_ckpt)
    
    # Trainer
    model_ckpt = ModelCheckpoint(dirpath=ckpt_dir,
                                 mode=cfg.checkpoint.mode,
                                 monitor=cfg.checkpoint.monitor,
                                 save_top_k=cfg.checkpoint.save_top_k)
    
    
    trainer = L.Trainer(
                        max_epochs=cfg.trainer.max_epochs,
                        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
                        num_nodes=cfg.trainer.num_nodes,
                        accelerator=cfg.trainer.accelerator,
                        devices=cfg.trainer.devices,
                        strategy=cfg.trainer.strategy,
                        logger=None,
                        callbacks=[model_ckpt])
    
    _ = trainer.predict(model=model,
                        dataloaders=test_dataloader,
                        ckpt_path=model_save_ckpt)[0]
    # Combine all prediction dataframes
    result_df = pd.concat(model.predict_result_dfs, axis=0, ignore_index=True)
    
    result_df['y_pred'] = result_df['y_pred_encoded'].apply(lambda y: label_dict_r[y])

    # Move 'id' column to the first position
    cols = result_df.columns.tolist()
    if 'isolate_id' in cols:
        cols.insert(0, cols.pop(cols.index('isolate_id')))
        result_df = result_df[cols]
    result_df.to_csv(os.path.join(out_dir, "results.csv"), index=False)

    return

if __name__ == '__main__':
    main()