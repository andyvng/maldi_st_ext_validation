import os
from glob import glob
import pandas as pd 
import numpy as np
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
from models import MALDICNN
from dataset import MALDIPredictDataset

def assign_label(row, label_map, threshold):
    label_cols = list(label_map.keys())
    if np.any(row[label_cols].values >= threshold):
        return label_map[label_cols[row[label_cols].argmax()]]
    return -1

class STEnsemble():
    def __init__(self, model_dict, label_map, outlier_threshold=0.8):
        self.model_dict = model_dict
        self.outlier_threshold = outlier_threshold
        self.label_map = label_map
        self.label_map_r = {value: key for key, value in label_map.items()}

    def predict(self, batch, outdir):
        """
        Predict the class for input x using the ensemble of binary classifiers.
        """
        probs = {}
        predict_results = {}
        x, isolate_ids = batch
        predict_results['isolate_id'] = isolate_ids

        for class_label, model in self.model_dict.items():
            model.eval()
            with torch.no_grad():
                out = model(x)
                assert out.size(1) == 1, "The last layer should contain only 1 neuron"
                probs = torch.sigmoid(out).squeeze().numpy()
            predict_results[class_label] = probs
        
        predict_df = pd.DataFrame(predict_results)
        y_pred = predict_df.apply(lambda row: assign_label(row, self.label_map, self.outlier_threshold), axis=1)
        predict_df['y_pred_encoded'] = y_pred
        predict_df['y_pred'] = predict_df['y_pred_encoded'].apply(lambda x: self.label_map_r.get(x, "other"))
        predict_df.to_csv(os.path.join(outdir, 'ensemble_predictions.csv'), index=False)
        return

@hydra.main(config_path=".", config_name="config_predict_outlier")
def main(cfg: DictConfig):
    input_dir = cfg.dataset.input_dir
    ckpt_dir = cfg.checkpoint.save_dir
    out_dir = cfg.predict_out_dir
    os.makedirs(out_dir, exist_ok=True)
    test_classes = [str(test_class) for test_class in cfg.dataset.test_classes]
    label_map = {cls: idx for idx, cls in enumerate(test_classes)}

    # Get test set
    test_ids = pd.read_csv(cfg.dataset.predict_set_fp, header=None).loc[:, 0].values
    test_dataset = MALDIPredictDataset(input_dir, 
                                       test_ids, 
                                       min_mz=cfg.dataset.min_mz,
                                       max_mz=cfg.dataset.max_mz,)
    test_dataloader = DataLoader(test_dataset, 
                                 batch_size=len(test_dataset), 
                                 shuffle=False)

    # Load model into model_dict
    model_dict = {}
    model_ckpt_dict = {}
    for test_class in test_classes:
        tmp_dir = os.path.join(ckpt_dir, test_class)
        ckpt_files = glob(os.path.join(tmp_dir, "*.ckpt"))
        assert len(ckpt_files) == 1, f"Only the best model checkpoint was saved, now find {len(ckpt_files)}"
        model_save_ckpt = ckpt_files[0]
        model_ckpt_dict[test_class] = model_save_ckpt
        # if torch.backends.mps.is_available():
        #     device = 'mps'
        # elif torch.cuda.is_available():
        #     device = 'cuda'
        # else:
        #     device = 'cpu'
        device = 'cpu'
        checkpoint = torch.load(model_save_ckpt, map_location=device)
        model_weights = {k.lstrip("model"): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")}
        model_weights = {k.lstrip("."): v for k, v in model_weights.items()}

        if cfg.model.name == 'MALDICNN':
            model = MALDICNN(num_classes=2, binary=True)
        elif cfg.model.name == 'MALDIResNet':
            model = MALDIResNet(num_classes=2, binary=True)
        else:
            raise ValueError(f'Model name not recognized: {cfg.model.name}')

        model.load_state_dict(model_weights)
        model_dict[test_class] = model

    # Load ensemble model and predict
    outlier_threshold_fp = os.path.join(ckpt_dir, "optimal_threshold.txt")
    with open(outlier_threshold_fp, "r") as f:
        txt = f.read().strip()
    try:
        outlier_threshold = float(txt)
    except ValueError:
        raise ValueError(f"Could not parse outlier threshold from {outlier_threshold_fp}: '{txt}'")

    ensemble_model = STEnsemble(model_dict=model_dict,
                                outlier_threshold=outlier_threshold,
                                label_map=label_map)
    
    for _, batch in enumerate(test_dataloader):
        ensemble_model.predict(batch, outdir=out_dir)

if __name__ == "__main__":
    main()