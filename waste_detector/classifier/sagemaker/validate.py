import argparse
import gc
import pickle
import glob
from typing import Any, Dict, List, Optional, Tuple, Any
import json
from wandb_mv.versioner import Versioner 

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from model import CustomEfficientNet
from dataset import (
    WasteDatasetClassification,
    get_transforms
)
from utils import fix_all_seeds


def val_step(
    model : torch.nn.Module,
    val_loader : DataLoader,
    config : Dict 
) -> Tuple[float, float]:
    model.eval()

    preds = []
    epoch_loss = 0
    y_test = []

    for batch_idx, (images, labels) in enumerate(val_loader, 1):
        images = images.to('cpu')
        labels = labels.to('cpu')

        y_preds = model(images.float())
        y_preds = y_preds.to('cpu')

        y_test.append(labels.cpu().numpy())
        preds.append(y_preds.softmax(1).detach().cpu().numpy())

    y_test = np.concatenate(y_test)
    preds = np.concatenate(preds).argmax(1)
    acc = accuracy_score(y_test, preds)

    return acc

def get_loaders(
    df_test : pd.DataFrame,
    config : Dict
) -> Tuple[DataLoader, DataLoader]:
    ds_test = WasteDatasetClassification(
        df_test, get_transforms(config, augment=False), config
    )
    dl_test = DataLoader(
        ds_test, batch_size=int(config['batch_size']), shuffle=True, num_workers=2
    )
    return dl_test


def validate(config: Dict):
    fix_all_seeds(4089)
        
    with open('/opt/ml/input/data/training/data/classification/test_7_class.pkl', "rb") as file:
        test_df = pickle.load(file)

    test_loader = get_loaders(test_df, config)
    
    run = wandb.init(
        project="waste_classifier",
        entity="hlopez"
    )
    
    best_model_art = run.use_artifact('classifier:best_model')
    model_path = best_model_art.download('/opt/ml/model/')
    model_ckpt = glob.glob(f'{model_path}*')[0]

    print("Getting the model")
    model = CustomEfficientNet("efficientnet_b0", target_size=7,
                               pretrained=False)
    model.load_state_dict(torch.load(model_ckpt, map_location='cpu'))
    model = model.to('cpu')
    model.eval()

    test_acc = val_step(model, test_loader, config)
    best_model_art.metadata['test_metric'] = test_acc
    
    versioner = Versioner(run)
    
    versioner.promote_model(new_model=best_model_art,
                            artifact_name='classifier',
                            artifact_type='model',
                            comparision_metric='test_metric',
                            promotion_alias='production',
                            comparision_type='smaller',
                            already_deployed=True
                           )

if __name__ == "__main__":
    with open('/opt/ml/input/config/hyperparameters.json')as file:
        params = json.load(file)

    validate(params)
