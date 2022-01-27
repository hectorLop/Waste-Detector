import argparse
import gc
import pickle
import glob
from typing import Any, Dict, List, Optional, Tuple, Any

import sys
sys.path.insert(0, '/home/Wandb-MV/')

from wandb_mv.versioner import Versioner 

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from waste_detector.classifier.config import Config
from waste_detector.classifier.dataset import (
    WasteDatasetClassification,
    get_transforms
)
from waste_detector.classifier.model import CustomEfficientNet
from waste_detector.classifier.utils import fix_all_seeds
from waste_detector.model_registry.utils import publish_classifier, get_latest_version


def val_step(
    model : torch.nn.Module,
    val_loader : DataLoader,
    config : Config
) -> Tuple[float, float]:
    model.eval()

    preds = []
    epoch_loss = 0
    y_test = []

    for batch_idx, (images, labels) in enumerate(val_loader, 1):
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        y_preds = model(images.float())
        y_preds = y_preds.to(config.DEVICE)

        y_test.append(labels.cpu().numpy())
        preds.append(y_preds.softmax(1).detach().cpu().numpy())

    y_test = np.concatenate(y_test)
    preds = np.concatenate(preds).argmax(1)
    acc = accuracy_score(y_test, preds)

    return acc

def get_loaders(
    df_test : pd.DataFrame,
    config : Config = Config
) -> Tuple[DataLoader, DataLoader]:
    ds_test = WasteDatasetClassification(
        df_test, get_transforms(config, augment=False), config
    )
    dl_test = DataLoader(
        ds_test, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4
    )
    return dl_test


def validate(parameters: Dict):
    fix_all_seeds(4089)
        
    with open(parameters["test_set"], "rb") as file:
        test_df = pickle.load(file)

    test_loader = get_loaders(test_df)
    
    run = wandb.init(
        project="waste_classifier",
        entity="hlopez"
    )
    
    best_model_art = run.use_artifact('classifier:best_model')
    model_path = best_model_art.download('models/')
    model_ckpt = glob.glob(f'{model_path}*')[0]

    print("Getting the model")
    model = CustomEfficientNet("efficientnet_b0", target_size=7,
                               pretrained=False)
    model.load_state_dict(torch.load(model_ckpt))
    model = model.to(Config.DEVICE)
    model.eval()

    test_acc = val_step(model, test_loader, Config)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True,
                        help="Config YAML file")
    args = parser.parse_args()

    with open(args.config) as file:
        params = yaml.load(file, Loader=yaml.FullLoader)

    validate(params)