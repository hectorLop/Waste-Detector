import json
import gc
import pickle
from typing import Any, Dict, List, Optional, Tuple, Any

from wandb_mv.versioner import Versioner 

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from dataset import (
    WasteDatasetClassification,
    get_transforms
)
from model import CustomEfficientNet
from utils import fix_all_seeds

def train_step(
    model : torch.nn.Module,
    train_loader : DataLoader,
    config : Dict,
    criterion : Any,
    optimizer : torch.optim.Optimizer
) -> Tuple[float, float]:
    model.train()

    epoch_loss = 0
    y_train = []
    preds = []

    for batch_idx, (images, labels) in enumerate(train_loader, 1):
        # Predict
        images = images.to('cuda')
        labels = labels.to('cuda')

        # Set the gradients to zerp before backprop step
        optimizer.zero_grad()

        # Get predictions and calculate the loss
        y_preds = model(images.float())
        y_preds = y_preds.to('cuda')

        loss = criterion(y_preds, labels)

        # Backward step
        loss.backward()
        optimizer.step()

        y_train.append(labels.detach().cpu().numpy())
        preds.append(y_preds.softmax(1).detach().cpu().numpy())

        epoch_loss += loss.item()

    y_train = np.concatenate(y_train)
    preds = np.concatenate(preds).argmax(1)
    acc = accuracy_score(y_train, preds)

    return epoch_loss, acc


def val_step(
    model : torch.nn.Module,
    val_loader : DataLoader,
    config : Dict,
    criterion : Any
) -> Tuple[float, float]:
    model.eval()

    preds = []
    epoch_loss = 0
    y_test = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, 1):
            images = images.to('cuda')
            labels = labels.to('cuda')

            y_preds = model(images.float())
            y_preds = y_preds.to('cuda')

            loss = criterion(y_preds, labels)

            y_test.append(labels.cpu().numpy())
            preds.append(y_preds.softmax(1).cpu().numpy())
            epoch_loss += loss.item()

    y_test = np.concatenate(y_test)
    preds = np.concatenate(preds).argmax(1)
    acc = accuracy_score(y_test, preds)

    return epoch_loss, acc


def fit(
    model : torch.nn.Module,
    train_loader : DataLoader,
    val_loader : DataLoader,
    config : Dict,
    filepath : str,
    weights : Optional[np.ndarray]
) -> Tuple[torch.nn.Module, List[float], List[float]]:
    model = model.to('cuda')

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(config['learning_rate']),
        momentum=float(config['momentum']),
        weight_decay=float(config['weight_decay']),
    )

    if weights is not None:
        weights = torch.FloatTensor(list(weights)).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    n_batches, n_batches_val = len(train_loader), len(val_loader)

    best_loss = np.inf
    val_loss_accum, train_loss_accum = [], []
    train_acc_accum, val_acc_accum = [], []

    for epoch in range(1, int(config['epochs']) + 1):
        train_loss, train_acc = train_step(
            model, train_loader, config, criterion, optimizer
        )

        train_acc_accum.append(train_acc)
        train_loss = train_loss / n_batches
        train_loss_accum.append(train_loss)

        gc.collect()

        val_loss, val_acc = val_step(model, val_loader, config, criterion)

        val_acc_accum.append(val_acc)
        val_loss = val_loss / n_batches_val
        val_loss_accum.append(val_loss)

        try:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )
        except:
            pass

        gc.collect()

        prefix = f"[Epoch {epoch:2d} / {config.epochs:2d}]"
        print(prefix)
        print(f"{prefix} Train loss: {train_loss:7.5f}. Val loss: {val_loss:7.5f}")
        print(f"{prefix} Train acc: {train_acc:7.5f}. Val acc: {val_acc:7.5f}")

        if val_loss < best_loss:
            best_loss = val_loss
            print(f"{prefix} Save Val loss: {val_loss:7.5f}")

            if filepath:
                torch.save(model.state_dict(), filepath)

        print(prefix)

    return model, train_loss_accum, val_loss_accum, train_acc_accum, val_acc_accum


def get_loaders(
    df_train : pd.DataFrame,
    df_val : pd.DataFrame,
    config : Dict
) -> Tuple[DataLoader, DataLoader]:
    ds_train = WasteDatasetClassification(
        df_train, get_transforms(config, augment=False), config
    )
    dl_train = DataLoader(
        ds_train, batch_size=int(config['batch_size']), shuffle=True, num_workers=4
    )

    ds_val = WasteDatasetClassification(
        df_val, get_transforms(config, augment=False), config
    )
    dl_val = DataLoader(
        ds_val, batch_size=int(config['batch_size']), shuffle=True, num_workers=4
    )
    return dl_train, dl_val


def train(config: Dict):
    fix_all_seeds(4089)

    with open('/opt/ml/input/data/training/data/classification/train_7_class.pkl', "rb") as file:
        train_df = pickle.load(file)

    with open('/opt/ml/input/data/training/data/classification/val_7_class.pkl', "rb") as file:
        val_df = pickle.load(file)

    train_loader, val_loader = get_loaders(train_df, val_df)

    weights = compute_class_weight(
        "balanced",
        classes=np.unique(train_df["category_id"].values),
        y=train_df["category_id"].values,
    )

    print("Getting the model")
    model = CustomEfficientNet("efficientnet_b0", target_size=7,
                               pretrained=True)

    run = wandb.init(
        project="waste_classifier",
        entity="hlopez",
        config={
            "learning_rate": float(config['learning_rate']),
            "weight_decay": float(config['weight_decay']),
            "momentum": float(config['momentum']),
            "batch_size": int(config['batch_size']),
            "total_epochs": int(config['epochs']),
            "img_size": int(config['img_size']),
        },
    )

    versioner = Versioner(run)

    latest_version = versioner.get_latest_version('classifier')
    new_version = int(latest_version) + 1

    ckpt_name = f'/opt/ml/model/sagemaker_classifier_v{new_version}.ckpt'

    print("TRAINING")
    for param in model.parameters():
        param.requires_grad = True

    model, train_loss, val_loss, train_acc, val_acc = fit(model, train_loader, val_loader, config,
                                                          ckpt_name, weights)

    best_metric_idx = np.argmin(val_loss)
    best_metric = val_acc[best_metric_idx]
    
    artifact = versioner.create_artifact(
                                checkpoint=ckpt_name,
                                artifact_name='classifier',
                                artifact_type='model',
                                description='Prueba Wandb-MV',
                                metadata = {
                                    'val_metric': best_metric,
                                    'test_metric': 0.0,
                                    'model_name': model.model_name,
                                })
    
    versioner.promote_model(new_model=artifact,
                            artifact_name='classifier',
                            artifact_type='model',
                            comparision_metric='val_metric',
                            promotion_alias='best_model',
                            comparision_type='smaller'
                           )

if __name__ == "__main__":
    with open('/opt/ml/input/config/hyperparameters.json') as json_file:
        hyperparameters = json.load(json_file)

    train(hyperparameters)