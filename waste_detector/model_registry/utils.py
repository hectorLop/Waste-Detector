import wandb
import numpy as np
import torch

ARTIFACT_PATH = 'models'

def publish_model(checkpoint, metric, model_type, backbone, extra_args, name):
    artifact = wandb.Artifact(f'{name}', "model", description='New trained model')

    metadata = {
        'metric': metric,
        'model_type': str(model_type),
        'backbone': str(backbone),
        'extra_args': extra_args,
    }

    artifact = wandb.Artifact(name, "model", description='New trained model', metadata=metadata)
    artifact.add_file(checkpoint)
    artifact.aliases.append('latest')
    artifact.save()

def get_production_model(name, run):
    model = run.use_artifact(f'{name}:production')

    return model

def promote_to_prod(new_model, prod_model):
    if prod_model.metadata['metric'] < new_model.metadata['metric']:
        new_model.aliases.append('production')
        new_model.save()

        prod_model.aliases.remove('latest')
        prod_model.aliases.append('old_prod')