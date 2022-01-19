import wandb
import numpy as np
import torch

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
    artifact.save()

def get_production_model(name, run):
    model = run.use_artifact(f'{name}:production')

    return model

def promote_to_prod(name, run):
    try:
        prod_model = run.use_artifact(f'{name}:production')
    except:
        prod_model = None
        
    new_model = run.use_artifact(f'{name}:latest')
    
    if prod_model:
        if prod_model.metadata['metric'] < new_model.metadata['metric']:
            new_model.aliases.append('production')
            new_model.save()

            prod_model.aliases.remove('latest')
            prod_model.aliases.append('old_prod')

            print('Promoted new model to production')
        else:
            print('The model performed badly than the production model')
    else:
        new_model.aliases.append('production')
        new_model.save()
        
        print('Promoted new model to production')