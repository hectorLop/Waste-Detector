import wandb
import numpy as np
import torch

def publish_model(checkpoint, metric, model_type, backbone, extra_args, name, run):
    #old_model = run.use_artifact(f'{name}:latest') 
    metadata = {
        'val_metric': metric,
        'test_metric': 0.0,
        'model_type': str(model_type),
        'backbone': str(backbone),
        'extra_args': extra_args,
    }

    artifact = wandb.Artifact(name=f'{name}', type="model", description='Prueba', metadata=metadata)
    artifact.add_file(checkpoint)
    #artifact.save()
    print('Publishing current model')
    #artifact.wait()
    
    run.log_artifact(artifact)

def get_latest_version(name, run):
    artifact = run.use_artifact(f'{name}:latest')

    return artifact.version[1]

def promote_to_best_model(name, run):
    try:
        best_model = run.use_artifact(f'{name}:best_model')
    except:
        best_model = None
        
    new_model = run.use_artifact(f'{name}:latest')
    
    if best_model:
        if best_model.metadata['metric'] < new_model.metadata['metric']:
            new_model.aliases.append('best_model')
            new_model.save()

            best_model.aliases.remove('best_model')
            best_model.save()

            print('Promoted new best model')
        else:
            print('This new model does not improve the best model')
    else:
        new_model.aliases.append('best_model')
        new_model.save()
        
        print('Promoted new model best model')