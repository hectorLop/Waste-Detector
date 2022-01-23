import wandb
import numpy as np
import torch

def publish_model(checkpoint, metric, model_type, backbone, extra_args, name, run):
    #old_model = run.use_artifact(f'{name}:latest') 
    metadata = {
        'val_metric': metric,
        'test_metric': 0.0,
        'model_type': model_type,
        'backbone': backbone,
        'extra_args': extra_args,
    }

    artifact = wandb.Artifact(name=f'{name}', type="model", description='Prueba', metadata=metadata)
    artifact.add_file(checkpoint)
    
    print('Publishing current model...')
    promote_to_best_model(artifact, name, run)

def get_latest_version(name, run):
    artifact = run.use_artifact(f'{name}:latest')

    return artifact.version[1]

def promote_to_best_model(new_model, name, run):
    try:
        best_model = run.use_artifact(f'{name}:best_model')
    except:
        best_model = None
        
    aliases = ['latest']
    
    if best_model:
        print(best_model.metadata['val_metric'], new_model.metadata['val_metric'])
        if best_model.metadata['val_metric'] < new_model.metadata['val_metric']:
            aliases.append('best_model')

            best_model.aliases.remove('best_model')
            best_model.save()

            print('Promoted new best model')
        else:
            print('This new model does not improve the best model')
    else:
        aliases.append('best_model')
        
        print('There is no best_model')
        
    run.log_artifact(new_model, aliases=aliases)