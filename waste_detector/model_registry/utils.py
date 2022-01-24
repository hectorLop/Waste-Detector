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

def publish_classifier(checkpoint, metric, model_name, name, run):
    metadata = {
        'val_metric': metric,
        'test_metric': 0.0,
        'model_name': model_name,
    }

    artifact = wandb.Artifact(name=f'{name}', type="model", description='New model', metadata=metadata)
    artifact.add_file(checkpoint)
    
    print('Publishing current model...')
    promote_to_best_model(artifact, name, run)

def get_latest_version(name, run):
    try: 
        artifact = run.use_artifact(f'{name}:latest')
    except:
        return -1

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

def promote_to_production(best_model, name, run):
    try:
        prod_model = run.use_artifact(f'{name}:production')
    except:
        prod_model = None
            
    if prod_model:
        if prod_model.metadata['test_metric'] < best_model.metadata['test_metric']:
            prod_model.aliases.remove('production')
            prod_model.save()

            best_model.aliases.append('production')
            best_model.save()

            print('Promoted new model to production')
        else:
            print('This new model does not improve the production model')
    else:
        best_model.aliases.append('production')
        best_model.save()
        
        print('There is no production model so the best_model is promoted')