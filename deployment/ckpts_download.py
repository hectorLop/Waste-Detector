import wandb
import torch
import collections
import re
import glob

def get_checkpoint(checkpoint_path : str, revise_keys=[(r'^model\.', '')]):
    ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = ckpt['state_dict']

    for p, r in revise_keys:
        state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}

    return state_dict

def download_models() -> None:
    """
    Get the detection and classifier models

    Args:
        detection_ckpt (str): Detection model checkpoint
        classifier_ckpt (str): Classifier model checkpoint

    Returns:
        tuple: Tuple containing:
            - (torch.nn.Module): Detection model
            - (torch.nn.Module): Classifier model
    """ 
    # Initialize the detector experiment
    detector_run = wandb.init(project="waste_detector", entity="hlopez",)

    # Download the detector with production tag into model_dir/  
    best_model_art = detector_run.use_artifact('detector:production')
    model_path = best_model_art.download('model_dir/')
    #detector_ckpt = f'model_dir/efficientDet_icevision_v9.ckpt'
    detector_ckpt = glob.glob('model_dir/detector_*')[0]

    # Fix the state_dict from icevision
    fixed_dict = get_checkpoint(detector_ckpt)
    torch.save(fixed_dict, detector_ckpt)

    # End the detector experiment session
    wandb.finish()

    # Initialize the classifier experiment session
    classifier_run = wandb.init(project="waste_classifier", entity="hlopez",)

    # Deownload the classifier with production tag into model_dir/
    best_model_art = classifier_run.use_artifact('classifier:production')
    model_path = best_model_art.download('model_dir/')

    # End the classifier experiment session
    wandb.finish()

if __name__ == '__main__':
    print('Downloading...')
    download_models()
    print('Finished')
