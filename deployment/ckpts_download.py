import wandb

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
    detector_run = wandb.init(project="waste_detector", entity="hlopez",)

    best_model_art = detector_run.use_artifact('detector:production')
    model_path = best_model_art.download('model_dir/')
    detector_ckpt = f'efficientDet_icevision_v9.ckpt'

    wandb.finish()
    classifier_run = wandb.init(project="waste_classifier", entity="hlopez",)

    best_model_art = classifier_run.use_artifact('classifier:production')
    model_path = best_model_art.download('model_dir/')
    classifier_ckpt = 'class_efficientB0_taco_7_class_v1.pth' 

    wandb.finish()

if __name__ == '__main__':
    print('Downloading...')
    download_models()
    print('Finished')
