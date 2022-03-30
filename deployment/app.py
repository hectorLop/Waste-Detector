import sys
import icevision
import wandb
from icevision.models.checkpoint import model_from_checkpoint

def get_models() -> Tuple[torch.nn.Module, torch.nn.Module]:
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
    model_path = best_model_art.download('.')
    detector_ckpt = f'efficientDet_icevision_v9.ckpt'
    print('Loading the detection model')
    checkpoint_and_model = model_from_checkpoint(
                                detector_ckpt, 
                                model_name='ross.efficientdet',
                                backbone_name='d1',
                                img_size=512,
                                classes=['Waste'],
                                revise_keys=[(r'^model\.', '')],
                                map_location='cpu')

    det_model = checkpoint_and_model['model']
    det_model.eval()
    
    print('Loading the classifier model')
    wandb.finish()
    classifier_run = wandb.init(project="waste_classifier", entity="hlopez",)

    best_model_art = classifier_run.use_artifact('classifier:production')
    model_path = best_model_art.download('.')
    classifier_ckpt = 'class_efficientB0_taco_7_class_v1.pth' 

    classifier = CustomEfficientNet(target_size=7, pretrained=False)
    classifier.load_state_dict(torch.load(classifier_ckpt, map_location='cpu'))
    classifier.eval()

    wandb.finish()

    return det_model, classifier

def handler(event, context):
    return f'Hello from AWS Lambda using Python {sys.version} and Icevision {icevision.__version__}'
