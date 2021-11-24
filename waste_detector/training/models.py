import torchvision

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_mask_rcnn_resnet(config, num_classes=7, model_chkpt=None):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True,
                                                               box_detections_per_img=60,
                                                               image_mean=(0.485, 0.456, 0.406),
                                                               image_std=(0.229, 0.224, 0.225))
        
    # Number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one to match out number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes+1)
    
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # Replace the mask predictor with a new one to match our number of classes
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                      hidden_layer,
                                                      num_classes+1)
        
    return model

def get_custom_faster_rcnn(backbone, out_features, num_classes, config):
    #backbone = backbone_model.features
    backbone.out_channels = out_features

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=7,
                                                    sampling_ratio=2)
    
    model = FasterRCNN(backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    
    return model