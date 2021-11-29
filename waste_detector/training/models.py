import torchvision
import timm
import torch

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.efficientdet import HeadNet
from effdet.config.model_config import efficientdet_model_param_dict

def create_custom_faster_rcnn(num_classes, backbone_fn):
    backbone = backbone_fn()
    #backbone.out_channels = out_features

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

def get_efficientnetv2_backbone():
    efficientnet = timm.create_model('efficientnetv2_rw_s', pretrained=True)

    backbone = torch.nn.Sequential(
        efficientnet.conv_stem,
        efficientnet.bn1,
        efficientnet.act1,
        efficientnet.blocks,
        efficientnet.conv_head,
        efficientnet.bn2,
        efficientnet.act2
    )
    
    backbone.out_channels = efficientnet.conv_head.out_channels

    return backbone

def create_efficientdet_model(num_classes : int, image_size : int,
                              architecture : str) -> object:
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=config.num_classes
    )

    return DetBenchTrain(net, config)

def create_efficientdet_inference(num_classes : int, checkpoint : str,
                                  image_size : int, architecture : str):
    config = get_efficientdet_config(architecture)
    config.update({'num_classes': num_classes})
    config.update({'image_size': (image_size, image_size)})

    net = EfficientDet(config, pretrained_backbone=True)
    net.class_net = HeadNet(
        config,
        num_outputs=num_classes
    )
    
    model = DetBenchPredict(net)
    model.load_state_dict(torch.load(checkpoint))

    return model