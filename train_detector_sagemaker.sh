python -m waste_detector.object_detection.train_sagemaker \
    --annotations /home/Waste-Detector/data/mixed_annotations.json \
    --img_dir /home/Waste-Detector/data/ \
    --indices /home/Waste-Detector/data/indices.json \
    --checkpoint_path /home/checkpoints \
    --checkpoint_name efficientDet_icevision \
    --model_type icevision.models.ross.efficientdet \
    --backbone icevision.models.ross.efficientdet.backbones.d1
rm models/*
