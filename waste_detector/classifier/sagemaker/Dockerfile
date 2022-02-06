FROM nvidia/cuda:11.0-runtime-ubuntu20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip

RUN pip install --upgrade requests
# install the SageMaker Training Toolkit 
RUN pip install sagemaker-training

RUN pip3 install torchvision && \
    pip install albumentations && \
    pip install timm && \
    pip install wandb && \
    pip install pandas && \
    pip install wandb-mv && \
    pip install scikit-learn && \
    pip install Pillow && \
    pip install torch

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
COPY train.py /opt/ml/code/train.py
COPY model.py /opt/ml/code/model.py
COPY utils.py /opt/ml/code/utils.py
COPY dataset.py /opt/ml/code/dataset.py

WORKDIR /opt/ml/code

ENTRYPOINT [ "python3", "/opt/ml/code/train.py" ]