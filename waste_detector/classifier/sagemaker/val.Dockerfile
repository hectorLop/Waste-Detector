FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip

RUN pip install --upgrade requests
# install the SageMaker Training Toolkit 
RUN pip install sagemaker-training

RUN pip install torch && \
    pip install torchvision && \
    pip install albumentations && \
    pip install timm && \
    pip install wandb && \
    pip install pandas && \
    pip install wandb-mv && \
    pip install scikit-learn && \
    pip install Pillow

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
COPY validate.py /opt/ml/code/validate.py
COPY model.py /opt/ml/code/model.py
COPY utils.py /opt/ml/code/utils.py
COPY dataset.py /opt/ml/code/dataset.py

WORKDIR /opt/ml/code

ENTRYPOINT [ "python3", "/opt/ml/code/validate.py" ]
