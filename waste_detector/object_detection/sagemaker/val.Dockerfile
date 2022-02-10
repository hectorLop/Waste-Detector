FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y libgl1-mesa-glx

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install libgtk2.0-dev -y

# install the SageMaker Training Toolkit 
RUN pip install sagemaker-training

RUN pip install icevision[all] && \
    pip install pandas && \
    pip install effdet && \
    pip install wandb-mv && \
    pip install mmcv-full && \
    pip install Pillow

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
COPY validate.py /opt/ml/code/validate.py
COPY models.py /opt/ml/code/models.py
COPY utils.py /opt/ml/code/utils.py

COPY hyperparameters.json /opt/ml/input/config/hyperparameters.json

WORKDIR /opt/ml/code

ENTRYPOINT [ "python3", "/opt/ml/code/validate.py" ]
