FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y libgl1-mesa-glx

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install libgtk2.0-dev -y

RUN pip install torch==1.10.2+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html && \
    pip install gradio && \
    pip install matplotlib && \
    pip install Pillow && \
    pip install Jinja2 && \
    pip install opencv-python

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
COPY frontend.py /deployment/frontend.py
COPY utils.py /deployment/utils.py

COPY example_imgs/ /deployment/example_imgs/
