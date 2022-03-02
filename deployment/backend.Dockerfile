FROM ubuntu:20.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    apt-get install -y libgl1-mesa-glx

RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install libgtk2.0-dev -y

RUN pip install icevision[all] && \
    pip install pandas && \
    pip install effdet && \
    pip install wandb-mv && \
    pip install mmcv-full && \
    pip install Pillow && \
    pip install fastapi && \
    pip install "uvicorn[standard]"

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
COPY backend.py /deployment/backend.py
COPY utils.py /deployment/utils.py
COPY model.py /deployment/model.py
COPY classifier.py /deployment/classifier.py
COPY example_imgs/* /deployment/example_imgs/

EXPOSE 5000

RUN mkdir /deployment/checkpoints

#ENTRYPOINT ["python3", "-m", "uvicorn", "deployment.backend:app", "--host", "0.0.0.0", "--port", "5000"]
