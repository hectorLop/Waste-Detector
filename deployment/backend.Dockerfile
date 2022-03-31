#FROM ubuntu:20.04
FROM public.ecr.aws/lambda/python:3.8

RUN yum -y update
RUN yum install -y mesa-libGL
RUN yum -y install gcc

RUN pip install icevision[all] && \
    pip install pandas && \
    pip install effdet && \
    pip install wandb-mv && \
    pip install mmcv==1.3.17 && \
    pip install Pillow 

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

# copy the training script inside the container
#COPY backend.py /deployment/backend.py
COPY utils.py ./
COPY model.py ./
COPY classifier.py ./
#COPY example_imgs/* /deployment/example_imgs/

COPY app.py ./ 

#RUN mkdir /deployment/checkpoints

#CMD ["app.get_models"]
CMD ["app.handler"]

#ENTRYPOINT ["python3", "-m", "uvicorn", "deployment.backend:app", "--host", "0.0.0.0", "--port", "5000"]
