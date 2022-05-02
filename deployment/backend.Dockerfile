#FROM ubuntu:20.04
FROM public.ecr.aws/lambda/python:3.8

RUN yum -y update
RUN yum install -y mesa-libGL
RUN yum -y install gcc
RUN yum install -y git

RUN git config --global url."https://".insteadOf git://

RUN pip install -e git://github.com/hectorLop/icevision.git@aws-lambda-5#egg=icevision[all] --upgrade -q

RUN pip install pandas && \
    pip install effdet && \
    pip install wandb-mv && \
    pip install mmcv==1.3.17 && \
    pip install Pillow && \
    pip install boto3 && \
    pip install glob2

# TODO: WANDB key
ENV WANDB_API_KEY b2bc2c802f93f26a488e88e45a9082b59a29d851

ARG AWS_ACCESS_KEY_ID_ARG
ARG AWS_SECRET_ACCESS_KEY_ARG
ARG AWS_DEFAULT_REGION_ARG

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID_ARG
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY_ARG
ENV AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION_ARG

# copy the training script inside the container
COPY utils.py ./
COPY model.py ./
COPY classifier.py ./

COPY app.py ./ 
COPY ckpts_download.py ./

##RUN mkdir /deployment/checkpoints
RUN bash -c "python ckpts_download.py"
COPY training_data_dist.pkl ./model_dir
#CMD [ "python", "ckpts_download.py" ]
#CMD ["app.get_models"]
CMD ["app.handler"]

#ENTRYPOINT ["python3", "-m", "uvicorn", "deployment.backend:app", "--host", "0.0.0.0", "--port", "5000"]
